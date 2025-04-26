#include "utils.h"
#include <tiny-cuda-nn/encodings/composite.h>

class TwoHeadedNetwork {
public:
	TwoHeadedNetwork()						   = default;
	TwoHeadedNetwork(const TwoHeadedNetwork &) = delete;

	TwoHeadedNetwork(const json &config, const uint32_t n_input_dims, const uint32_t dims_output) {

		m_n_input_dims = n_input_dims;
		m_dims_output  = dims_output;

		json encoding_opts	= config.value("encoding", json::object());
		json loss_opts		= config.value("loss", json::object());
		json optimizer_opts = config.value("optimizer", json::object());
		json network_opts	= config.value("net_base", json::object());
		json net1_opts		= config.value("net1", json::object());
		json net2_opts		= config.value("net2", json::object());

		assert(network_opts["n_output_dims"] == M_dims_transfer);
		assert(net1_opts["n_input_dims"] == M_dims_transfer);
		assert(net2_opts["n_input_dims"] == M_dims_transfer);

		net1_opts["n_output_dims"] = dims_output;
		net2_opts["n_output_dims"] = dims_output;

		// base_net [no loss]
		m_net_base = std::make_shared<NetworkWithInputEncoding<precision_t>>(
			n_input_dims, M_dims_transfer, encoding_opts, network_opts);
		m_optimizer_base.reset(create_optimizer<precision_t>(optimizer_opts));
		m_trainer_base = std::make_shared<Trainer<float, precision_t, precision_t>>(
			m_net_base, m_optimizer_base, nullptr);

		json net_head_opts[2] = {net1_opts, net2_opts};

		for (int net_idx = 0; net_idx < 2; ++net_idx) {
			// net1 & net2
			m_net_heads[net_idx].reset(create_network<precision_t>(net_head_opts[net_idx]));
			m_optimizer_heads[net_idx].reset(create_optimizer<precision_t>(optimizer_opts));
			m_loss_heads[net_idx].reset(create_loss<precision_t>(loss_opts));
			m_trainer_heads[net_idx] =
				std::make_shared<Trainer<precision_t, precision_t, precision_t>>(
					m_net_heads[net_idx], m_optimizer_heads[net_idx], m_loss_heads[net_idx]);
		}
	}

	TwoHeadedNetwork &operator=(const TwoHeadedNetwork &) = delete;
	~TwoHeadedNetwork() {}

	void print_info() const {
		size_t net_params = 0;
		net_params += m_net_base->n_params();
		net_params += m_net_heads[0]->n_params();
		net_params += m_net_heads[1]->n_params();

		size_t trainer_params = 0;
		trainer_params += m_trainer_base->n_params();
		trainer_params += m_trainer_heads[0]->n_params();
		trainer_params += m_trainer_heads[1]->n_params();

		size_t optimizer_params = 0;
		optimizer_params += m_optimizer_base->n_weights();
		optimizer_params += m_optimizer_heads[0]->n_weights();
		optimizer_params += m_optimizer_heads[1]->n_weights();

		std::cout << "Network has " << net_params << " parameters." << std::endl
				  << "Trainer has " << trainer_params << " parameters." << std::endl
				  << "Optimizer has " << optimizer_params << " parameters." << std::endl;
	}

	void set_training_batch_size(const uint32_t batch_size) {
		if (m_training_batch_size == batch_size) {
			return;
		}

		m_training_batch_size = batch_size;
		m_transfer_buffer_y.resize(M_dims_transfer, batch_size);
		m_transfer_buffer_dL_dy.resize(M_dims_transfer, batch_size);
		m_transfer_buffer_dL_dy2.resize(M_dims_transfer, batch_size);
	}

	void set_inference_batch_size(const uint32_t batch_size) {
		if (m_inference_batch_size == batch_size) {
			return;
		}
		m_inference_batch_size = batch_size;
		m_transfer_buffer_inference.resize(M_dims_transfer, batch_size);
	}

	void inference(cudaStream_t stream, const GPUMatrix<float> &input, GPUMatrix<float> &output1,
				   GPUMatrix<float> &output2) {
		set_inference_batch_size(input.cols());

		m_net_base->inference_mixed_precision(stream, input, m_transfer_buffer_inference);
		m_net_heads[0]->inference(stream, m_transfer_buffer_inference, output1);
		m_net_heads[1]->inference(stream, m_transfer_buffer_inference, output2);
	}

	float training_step(cudaStream_t stream, const GPUMatrix<float> &input,
						const GPUMatrix<float> &target1, const GPUMatrix<float> &target2,
						const bool get_loss) {

		float ret = 0.0f;

		const float loss_scale = default_loss_scale<precision_t>();

		std::unique_ptr<Trainer<float, precision_t, precision_t>::ForwardContext> ctx_base;
		std::unique_ptr<Trainer<precision_t, precision_t, precision_t>::ForwardContext> ctx_head1;
		std::unique_ptr<Trainer<precision_t, precision_t, precision_t>::ForwardContext> ctx_head2;
		{
			// Execute forward and backward in a CUDA graph for maximum performance.
			auto capture_guard = m_graph.capture_guard(stream);

			// [STEP 1] forward pass

			// [STEP 1.1] net_base forward
			// [notes]
			// (1) now we use external dL_dy, so we don't use target in forward()
			// (2) here only set m_transfer_buffer_dL_dy to ctx_base->dL_doutput, no computation
			static GPUMatrix<float> target_tmp;
			ctx_base = m_trainer_base->forward(stream, loss_scale, input, target_tmp, nullptr,
											   false, false, &m_transfer_buffer_dL_dy);

			// [STEP 1.2] net1 & net2 forward
			ctx_head1 = m_trainer_heads[0]->forward(stream, loss_scale, ctx_base->output, target1,
													nullptr, false, false, nullptr);
			ctx_head2 = m_trainer_heads[1]->forward(stream, loss_scale, ctx_base->output, target2,
													nullptr, false, false, nullptr);

			// [STEP 2] backward pass
			// [STEP 2.1] net1 & net2 backward
			m_trainer_heads[0]->backward(stream, *ctx_head1, ctx_base->output,
										 &m_transfer_buffer_dL_dy, false, GradientMode::Overwrite);
			m_trainer_heads[1]->backward(stream, *ctx_head2, ctx_base->output,
										 &m_transfer_buffer_dL_dy2, false, GradientMode::Overwrite);

			// [STEP 2.2] accumulate the gradients
			linear_kernel(calculate_gradient_sum, 0, stream,
						  M_dims_transfer * m_training_batch_size, m_transfer_buffer_dL_dy.data(),
						  m_transfer_buffer_dL_dy2.data());

			// [STEP 2.3] net_base backward
			m_trainer_base->backward(stream, *ctx_base, input, nullptr, false,
									 GradientMode::Overwrite);

			// [STEP 3] optimizer step
			m_trainer_heads[0]->optimizer_step(stream, loss_scale);
			m_trainer_heads[1]->optimizer_step(stream, loss_scale);
			m_trainer_base->optimizer_step(stream, loss_scale);
		}

		// [STEP 4] calcuate the loss if needed
		if (get_loss) {
			float loss1 = m_trainer_heads[0]->loss(stream, *ctx_head1);
			float loss2 = m_trainer_heads[1]->loss(stream, *ctx_head2);

			ret = (loss1 + loss2) / 2.0f;
		}

		return ret;
	}

	void test_inference_time(const uint32_t iters, cudaStream_t stream,
							 const GPUMatrix<float> &input, GPUMatrix<float> &output1,
							 GPUMatrix<float> &output2) {
		cudaStreamSynchronize(stream);
		auto t1 = std::chrono::steady_clock::now();
		for (int i = 0; i < iters; ++i) {
			inference(stream, input, output1, output2);
		}
		cudaStreamSynchronize(stream);
		auto t2			= std::chrono::steady_clock::now();
		auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout << "Inference time: " << iters << " iterations, " << total_time << " ms total, "
				  << (total_time / (float) iters) << " ms/iters" << std::endl;
	}

private:
	CudaGraph m_graph;

	uint32_t m_n_input_dims;
	static constexpr uint32_t M_dims_transfer = 64;
	uint32_t m_dims_output;

	// net_base
	std::shared_ptr<NetworkWithInputEncoding<precision_t>> m_net_base;
	std::shared_ptr<Optimizer<precision_t>> m_optimizer_base;
	std::shared_ptr<Trainer<float, precision_t, precision_t>> m_trainer_base;
	// net1 & net2
	std::shared_ptr<Network<precision_t>> m_net_heads[2];
	std::shared_ptr<Optimizer<precision_t>> m_optimizer_heads[2];
	std::shared_ptr<Trainer<precision_t, precision_t, precision_t>> m_trainer_heads[2];
	std::shared_ptr<Loss<precision_t>> m_loss_heads[2];

	uint32_t m_training_batch_size{0};
	GPUMatrix<precision_t> m_transfer_buffer_y{M_dims_transfer, 1}; // set 1 for resize()
	GPUMatrix<precision_t> m_transfer_buffer_dL_dy{M_dims_transfer, 1};
	GPUMatrix<precision_t> m_transfer_buffer_dL_dy2{M_dims_transfer, 1};

	uint32_t m_inference_batch_size{0};
	GPUMatrix<precision_t> m_transfer_buffer_inference{M_dims_transfer, 1};
};

int main(int argc, char *argv[]) {
	std::cout << "Project Directory: " << PROJECT_DIRECTORY << std::endl
			  << "Output Directory: " << OUTPUT_DIRECTORY << std::endl;

	try {
		uint32_t compute_capability = cuda_compute_capability();
		if (compute_capability < MIN_GPU_ARCH) {
			std::cerr << "Warning: Insufficient compute capability " << compute_capability
					  << " detected. "
					  << "This program was compiled for >=" << MIN_GPU_ARCH
					  << " and may thus behave unexpectedly." << std::endl;
		}

		std::string config_path;
		if (argc != 2) {
			std::cout << "USAGE: " << argv[0] << " "
					  << "path-to-config.json" << std::endl
					  << "Sample config json files are provided in 'data/configs'."
					  << "Fallback to default: data/configs/two_out.json" << std::endl;
			config_path = PROJECT_DIRECTORY "data/configs/two_out.json";
		} else {
			config_path = PROJECT_DIRECTORY + std::string(argv[1]);
		}

		std::cout << "Loading custom json config '" << config_path << "'." << std::endl;
		std::ifstream f{config_path};
		json config = json::parse(f, nullptr, true, /*skip_comments=*/true);

		// [STEP#1] load images to cuda textures
		const uint32_t image_count = 2;

		std::vector<std::string> image_names;
		// should keep gpu data to avoid deallocation
		std::vector<std::shared_ptr<GPUMemory<float>>> images;
		std::vector<uint2> sizes;
		GPUMemory<float> xs_and_ys;

		std::vector<cudaTextureObject_t> textures;
		read_images(config["images"], images, image_names, textures, sizes, xs_and_ys);
		assert(textures.size() == image_count);

		// [STEP#2]: train the model by sampling the above image and optimizing an error metric

		// Various constants for the network and optimization
		const uint32_t batch_size		= 1 << 18;
		const uint32_t n_training_steps = config.value("training_steps", 10000000);
		const uint32_t n_input_dims		= 2; // 2-D image coordinate
		const uint32_t n_output_dims	= 3; // RGB color

		cudaStream_t inference_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
		cudaStream_t training_stream = inference_stream;

		const int sampling_width  = sizes.back().x;
		const int sampling_height = sizes.back().y;
		const uint32_t n_coords	  = sampling_width * sampling_height;

		uint32_t n_coords_padded = next_multiple(n_coords, BATCH_SIZE_GRANULARITY);

		default_rng_t rng{1337};

		// Auxiliary matrices for training
		GPUMatrix<float> training_target1(n_output_dims, batch_size);
		GPUMatrix<float> training_target2(n_output_dims, batch_size);
		GPUMatrix<float> training_batch(n_input_dims, batch_size);

		// Auxiliary matrices for evaluation
		GPUMatrix<float> prediction1(n_output_dims, n_coords_padded);
		GPUMatrix<float> prediction2(n_output_dims, n_coords_padded);
		GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims, n_coords_padded);

		// network related variables
		json nn_config = config.value("network", json::object());
		TwoHeadedNetwork network(nn_config, n_input_dims, n_output_dims);
		network.set_training_batch_size(batch_size);

		network.print_info();

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		float tmp_loss			  = 0;
		uint32_t tmp_loss_counter = 0;

		std::cout << "Beginning optimization with " << n_training_steps << " training steps."
				  << std::endl;

		uint32_t interval = 10;

		GPUMemory<uint8_t> ldr_image{};

		if (config.value("test_inference_time", false)) {
			for (int i = 0; i < 3; ++i) {
				network.test_inference_time(1000, inference_stream, inference_batch, prediction1,
											prediction2);
			}
		}

		for (uint32_t iter_idx = 0; iter_idx < n_training_steps; ++iter_idx) {
			bool print_loss = (iter_idx % interval == 0) || (iter_idx + 1 == n_training_steps);
			bool visualize_learned_func = print_loss;
			{
				generate_random_uniform<float>(training_stream, rng, batch_size * n_input_dims,
											   training_batch.data());
				linear_kernel(eval_image<3>, 0, training_stream, batch_size, textures[0],
							  training_batch.data(), training_target1.data());
				linear_kernel(eval_image<3>, 0, training_stream, batch_size, textures[1],
							  training_batch.data(), training_target2.data());
			}

			// Training step
			{
				bool get_loss			 = (iter_idx % std::min(interval, (uint32_t) 100) == 0);
				float loss_curretnt_iter = network.training_step(
					training_stream, training_batch, training_target1, training_target2, get_loss);

				if (get_loss) {
					tmp_loss += loss_curretnt_iter;
					++tmp_loss_counter;
				}
			}

			// Debug outputs
			{
				if (print_loss) {
					std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
					std::cout << "Step#" << iter_idx << ": "
							  << "loss=" << tmp_loss / (float) tmp_loss_counter << " time="
							  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
									 .count()
							  << "[ms]" << std::endl;

					tmp_loss		 = 0;
					tmp_loss_counter = 0;
				}

				if (visualize_learned_func) {
					network.inference(inference_stream, inference_batch, prediction1, prediction2);
					std::cout << "Writing images at steps: " << iter_idx << "...";
					auto filename = image_names[0] + "-" + std::to_string(iter_idx) + ".png";
					save_image(prediction1.data(), sampling_width, sampling_height, 3,
							   n_output_dims, filename, &ldr_image);
					filename = image_names[1] + "-" + std::to_string(iter_idx) + ".png";
					save_image(prediction2.data(), sampling_width, sampling_height, 3,
							   n_output_dims, filename, &ldr_image);
					std::cout << " Done." << std::endl;
				}

				// Don't count visualizing as part of timing
				// (assumes visualize_learned_pdf is only true when print_loss is true)
				if (print_loss) {
					begin = std::chrono::steady_clock::now();
				}
			}

			if (print_loss && iter_idx > 0 && interval < 1000) {
				interval *= 10;
			}
		}

		free_all_gpu_memory_arenas();

		// If only the memory arenas pertaining to a single stream are to be freed, use
		// free_gpu_memory_arena(stream);
	} catch (const std::exception &e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}
