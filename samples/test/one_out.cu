#include "utils.h"

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
					  << "Fallback to default: data/configs/one_out.json" << std::endl;
			config_path = PROJECT_DIRECTORY "data/configs/one_out.json";
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

		GPUMemory<cudaTextureObject_t> textures_gpu(image_count);
		{
			std::vector<cudaTextureObject_t> textures;
			read_images(config["images"], images, image_names, textures, sizes, xs_and_ys);
			assert(textures.size() == image_count);
			textures_gpu.copy_from_host(textures.data());
		}

		// [STEP#2]: train the model by sampling the above image and optimizing an error metric

		// Various constants for the network and optimization
		const uint32_t batch_size		= 1 << 18;
		const uint32_t n_training_steps = config.value("training_steps", 10000000);
		const uint32_t n_input_dims		= 2; // 2-D image coordinate
		const uint32_t n_channels		= 3;
		const uint32_t n_output_dims	= n_channels * image_count; // RGB color

		cudaStream_t inference_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
		cudaStream_t training_stream = inference_stream;

		const int sampling_width  = sizes.back().x;
		const int sampling_height = sizes.back().y;
		const uint32_t n_coords	  = sampling_width * sampling_height;

		uint32_t n_coords_padded = next_multiple(n_coords, BATCH_SIZE_GRANULARITY);

		default_rng_t rng{1337};

		// Auxiliary matrices for training
		GPUMatrix<float> training_target(n_output_dims, batch_size);
		GPUMatrix<float> training_batch(n_input_dims, batch_size);

		// Auxiliary matrices for evaluation
		GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
		GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims, n_coords_padded);

		json nn_config		= config.value("network", json::object());
		json encoding_opts	= nn_config.value("encoding", json::object());
		json loss_opts		= nn_config.value("loss", json::object());
		json optimizer_opts = nn_config.value("optimizer", json::object());
		json network_opts	= nn_config.value("network", json::object());

		std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
		std::shared_ptr<Optimizer<precision_t>> optimizer{
			create_optimizer<precision_t>(optimizer_opts)};
		std::shared_ptr<NetworkWithInputEncoding<precision_t>> network =
			std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims,
																	encoding_opts, network_opts);

		auto trainer =
			std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

		std::cout << "Network has " << network->n_params() << " parameters." << std::endl;
		std::cout << "Trainer has " << trainer->n_params() << " parameters." << std::endl;
		std::cout << "Optimizer has " << optimizer->n_weights() << " parameters." << std::endl;

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		float tmp_loss			  = 0;
		uint32_t tmp_loss_counter = 0;

		std::cout << "Beginning optimization with " << n_training_steps << " training steps."
				  << std::endl;

		uint32_t interval = 10;

		GPUMemory<uint8_t> ldr_image{};

		// test inference time
		if (config.value("test_inference_time", false)) {
			auto measure_inference_time = [&](uint32_t iters) {
				cudaStreamSynchronize(inference_stream);
				auto t1 = std::chrono::steady_clock::now();
				for (int i = 0; i < iters; ++i) {
					network->inference(inference_stream, inference_batch, prediction);
				}
				cudaStreamSynchronize(inference_stream);
				auto t2 = std::chrono::steady_clock::now();
				auto total_time =
					std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
				std::cout << "Inference time: " << iters << " iterations, " << total_time
						  << " ms total, " << (total_time / (float) iters) << " ms/iters"
						  << std::endl;
			};
			for (int i = 0; i < 3; ++i) {
				measure_inference_time(1000);
			}
		}

		for (uint32_t iter_idx = 0; iter_idx < n_training_steps; ++iter_idx) {
			bool print_loss = (iter_idx % interval == 0) || (iter_idx + 1 == n_training_steps);
			bool visualize_learned_func = print_loss;
			{
				generate_random_uniform<float>(training_stream, rng, batch_size * n_input_dims,
											   training_batch.data());
				linear_kernel(eval_images<image_count>, 0, training_stream, batch_size,
							  textures_gpu.data(), training_batch.data(), training_target.data());
			}

			// Training step
			{
				auto ctx = trainer->training_step(training_stream, training_batch, training_target);

				if (iter_idx % std::min(interval, (uint32_t) 100) == 0) {
					tmp_loss += trainer->loss(training_stream, *ctx);
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
					network->inference(inference_stream, inference_batch, prediction);
					for (int i = 0; i < image_count; ++i) {
						auto filename = image_names[i] + "-" + std::to_string(iter_idx) + ".png";
						std::cout << "Writing '" << filename << "'... ";

						save_image(prediction.data() + (i * 3), sampling_width, sampling_height, 3,
								   n_output_dims, filename, &ldr_image);
						std::cout << "done." << std::endl;
					}
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
