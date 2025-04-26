#include "utils.h"

std::shared_ptr<GPUMemory<float>> load_image(const std::string &filename, int &width, int &height) {
	std::string path = PROJECT_DIRECTORY + filename;
	std::cout << "Loading image '" << path << std::endl;
	// width * height * RGBA
	float *out = load_stbi(&width, &height, path.c_str());

	auto result = std::make_shared<GPUMemory<float>>(width * height * 4);
	result->copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

template <typename T>
void save_image(const T *image, int width, int height, int n_channels, int channel_stride,
				const std::string &filename, GPUMemory<uint8_t> *image_ldr) {

	const bool generate_ldr = image_ldr == nullptr;
	if (generate_ldr) {
		image_ldr = new GPUMemory<uint8_t>(width * height * n_channels);
	} else {
		const int image_size = width * height * n_channels;
		if (image_ldr->size() != image_size) {
			image_ldr->resize(image_size);
		}
	}

	linear_kernel(to_ldr<T>, 0, nullptr, width * height * n_channels, n_channels, channel_stride,
				  image, image_ldr->data());

	std::vector<uint8_t> image_ldr_host(width * height * n_channels);
	CUDA_CHECK_THROW(cudaMemcpy(image_ldr_host.data(), image_ldr->data(), image_ldr->size(),
								cudaMemcpyDeviceToHost));

	save_stbi(image_ldr_host.data(), width, height, n_channels,
			  (OUTPUT_DIRECTORY + filename).c_str());

	if (generate_ldr) {
		delete image_ldr;
	}
}

GPUMemory<float> generate_xys(int width, int height) {
	int sampling_width	= width;
	int sampling_height = height;

	// Uncomment to fix the resolution of the training task independent of input image
	// int sampling_width	= 1024;
	// int sampling_height = 1024;

	uint32_t n_coords		 = sampling_width * sampling_height;
	uint32_t n_coords_padded = next_multiple(n_coords, BATCH_SIZE_GRANULARITY);

	GPUMemory<float> xs_and_ys(n_coords_padded * 2);

	std::vector<float> host_xs_and_ys(n_coords_padded * 2);
	for (int y = 0; y < sampling_height; ++y) {
		for (int x = 0; x < sampling_width; ++x) {
			int idx					= (y * sampling_width + x) * 2;
			host_xs_and_ys[idx + 0] = (float) (x + 0.5) / (float) sampling_width;
			host_xs_and_ys[idx + 1] = (float) (y + 0.5) / (float) sampling_height;
		}
	}

	xs_and_ys.copy_from_host(host_xs_and_ys.data());

	return xs_and_ys;
}

void read_images(json &images, std::vector<std::shared_ptr<GPUMemory<float>>> &images_data,
				 std::vector<std::string> &image_names, std::vector<cudaTextureObject_t> &textures,
				 std::vector<uint2> &sizes, GPUMemory<float> &xys) {
	textures.clear();
	sizes.clear();

	int sampling_width	= 1024;
	int sampling_height = 1024;
	uint32_t n_coords	= sampling_width * sampling_height;
	GPUMemory<float> sampled_image(n_coords * 3);

	xys = generate_xys(sampling_width, sampling_height);

	int nums = images.size();
	std::cout << "there is total " << nums << " images" << std::endl
			  << "We only use first 2 images for training" << std::endl;
	nums = std::min(nums, 2);
	for (int i = 0; i < nums; ++i) {
		std::string image_path = images[i].get<std::string>();

		// First step: load images that we'd like to learn
		int width, height;

		auto image = load_image(image_path, width, height);
		// std::cout << "ptr: " << image->data() << std::endl;

		// Second step: create a cuda texture out of this image. It'll be used to generate
		// training data efficiently on the fly
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType			   = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = image->data();
		resDesc.res.pitch2D.desc =
			cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		resDesc.res.pitch2D.width		 = width;
		resDesc.res.pitch2D.height		 = height;
		resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode		 = cudaFilterModeLinear;
		texDesc.normalizedCoords = true;
		texDesc.addressMode[0]	 = cudaAddressModeClamp;
		texDesc.addressMode[1]	 = cudaAddressModeClamp;
		texDesc.addressMode[2]	 = cudaAddressModeClamp;

		cudaTextureObject_t texture;
		CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));

		// Third step: sample a reference image to dump to disk. Visual comparison of this
		// reference image and the learned function will be eventually possible.

		linear_kernel(eval_image<3>, 0, nullptr, n_coords, texture, xys.data(),
					  sampled_image.data());
		// std::cout << "texture: " << texture << std::endl;

		// construct return values

		// image name: aa/cc/bb.jpg => bb [no suffix]
		std::string::size_type pos = image_path.find_last_of("/");
		if (pos != std::string::npos) {
			image_path = image_path.substr(pos + 1);
		}
		save_image(sampled_image.data(), sampling_width, sampling_height, 3, 3, image_path);

		pos = image_path.find_last_of(".");
		if (pos != std::string::npos) {
			image_path = image_path.substr(0, pos);
		}

		// should keep this to avoid deallocation
		images_data.push_back(image);
		image_names.push_back(image_path);
		textures.push_back(texture);
		sizes.push_back(make_uint2(width, height));
	}

	// fix training size
	sizes.push_back(make_uint2(sampling_width, sampling_height));
}

//////////////////////////////////////
//////////// CUDA kernels ////////////
//////////////////////////////////////

__global__ void calculate_gradient_sum(const uint32_t n_elements, __half *__restrict__ gradient_1,
									   const __half *__restrict__ gradient_2) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_elements) return;
	gradient_1[tid] += gradient_2[tid];
}

template <typename T>
__global__ void to_ldr(const uint32_t num_elements, const uint32_t n_channels,
					   const uint32_t stride, const T *__restrict__ in, uint8_t *__restrict__ out) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_elements) return;

	const uint32_t pixel   = tid / n_channels;
	const uint32_t channel = tid - pixel * n_channels;

	out[tid] =
		(uint8_t) (powf(fmaxf(fminf(in[pixel * stride + channel], 1.0f), 0.0f), 1.0f / 2.2f) *
					   255.0f +
				   0.5f);
}

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture,
						   float *__restrict__ xs_and_ys, float *__restrict__ result) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_elements) return;

	uint32_t output_idx = tid * stride;
	uint32_t input_idx	= tid * 2;

	float4 val			   = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);
	result[output_idx + 0] = val.x;
	result[output_idx + 1] = val.y;
	result[output_idx + 2] = val.z;

	for (uint32_t i = 3; i < stride; ++i) {
		result[output_idx + i] = 1;
	}
}

template <uint32_t image_count>
__global__ void eval_images(uint32_t n_elements, cudaTextureObject_t *texture,
							float *__restrict__ xs_and_ys, float *__restrict__ result) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_elements) return;

	uint32_t output_idx = tid * 3 * image_count;
	uint32_t input_idx	= tid * 2;

	for (int i = 0; i < image_count; ++i) {
		float4 val = tex2D<float4>(texture[i], xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);

		uint32_t offset = i * 3;

		result[output_idx + offset + 0] = val.x;
		result[output_idx + offset + 1] = val.y;
		result[output_idx + offset + 2] = val.z;
	}
}

// template instance
template void save_image<float>(const float *image, int width, int height, int n_channels,
								int channel_stride, const std::string &filename,
								GPUMemory<uint8_t> *image_ldr);
template __global__ void to_ldr<float>(const uint32_t num_elements, const uint32_t n_channels,
									   const uint32_t stride, const float *__restrict__ in,
									   uint8_t *__restrict__ out);
template __global__ void eval_images<2>(uint32_t n_elements, cudaTextureObject_t *texture,
										float *__restrict__ xs_and_ys, float *__restrict__ result);
template __global__ void eval_image<3>(uint32_t n_elements, cudaTextureObject_t texture,
									   float *__restrict__ xs_and_ys, float *__restrict__ result);
