#pragma once

#include "common.h"

std::shared_ptr<GPUMemory<float>> load_image(const std::string &filename, int &width, int &height);

template <typename T>
void save_image(const T *image, int width, int height, int n_channels, int channel_stride,
				const std::string &filename, GPUMemory<uint8_t> *image_ldr = nullptr);

// device memory for x,y coordinates
GPUMemory<float> generate_xys(int width, int height);

void read_images(json &images, std::vector<std::shared_ptr<GPUMemory<float>>> &images_data,
				 std::vector<std::string> &image_names, std::vector<cudaTextureObject_t> &textures,
				 std::vector<uint2> &sizes, GPUMemory<float> &xys);

//////////////////////////////////////
//////////// CUDA kernels ////////////
//////////////////////////////////////

// gradient_1 += gradient_2
__global__ void calculate_gradient_sum(const uint32_t n_elements, __half *__restrict__ gradient_1,
									   const __half *__restrict__ gradient_2);

template <typename T>
__global__ void to_ldr(const uint32_t num_elements, const uint32_t n_channels,
					   const uint32_t stride, const T *__restrict__ in, uint8_t *__restrict__ out);

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture,
						   float *__restrict__ xs_and_ys, float *__restrict__ result);

// only RGB, stride = 3
template <uint32_t image_count>
__global__ void eval_images(uint32_t n_elements, cudaTextureObject_t *texture,
							float *__restrict__ xs_and_ys, float *__restrict__ result);
