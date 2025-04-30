// modified from : relative_l2.h

#pragma once
#include "../nrrs.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

namespace tcnn {

template <typename T>
__global__ void nrrs_twohead_l_loss(const uint32_t n_elements, const uint32_t stride,
									const float loss_scale, const T *__restrict__ predictions,
									float *__restrict__ targets, float *__restrict__ values,
									T *__restrict__ gradients, const float *__restrict__ data_pdf,
									const bool clampOn, const float clampMax) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n_elements) return;

	constexpr uint32_t TARGET_DIMS		= 6;
	constexpr uint32_t TARGET_L_OFFSET	= 0;
	constexpr uint32_t TARGET_L2_OFFSET = 0;
	constexpr uint32_t DIMS				= 3;

	const uint32_t intra_elem_idx = tid % stride;
	const uint32_t inter_elem_idx = tid / stride;

	// TODO: optimize, now only 3 thread is active
	if (intra_elem_idx >= DIMS) {
		values[tid]	   = 0;
		gradients[tid] = 0;
		return;
	}

	const uint32_t target_idx = inter_elem_idx * TARGET_DIMS + intra_elem_idx + TARGET_L_OFFSET;
	const uint32_t n_total	  = n_elements / stride * DIMS;
	const float pdf			  = data_pdf ? data_pdf[target_idx] : 1;

	const float target = targets[target_idx];

	// #####[mean]
	// no activation for mean
	const float mean = (float) predictions[tid];

	const float prediction_sq_plus_epsilon = mean * mean + NRRS_EPSILON;
	const float diff					   = mean - target;
	const float diff2					   = diff * diff;

	// [IM] write this for loss_l2
	targets[target_idx + TARGET_L2_OFFSET] = target;

	float loss_mean = diff2 / prediction_sq_plus_epsilon / pdf / n_total;

	float scale_mean = 1.0f;
	if (clampOn) {
		scale_mean = loss_mean > clampMax ? clampMax / loss_mean : 1.0f;
	}

	loss_mean	= scale_mean * loss_mean;
	values[tid] = loss_mean;

	float grad_mean = scale_mean * 2 * diff / prediction_sq_plus_epsilon;
	grad_mean		= grad_mean / pdf / n_total;
	gradients[tid]	= (T) (loss_scale * grad_mean);
}

template <typename T> class NRRSTwoHeadLLoss : public Loss<T> {
public:
	void evaluate(cudaStream_t stream, const float loss_scale, const GPUMatrix<T> &prediction,
				  const GPUMatrix<float> &target, GPUMatrix<float> &values, GPUMatrix<T> &gradients,
				  const GPUMatrix<float> *data_pdf = nullptr) const override {
		const uint32_t dims	  = target.m();
		const uint32_t stride = prediction.m();

		CHECK_THROW(prediction.n() == target.n());
		CHECK_THROW(values.m() == stride);
		CHECK_THROW(gradients.m() == stride);
		CHECK_THROW(!data_pdf || data_pdf->m() == dims);

		CHECK_THROW(dims == 3);

		// target.data() is elements x 6
		// decode to 6

		linear_kernel(nrrs_twohead_l_loss<T>, 0, stream, prediction.n_elements(), stride,
					  loss_scale, prediction.data(), target.data(), values.data(), gradients.data(),
					  data_pdf ? data_pdf->data() : nullptr, mClampOn, mClampMax);
	}

	void update_hyperparams(const json &params) override {
		mClampMax = params.value("clamp_max", mClampMax);
		mClampOn  = params.value("clamp_on", mClampOn);

		if (!(params.size() == 1 && params.contains("offset"))) {
			printf("[NRRS_TwoHead L Loss] update hyperparams: %s\n", params.dump().c_str());
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "NRRS_TwoHead_L"},
			{"clamp_max", mClampMax},
			{"clamp_on", mClampOn},
		};
	}

private:
	bool mClampOn{false};
	float mClampMax{500.0f};
};

} // namespace tcnn
