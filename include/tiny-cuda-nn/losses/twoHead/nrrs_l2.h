// modified from : relative_l2.h

#pragma once
#include "../nrrs.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

namespace tcnn {

template <typename T>
__global__ void nrrs_twohead_l2_loss(const uint32_t n_elements, const uint32_t stride,
									 const float loss_scale, const T *__restrict__ predictions,
									 const float *__restrict__ targets, float *__restrict__ values,
									 T *__restrict__ gradients, const float *__restrict__ data_pdf,
									 const bool clampOn, const float clampMax,
									 const bool trainSigma, const bool onlyTrainL) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n_elements) return;

	constexpr uint32_t TARGET_DIMS		= 6;
	constexpr uint32_t TARGET_L2_OFFSET = 3;
	constexpr uint32_t DIMS				= 3;

	const uint32_t intra_elem_idx = tid % stride;
	const uint32_t inter_elem_idx = tid / stride;

	// TODO: optimize, now only 3 thread is active
	if (intra_elem_idx >= DIMS) {
		values[tid]	   = 0;
		gradients[tid] = 0;
		return;
	}

	const uint32_t target_idx = inter_elem_idx * TARGET_DIMS + intra_elem_idx + TARGET_L2_OFFSET;
	const uint32_t n_total	  = n_elements / stride * DIMS;
	const float pdf			  = data_pdf ? data_pdf[target_idx] : 1;

	// we set this in the loss nrrs_l
	const float diff2 = targets[target_idx];

	if (trainSigma) {
		// #####[sigma]
		const float sigma_raw = (float) predictions[tid];
		// NLL
		// loss = (mean - x)*(mean - x) / (2 * sigma*sigma) + log(sigma)

		// activation: softplus & y = 0.5x + ln2
		float sigma = sigma_raw;
		float dsigma_dsigma_raw;
		bb_activation_and_gradient(sigma, dsigma_dsigma_raw);

		const float sigma2 = sigma * sigma;

		// [TODO] infact, during training, we just need gradient
		float loss_sigma =
			(diff2 / (2 * sigma2 + NRRS_EPSILON) + logf(sigma + NRRS_EPSILON)) / pdf / n_total;

		float scale_sigma = 1.0f;
		if (clampOn) {
			scale_sigma = loss_sigma > clampMax ? clampMax / scale_sigma : 1.0f;
		}

		values[tid] = scale_sigma * loss_sigma;

		// d(loss)/d(sigma)
		float grad_sigma =
			scale_sigma * (1 / (sigma + NRRS_EPSILON) - diff2 / (sigma2 * sigma + NRRS_EPSILON));

		grad_sigma	   = grad_sigma * dsigma_dsigma_raw / pdf / n_total;
		gradients[tid] = (T) (loss_scale * grad_sigma);
	} else {
		// // #####[X2]
		// float scale_x2 = 0.01f;
		// const float x2 = (float) predictions[tid];
		// // const float target_x2 = (float) targets[target_idx];
		// const float target_x2		   = target * target;
		// const float x2_sq_plus_epsilon = x2 * x2 + NRRS_EPSILON;
		// const float diff_x2			   = x2 - target_x2;

		// float loss_x2 = scale_x2 * diff_x2 * diff_x2 / x2_sq_plus_epsilon / pdf / n_total;
		// if (clampOn) {
		// 	const float scale1 = loss_x2 > clampMax ? clampMax / loss_x2 : 1.0f;
		// 	loss_x2 *= scale1;
		// 	scale_x2 *= scale1;
		// }

		// // if nan, set to 0
		// bool loss_x2_is_nan			= isnan(loss_x2);
		// values[tid]	= loss_x2_is_nan ? 0 : loss_x2;
		// float gradient_x2			= 2 * diff_x2 / x2_sq_plus_epsilon / pdf;
		// gradient_x2					= scale_x2 * loss_scale * gradient_x2 / n_total;
		// gradients[tid] = (T) (loss_x2_is_nan ? 0 : gradient_x2);

		// ##### Variance = E[(x - E[x])^2]
		// loss = || pre - diff2 ||^2 / pre^2
		const float target_desired = onlyTrainL ? 0 : diff2;

		const float prediction_x2	 = (float) predictions[tid];
		const float diff_x2			 = prediction_x2 - target_desired;
		const float diff_x2_2		 = diff_x2 * diff_x2;
		const float prediction_x2_sq = prediction_x2 * prediction_x2 + NRRS_EPSILON;

#ifdef BB_TCNN_DEBUG_MODE
		if ((isinf(target_desired) || isnan(target_desired) || isnan(prediction_sq_plus_epsilon)) ||
			(isinf(diff_x2_2) || isnan(diff_x2_2) || isnan(prediction_x2_sq))) {
			printf("[%d]: [L] prediction = %g, target = %g, diff = %g, "
				   "[Var] prediction_x2 = %g, target = %g, diff_x2 = %g\n",

				   tid, mean, target, diff, prediction_x2, target_desired, diff_x2);
		}
#endif

		float loss_x2  = diff_x2_2 / prediction_x2_sq / pdf / n_total;
		float scale_x2 = 1.0f;
		if (clampOn) {
			scale_x2 = loss_x2 > clampMax ? clampMax / loss_x2 : 1.0f;
		}

		values[tid]	   = scale_x2 * loss_x2;
		float grad_x2  = scale_x2 * 2 * diff_x2 / prediction_x2_sq;
		grad_x2		   = grad_x2 / pdf / n_total;
		gradients[tid] = (T) (loss_scale * grad_x2);

#ifdef BB_TCNN_DEBUG_MODE
		if (isnan(loss_x2) || isinf(loss_x2) || isnan(gradient_x2) || isinf(gradient_x2)) {
			printf("L2 [%d]: loss_x2 = %g, gradient_x2 = %g, prediction = %g, L2 = %g\n", tid,
				   loss_x2, gradient_x2, (float) predictions[tid + 1], (float) predictions[tid]);
		}
#endif
	}

	// values[tid]	= 0;
	// gradients[tid] = 0;
}

template <typename T> class NRRSTwoHeadL2Loss : public Loss<T> {
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

		linear_kernel(nrrs_twohead_l2_loss<T>, 0, stream, prediction.n_elements(), stride,
					  loss_scale, prediction.data(), target.data(), values.data(), gradients.data(),
					  data_pdf ? data_pdf->data() : nullptr, mClampOn, mClampMax, mTrainSigma,
					  mOnlyTrainL);
	}

	void update_hyperparams(const json &params) override {
		mClampMax	= params.value("clamp_max", mClampMax);
		mClampOn	= params.value("clamp_on", mClampOn);
		mTrainSigma = params.value("train_sigma", mTrainSigma);
		mStep		= params.value("step", mStep);

		mOnlyTrainL = mStep == 0;

		if (!(params.size() == 1 && params.contains("offset"))) {
			printf("[NRRS_TwoHead L2 Loss] update hyperparams: %s,{\"only_train_L\":%d}\n",
				   params.dump().c_str(), mOnlyTrainL);
		}
	}

	json hyperparams() const override {
		return {{"otype", "NRRS_TwoHead_L2"},
				{"clamp_max", mClampMax},
				{"clamp_on", mClampOn},
				{"train_sigma", mTrainSigma},
				{"only_train_L", mOnlyTrainL}};
	}

private:
	int mStep{0};
	bool mClampOn{false};
	float mClampMax{500.0f};
	bool mOnlyTrainL{false}; // false: train both L and L2; true: only train L, controlled by step
	bool mTrainSigma{true};	 // train sigma or X2
};

} // namespace tcnn
