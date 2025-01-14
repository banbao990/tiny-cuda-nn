// modified from : relative_l2.h

#pragma once
#include "nrrs.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

namespace tcnn {

template <typename T>
__global__ void nrrs_ll2_loss(const uint32_t n_elements, const uint32_t stride,
							  const float loss_scale, const T *__restrict__ predictions,
							  const float *__restrict__ targets, float *__restrict__ values,
							  T *__restrict__ gradients, const float *__restrict__ data_pdf,
							  const bool clampOn, const float clampMax, const bool trainSigma) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t intra_elem_idx = i % stride;
	const uint32_t inter_elem_idx = i / stride;
	const uint32_t training_dims  = 6; // orignial is dims
	const uint32_t calc_dims	  = 3; // x calc data(x) & data(x+1)
	constexpr uint32_t dims		  = 6;

	if (intra_elem_idx >= training_dims) {
		values[i]	 = 0;
		gradients[i] = 0;
		return;
	}

	// the difference is not set 0
	if (intra_elem_idx >= calc_dims) {
		return;
	}

	// TODO: optimize, now only 3 thread is active

	const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;
	const uint32_t n_total	  = n_elements / stride * training_dims;
	const float pdf			  = data_pdf ? data_pdf[target_idx] : 1;

	const float target = targets[target_idx];

	// #####[mean]
	// no activation for mean
	const float mean = (float) predictions[i];

	const float prediction_sq_plus_epsilon = mean * mean + NRRS_EPSILON;
	const float diff					   = mean - target;
	const float diff2					   = diff * diff;

	float loss_mean = diff2 / prediction_sq_plus_epsilon / pdf / n_total;

	float scale_mean = 1.0f;
	if (clampOn) {
		scale_mean = loss_mean > clampMax ? clampMax / loss_mean : 1.0f;
	}

	loss_mean = scale_mean * loss_mean;
	values[i] = loss_mean;

	float grad_mean = scale_mean * 2 * diff / prediction_sq_plus_epsilon;
	grad_mean		= grad_mean / pdf / n_total;
	gradients[i]	= (T) (loss_scale * grad_mean);

	if (trainSigma) {
		// #####[sigma]
		const float sigma_raw = (float) predictions[i + BB_SIGMA_OFFSET];
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

		values[i + BB_L2_OFFSET] = scale_sigma * loss_sigma;

		// d(loss)/d(sigma)
		float grad_sigma =
			scale_sigma * (1 / (sigma + NRRS_EPSILON) - diff2 / (sigma2 * sigma + NRRS_EPSILON));

		grad_sigma					   = grad_sigma * dsigma_dsigma_raw / pdf / n_total;
		gradients[i + BB_SIGMA_OFFSET] = (T) (loss_scale * grad_sigma);
	} else {
		// // #####[X2]
		// float scale_x2 = 0.01f;
		// const float x2 = (float) predictions[i + BB_L2_OFFSET];
		// // const float target_x2 = (float) targets[target_idx + BB_L2_OFFSET];
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
		// values[i + BB_L2_OFFSET]	= loss_x2_is_nan ? 0 : loss_x2;
		// float gradient_x2			= 2 * diff_x2 / x2_sq_plus_epsilon / pdf;
		// gradient_x2					= scale_x2 * loss_scale * gradient_x2 / n_total;
		// gradients[i + BB_L2_OFFSET] = (T) (loss_x2_is_nan ? 0 : gradient_x2);

		// ##### Variance = E[(x - E[x])^2]
		// loss = || pre - diff2 ||^2 / pre^2
		const float prediction_x2	 = (float) predictions[i + BB_L2_OFFSET];
		const float diff_x2			 = prediction_x2 - diff2;
		const float diff_x2_2		 = diff_x2 * diff_x2;
		const float prediction_x2_sq = prediction_x2 * prediction_x2 + NRRS_EPSILON;

		float loss_x2  = diff_x2_2 / prediction_x2_sq / pdf / n_total;
		float scale_x2 = 1.0f;
		if (clampOn) {
			scale_x2 = loss_x2 > clampMax ? clampMax / loss_x2 : 1.0f;
		}

		values[i + BB_L2_OFFSET]	= scale_x2 * loss_x2;
		float grad_x2				= scale_x2 * 2 * diff_x2 / prediction_x2_sq;
		grad_x2						= grad_x2 / pdf / n_total;
		gradients[i + BB_L2_OFFSET] = (T) (loss_scale * grad_x2);

		// check nan
		// if (isnan(loss_x2) || isinf(loss_x2) || isnan(gradient_x2) || isinf(gradient_x2)) {
		// 	printf("L2 [%d]: loss_x2 = %g, gradient_x2 = %g, prediction = %g, L2 = %g\n", i,
		// 		   loss_x2, gradient_x2, (float) predictions[i + 1], (float) predictions[i]);
		// }
	}

	// values[i + BB_L2_OFFSET]	= 0;
	// gradients[i + BB_L2_OFFSET] = 0;
}

template <typename T> class NRRSLL2Loss : public Loss<T> {
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

		CHECK_THROW(dims == 6);

		linear_kernel(nrrs_ll2_loss<T>, 0, stream, prediction.n_elements(), stride, loss_scale,
					  prediction.data(), target.data(), values.data(), gradients.data(),
					  data_pdf ? data_pdf->data() : nullptr, mClampOn, mClampMax, mTrainSigma);
	}

	void update_hyperparams(const json &params) override {
		mClampMax	= params.value("clamp_max", mClampMax);
		mClampOn	= params.value("clamp_on", mClampOn);
		mTrainSigma = params.value("train_sigma", mTrainSigma);

		if (!(params.size() == 1 && params.contains("offset"))) {
			printf("[NRRS_LL2 Loss] update hyperparams: %s\n", params.dump().c_str());
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "NRRS_LL2"},
			{"clamp_max", mClampMax},
			{"clamp_on", mClampOn},
			{"train_sigma", mTrainSigma},
		};
	}

private:
	bool mClampOn{false};
	float mClampMax{500.0f};
	bool mTrainSigma{true}; // train sigma or X2
};

} // namespace tcnn
