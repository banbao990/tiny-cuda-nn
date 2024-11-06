/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   rrs_loss.h
 *  @author banbao990, PKU
 *  @brief Neural-RRS Loss
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

#define BB_TCNN_DEBUG_MODE
// #undef BB_TCNN_DEBUG_MODE

namespace tcnn {
#define NRRS_EPSILON 1e-2f

template <typename T>
__global__ void
neural_rrs_loss_L_L2_v2(const uint32_t n_elements, const uint32_t stride, const uint32_t dims,
						const float loss_scale, const T *__restrict__ predictions,
						const float *__restrict__ targets, float *__restrict__ values,
						T *__restrict__ gradients, const float *__restrict__ data_pdf = nullptr,
						const bool clampOn = false, const float clampMax = 10.0f
#ifdef BB_TCNN_DEBUG_MODE
						,
						const uint32_t showLossIndex = 0
#endif
);

template <typename T>
__global__ void
neural_rrs_loss_L_L2(const uint32_t n_elements, const uint32_t stride, const uint32_t dims,
					 const float loss_scale, const T *__restrict__ predictions,
					 const float *__restrict__ targets, float *__restrict__ values,
					 T *__restrict__ gradients, const float *__restrict__ data_pdf = nullptr,
					 const bool clampOn = false, const float clampMax = 10.0f,
					 const uint32_t step = 1
#ifdef BB_TCNN_DEBUG_MODE
					 ,
					 const uint32_t showLossIndex = 0
#endif
);

template <typename T>
__global__ void
neural_rrs_loss_rrs(const uint32_t n_elements, const uint32_t stride, const uint32_t dims,
					const float loss_scale, const uint32_t step, const float *__restrict__ thp,
					const float *__restrict__ pdf, const float *__restrict__ error,
					const float *__restrict__ ref_mean, const float *__restrict__ error_sum,
					const float *__restrict__ sample_weight, const uint32_t pixels_num,
					const T *__restrict__ predictions, const float *__restrict__ targets,
					float *__restrict__ values, T *__restrict__ gradients,
					const float *__restrict__ data_pdf = nullptr, const bool clampOn = false,
					const float clampMax = 10.0f
#ifdef BB_TCNN_DEBUG_MODE
					,
					const uint32_t showLossIndex = 0
#endif
) {

	const uint32_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx >= n_elements) return;

	const uint32_t prediction_idx = thread_idx * stride;

	if (step == 3) {
#define BB_GAMMA1 1.0f
#define BB_GAMMA2 1.0f
#define BB_GAMMA3 0 // 1e-4f
		// now the output is rrs

		const float var_sum		   = *error_sum;
		const float rrs_loss_scale = 1e-3f;

		float rrs					 = (float) predictions[prediction_idx + 6];
		float var					 = error[thread_idx];
		float path_pdf				 = pdf[thread_idx];
		const float pixel_err_weight = sample_weight[thread_idx];

		// const float sigmoid_scale = 20.0f;
		// sigmoid, s(x) = 1 / (1 + exp(-x))
		// rrs = 1.0f / (1.0f + expf(-rrs));

		float dactivate_drrs = 1.0f;
		// s'(x) = s(x) * (1 - s(x))
		// dactivate_drrs = rrs * (1.0f - rrs) * sigmoid_scale;
		// rrs *= sigmoid_scale;

		{ // k = 1
			// loss
			const uint32_t prediction_rrs_idx = prediction_idx + 6;
			float e1						  = var - var_sum / pixels_num;
			values[prediction_rrs_idx] =
				rrs_loss_scale *
				(pixel_err_weight * (BB_GAMMA1 * abs(e1) + BB_GAMMA2 * var) + BB_GAMMA3 * rrs);

			// gradient
			float dE_dvar =
				BB_GAMMA1 * ((e1 > 0 ? 1 : -1) * (float(pixels_num - 1) / float(pixels_num))) +
				BB_GAMMA2;

			// dE_dvar *= expf(var) - 1.0f; // log scale

			// rel var
			// float rel_inv = ref_mean[thread_idx];
			// rel_inv		  = max(rel_inv * rel_inv, 1e-4f);
			// rel_inv		  = rel_inv > 0 ? 1.0f / rel_inv : 0;
			// TODO: this may not needed as we divided by it when use it
			constexpr float rel_inv = 1.0f;

			float path_var = 0.0f;
			{
				float r	  = (float) predictions[prediction_idx + 0];
				float g	  = (float) predictions[prediction_idx + 1];
				float b	  = (float) predictions[prediction_idx + 2];
				float r2  = (float) predictions[prediction_idx + 3];
				float g2  = (float) predictions[prediction_idx + 4];
				float b2  = (float) predictions[prediction_idx + 5];
				float ex  = (r + g + b) / 3.0f;
				float ex2 = (r2 + g2 + b2) / 3.0f;
				path_var  = max(ex2 - ex * ex, 0.0f);
			}
			float dvar_drrs = -path_pdf * path_var / max(rrs * rrs, 1e-4f);

			gradients[prediction_rrs_idx] =
				(T) (loss_scale * rrs_loss_scale *
					 (pixel_err_weight * (dE_dvar * rel_inv * dvar_drrs * 100.0f) + BB_GAMMA3) *
					 dactivate_drrs);
		}

#undef BB_GAMMA1
#undef BB_GAMMA2
#undef BB_GAMMA3

	} else {
		const uint32_t target_idx = thread_idx * 3; // thp: dim = 3

		float rrs_gt			   = 1.0f;
		const float rrs_loss_scale = 1.0f;
		// float luminance = 1.0f;
		if (step == 1) {
			// set all rrs = 1
		} else if (step == 2) {
			// step 2: ADRRS
			const float r  = (float) predictions[prediction_idx + 0];
			const float g  = (float) predictions[prediction_idx + 1];
			const float b  = (float) predictions[prediction_idx + 2];
			const float rp = (float) thp[target_idx + 0];
			const float gp = (float) thp[target_idx + 1];
			const float bp = (float) thp[target_idx + 2];

			rrs_gt = (r * rp + g * gp + b * bp) / 3.0f;
			// clamp
			// rrs_gt = fmaxf(0.5f, fminf(20.0f, rrs_gt));
			// luminance = (0.299f * r + 0.587f * g + 0.114f * b);
		}

		const uint32_t prediction_rrs_idx = prediction_idx + 6;

		const float prediction_ori = (float) predictions[prediction_rrs_idx];
		// activation: sigmoid
		// const float prediction = 1.0f / (1.0f + expf(-prediction_ori));
		// activation: softplus
		// const float prediction_ori_exp = expf(prediction_ori);
		// const float prediction = log1pf(prediction_ori_exp);
		float prediction	 = prediction_ori;
		float dactivate_drrs = 1.0f;

		// const float sigmoid_scale = 20.f;
		// prediction				  = 1.0f / (1.0f + expf(-prediction_ori));
		// dactivate_drrs			  = prediction * (1.0f - prediction) * sigmoid_scale;
		// prediction *= sigmoid_scale;

		const float difference = rrs_loss_scale * dactivate_drrs * (prediction - rrs_gt);
		const uint32_t n_total = n_elements;
		const float pdf		   = data_pdf ? data_pdf[target_idx] : 1;

		// const float prediction_sq_plus_epsilon = luminance * luminance + NRRS_EPSILON;
		const float prediction_sq_plus_epsilon = prediction * prediction + NRRS_EPSILON;

		float v = difference * difference / prediction_sq_plus_epsilon / pdf / n_total;

		float scale = 1.0f;
		if (clampOn) {
			scale = v > clampMax ? clampMax / v : 1.0f;
		}
		values[prediction_rrs_idx] = scale * v;

		float gradient = 2 * difference / prediction_sq_plus_epsilon / pdf;
		// sigmoid
		// gradient *= prediction * (1.0f - prediction);
		// softplus
		// gradient *= prediction_ori_exp / (1.0f + prediction_ori_exp);
		gradients[prediction_rrs_idx] = (T) (scale * loss_scale * gradient / n_total);

		// if (step == 2) {
		// 	// d(output)/d(L)
		// 	for (int i = 0; i < 3; ++i) {
		// 		float r_thp		 = (float) thp[target_idx + i];
		// 		float r_gradiant = -2.0f / 3 * r_thp * difference / prediction_sq_plus_epsilon /
		// pdf; 		gradients[prediction_idx + i] += (T) (scale * loss_scale * r_gradiant /
		// n_total);
		// 	}
		// }
	}

#ifdef BB_TCNN_DEBUG_MODE
#define IDX(x) (prediction_idx + x)
	// TODO: for visualization
	// if (step == 3) {
	// 	float sum = values[IDX(6)] / 7;
	// 	for (int i = 0; i < 7; ++i) {
	// 		values[IDX(i)] = sum;
	// 	}
	// }
	if (showLossIndex != 0) {
		float sum = 0;
		if (showLossIndex == 1) {
			for (int i = 0; i < 3; ++i) {
				sum += values[IDX(i)];
			}
		} else if (showLossIndex == 2) {
			for (int i = 3; i < 6; ++i) {
				sum += values[IDX(i)];
			}
		} else if (showLossIndex == 3) {
			sum = values[IDX(6)];
		}
		sum /= 7;
		for (int i = 0; i < 7; ++i) {
			values[IDX(i)] = sum;
		}
#undef IDX
	}
#endif
}

template <typename T> class NeuralRRSLoss : public Loss<T> {
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

		linear_kernel(neural_rrs_loss_L_L2<T>, 0, stream, prediction.n_elements(), stride, dims,
					  loss_scale, prediction.data(), target.data(), values.data(), gradients.data(),
					  data_pdf ? data_pdf->data() : nullptr, mClampOn, mClampMax, mStep
#ifdef BB_TCNN_DEBUG_MODE
					  ,
					  mShowLossIndex
#endif
		);

		const float *thpPtr	  = mThp + mOffset * 3;		// sizeof(mThp)/sizeof(float) = 3
		const float *pdfPtr	  = mPdf + mOffset * 3;		// sizeof(mPdf)/sizeof(float) = 3
		const float *errorPtr = mError + mOffset * 1;	// sizeof(mError)/sizeof(float) = 1
		const float *refPtr	  = mRefMean + mOffset * 1; // sizeof(mRefMean)/sizeof(float) = 1

		linear_kernel(neural_rrs_loss_rrs<T>, 0, stream, prediction.n_elements() / stride, stride,
					  dims, loss_scale, mStep, thpPtr, pdfPtr, errorPtr, refPtr,
					  mLossSumErrorGPUPtr, mSampleWeight, mPixels, prediction.data(), target.data(),
					  values.data(), gradients.data(), data_pdf ? data_pdf->data() : nullptr,
					  mClampOn, mClampMax
#ifdef BB_TCNN_DEBUG_MODE
					  ,
					  mShowLossIndex
#endif
		);
	}

	void update_hyperparams(const json &params) override {
		mOffset		   = params.value("offset", mOffset);
		mClampMax	   = params.value("clamp_max", mClampMax);
		mClampOn	   = params.value("clamp_on", mClampOn);
		mStep		   = params.value("step", mStep);
		mShowLossIndex = params.value("show_loss_index", mShowLossIndex);
		mPixels		   = params.value("pixels", mPixels);

		mThp		  = (float *) params.value("thp", (uint64_t) mThp);
		mPdf		  = (float *) params.value("pdf", (uint64_t) mPdf);
		mError		  = (float *) params.value("error", (uint64_t) mError);
		mRefMean	  = (float *) params.value("ref_mean", (uint64_t) mRefMean);
		mSampleWeight = (float *) params.value("sample_weight", (uint64_t) mSampleWeight);
		mLossSumErrorGPUPtr =
			(float *) params.value("error_sum_ptr", (uint64_t) mLossSumErrorGPUPtr);

		if (!(params.size() == 1 && params.contains("offset"))) {
			printf("[NeuralRRSLoss] update hyperparams: %s\n", params.dump().c_str());
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "NeuralRRSLoss"}, {"clamp_max", mClampMax}, {"clamp_on", mClampOn},
			// {"step", mStep},
			// {"show_loss_index", mShowLossIndex},
		};
	}

	bool mClampOn{false};
	float mClampMax{500.0f};
	uint32_t mStep{1};
	uint32_t mShowLossIndex{0};

	uint32_t mPixels{1u}; // the number of pixels
	uint32_t mOffset{0};  // offset for read the following data
	float *mThp;
	float *mPdf;
	float *mError;
	float *mSampleWeight;
	float *mRefMean; // this is 1 floats for each element

	float *mLossSumErrorGPUPtr; // the GPU address of the sum of error
};

template <typename T>
__global__ void neural_rrs_loss_L_L2(const uint32_t n_elements, const uint32_t stride,
									 const uint32_t dims, const float loss_scale,
									 const T *__restrict__ predictions,
									 const float *__restrict__ targets, float *__restrict__ values,
									 T *__restrict__ gradients, const float *__restrict__ data_pdf,
									 const bool clampOn, const float clampMax, const uint32_t step
#ifdef BB_TCNN_DEBUG_MODE
									 ,
									 const uint32_t showLossIndex
#endif
) {

	const uint32_t j = threadIdx.x + blockIdx.x * blockDim.x;
	if (j >= n_elements) return;

	// why slower? you idiot! DIVERGENCE !!!
	// const uint32_t warp	 = 32;
	// const uint32_t warp2 = warp * warp;
	// const uint32_t j1 = j / warp2;
	// const uint32_t j2 = j % warp2;
	// const uint32_t j3 = j2 / warp;
	// const uint32_t j4 = j2 % warp;
	// const uint32_t i = j1 * warp2 + j4 * warp + j3;

	const uint32_t i = j;

	const uint32_t intra_elem_idx = i % stride;
	const uint32_t inter_elem_idx = i / stride;
	const uint32_t training_dims  = 6; // orignial is dims
	const uint32_t calc_dims	  = 3; // x calc data(x) & data(x+3)

	if (intra_elem_idx >= training_dims) {
		values[i]	 = 0;
		gradients[i] = 0;
		return;
	}
	if (intra_elem_idx >= calc_dims) return; // the difference is not set 0

	const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;
	// TODO: calc_dims = training_dims - calc_dims => correct
	const uint32_t n_total = n_elements / stride * calc_dims;
	const float pdf		   = data_pdf ? data_pdf[target_idx] : 1;

	{ // L
		const float prediction = (float) predictions[i];

		float r = (float) predictions[i - intra_elem_idx + 0];
		float g = (float) predictions[i - intra_elem_idx + 1];
		float b = (float) predictions[i - intra_elem_idx + 2];

		const float luminance = (0.299f * r + 0.587f * g + 0.114f * b);

		const float prediction_sq_plus_epsilon = luminance * luminance + NRRS_EPSILON;

		const float difference = prediction - targets[target_idx];

		float v = difference * difference / prediction_sq_plus_epsilon / pdf / n_total;

		float scale = 1.0f;
		if (clampOn) {
			scale = v > clampMax ? clampMax / v : 1.0f;
		}
		values[i] = scale * v;

		float gradient = 2 * difference / prediction_sq_plus_epsilon / pdf;
		gradients[i]   = (T) (scale * loss_scale * gradient / n_total);
	}
	{ // L^2
		float scale							   = 0.01f;
		const float prediction				   = (float) predictions[i + 3];
		const float prediction_sq_plus_epsilon = prediction * prediction + NRRS_EPSILON;
		const float difference				   = prediction - targets[target_idx + 3];

		float v = scale * difference * difference / prediction_sq_plus_epsilon / pdf / n_total;
		if (clampOn) {
			const float scale1 = v > clampMax ? clampMax / v : 1.0f;
			v *= scale1;
			scale *= scale1;
		}

		values[i + 3]	 = v;
		float gradient	 = 2 * difference / prediction_sq_plus_epsilon / pdf;
		gradients[i + 3] = (T) (scale * loss_scale * gradient / n_total);
	}
}

// TODO: stride = 16, so the output dim shoukld <= 16
// Memory Aceess Divergence
template <typename T>
__global__ void
neural_rrs_loss_L_L2_v2(const uint32_t n_elements, const uint32_t stride, const uint32_t dims,
						const float loss_scale, const T *__restrict__ predictions,
						const float *__restrict__ targets, float *__restrict__ values,
						T *__restrict__ gradients, const float *__restrict__ data_pdf,
						const bool clampOn, const float clampMax
#ifdef BB_TCNN_DEBUG_MODE
						,
						const uint32_t showLossIndex
#endif
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t prediction_idx = i * stride;
	const uint32_t target_idx	  = i * dims;

	const float pdf = data_pdf ? data_pdf[target_idx] : 1;

	{ // L
		const uint32_t n_total = n_elements * 3;

		float r = (float) predictions[prediction_idx + 0];
		float g = (float) predictions[prediction_idx + 1];
		float b = (float) predictions[prediction_idx + 2];

		const float luminance				   = (0.299f * r + 0.587f * g + 0.114f * b);
		const float prediction_sq_plus_epsilon = luminance * luminance + NRRS_EPSILON;

		for (int j = 0; j < 3; ++j) {
			const uint32_t p_idx   = prediction_idx + j;
			const float prediction = (float) predictions[p_idx];
			const float difference = prediction - targets[target_idx + j];

			float v = difference * difference / prediction_sq_plus_epsilon / pdf / n_total;

			float scale = 1.0f;
			if (clampOn) {
				scale = v > clampMax ? clampMax / v : 1.0f;
			}
			values[p_idx] = scale * v;

			float gradient	 = 2 * difference / prediction_sq_plus_epsilon / pdf;
			gradients[p_idx] = (T) (scale * loss_scale * gradient / n_total);
		}
	}

	{ // L^2
		const uint32_t n_total = n_elements * 3;

		for (int j = 3; j < 6; ++j) {
			const uint32_t p_idx = prediction_idx + j;

			float scale							   = 0.01f;
			const float prediction				   = (float) predictions[p_idx];
			const float prediction_sq_plus_epsilon = prediction * prediction + NRRS_EPSILON;
			const float difference				   = prediction - targets[target_idx + j];

			float v = scale * difference * difference / prediction_sq_plus_epsilon / pdf / n_total;
			if (clampOn) {
				const float scale1 = v > clampMax ? clampMax / v : 1.0f;
				v *= scale1;
				scale *= scale1;
			}

			values[p_idx]	 = v;
			float gradient	 = 2 * difference / prediction_sq_plus_epsilon / pdf;
			gradients[p_idx] = (T) (scale * loss_scale * gradient / n_total);
		}
	}

	{ // last
		for (int j = 6; j < stride; ++j) {
			const uint32_t p_idx = prediction_idx + j;
			values[p_idx]		 = 0;
			gradients[p_idx]	 = 0;
		}
	}

	{
#ifdef BB_TCNN_DEBUG_MODE
		if (showLossIndex == 1) {
			values[i + 3] = values[i] = values[i] / 2;
		} else if (showLossIndex == 2) {
			values[i] = values[i + 3] = values[i + 3] / 2;
		}
#endif
	}
}

} // namespace tcnn
