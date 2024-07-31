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
					 const bool clampOn = false, const float clampMax = 10.0f
#ifdef BB_TCNN_DEBUG_MODE
					 ,
					 const uint32_t showLossIndex = 0
#endif
);

template <typename T>
__global__ void
neural_rrs_loss_rrs(const uint32_t n_elements, const uint32_t stride, const uint32_t dims,
					const float loss_scale, const uint32_t step, const float *__restrict__ thp,
					const T *__restrict__ predictions, const float *__restrict__ targets,
					float *__restrict__ values, T *__restrict__ gradients,
					const float *__restrict__ data_pdf = nullptr, const bool clampOn = false,
					const float clampMax = 10.0f
#ifdef BB_TCNN_DEBUG_MODE
					,
					const uint32_t showLossIndex = 0
#endif
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t prediction_idx = i * stride;
	const uint32_t target_idx	  = i * 3; // thp: dim = 3

	float rrs_gt = 1.0f;

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
		rrs_gt = fmaxf(0.5f, fminf(20.0f, rrs_gt));
	}

	const uint32_t prediction_rrs_idx = prediction_idx + 6;

	const float prediction = (float) predictions[prediction_rrs_idx];
	const float difference = 5.0f * (prediction - rrs_gt);
	const uint32_t n_total = n_elements;
	const float pdf		   = data_pdf ? data_pdf[target_idx] : 1;

	const float prediction_sq_plus_epsilon = prediction * prediction + 0.01f;

	float v = difference * difference / prediction_sq_plus_epsilon / pdf / n_total;

	float scale = 1.0f;
	if (clampOn) {
		scale = v > clampMax ? clampMax / v : 1.0f;
	}
	values[prediction_rrs_idx] = scale * v;

	float gradient				  = 2 * difference / prediction_sq_plus_epsilon / pdf;
	gradients[prediction_rrs_idx] = (T) (scale * loss_scale * gradient / n_total);

#ifdef BB_TCNN_DEBUG_MODE
	if (showLossIndex == 3) {
		for (int i = 0; i < 7; ++i) {
			values[prediction_idx + i] = values[prediction_idx + 6] / 7;
		}
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

		// only training L, L^2
		linear_kernel(neural_rrs_loss_L_L2<T>, 0, stream, prediction.n_elements(), stride, dims,
					  loss_scale, prediction.data(), target.data(), values.data(), gradients.data(),
					  data_pdf ? data_pdf->data() : nullptr, mClampOn, mClampMax
#ifdef BB_TCNN_DEBUG_MODE
					  ,
					  mShowLossIndex
#endif
		);

		linear_kernel(neural_rrs_loss_rrs<T>, 0, stream, prediction.n_elements() / stride, stride,
					  dims, loss_scale, mStep, mThp, prediction.data(), target.data(),
					  values.data(), gradients.data(), data_pdf ? data_pdf->data() : nullptr,
					  mClampOn, mClampMax
#ifdef BB_TCNN_DEBUG_MODE
					  ,
					  mShowLossIndex
#endif
		);
	}

	void update_hyperparams(const json &params) override {
		mClampMax	   = params.value("clamp_max", mClampMax);
		mClampOn	   = params.value("clamp_on", mClampOn);
		mStep		   = params.value("step", mStep);
		mShowLossIndex = params.value("show_loss_index", mShowLossIndex);
		mThp		   = (float *) params.value("thp", (uint64_t) mThp);
		printf("[neural_rrs_loss] update hyperparams!\n");
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
	float *mThp;
};

template <typename T>
__global__ void neural_rrs_loss_L_L2(const uint32_t n_elements, const uint32_t stride,
									 const uint32_t dims, const float loss_scale,
									 const T *__restrict__ predictions,
									 const float *__restrict__ targets, float *__restrict__ values,
									 T *__restrict__ gradients, const float *__restrict__ data_pdf,
									 const bool clampOn, const float clampMax
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

		const float prediction_sq_plus_epsilon = luminance * luminance + 0.01f;

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
		const float prediction_sq_plus_epsilon = prediction * prediction + 0.01f;
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
		const float prediction_sq_plus_epsilon = luminance * luminance + 0.01f;

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
			const float prediction_sq_plus_epsilon = prediction * prediction + 0.01f;
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
