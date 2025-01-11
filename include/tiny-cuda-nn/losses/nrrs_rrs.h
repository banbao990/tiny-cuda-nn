#pragma once
#include "nrrs.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/reduce_sum.h>

namespace tcnn {

#define LL2_STRIDE 6

__global__ void prinf_grad(const uint32_t n_elements, float *grad, const uint32_t offset,
						   const uint32_t step) {
	const uint32_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx >= n_elements) return;

	if (offset == 0 && step == 3) {
		printf("max abs(grad): %g\n", grad[0]);
		grad[0] = 0.0f;
	}
}

template <typename T>
__global__ void nrrs_rrs_loss(
	const uint32_t n_elements, const uint32_t dims, const float loss_scale, const uint32_t step,
	const float *__restrict__ thp, const float *__restrict__ pdf, const float *__restrict__ error,
	const float *__restrict__ ref_mean, const float *__restrict__ error_sum,
	const float *__restrict__ sample_weight, const __half *__restrict__ ll2,
	const uint32_t pixels_num, const T *__restrict__ predictions, const float *__restrict__ targets,
	float *__restrict__ values, T *__restrict__ gradients,
	const float *__restrict__ data_pdf = nullptr, const bool clampOn = false,
	const float clampMax = 10.0f, const bool trainSigma = true, const float gamma1 = 1.0f,
	const float gamma2 = 1.0f, const float gamma3 = 1.0f
#ifdef BB_TCNN_DEBUG_MODE
	,
	const float *error_per_pixel = nullptr, const uint32_t showLossIndex = 0,
	const uint32_t *pixelID = nullptr, const int32_t debugPixel = -1,
	uint32_t *pixel_debug_buffer = nullptr, float pdf_lower_bound = 0.5f, float *grad_max = nullptr
#endif
) {

	const uint32_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx >= n_elements) return;

	constexpr uint32_t stride	  = 16;
	const uint32_t prediction_idx = thread_idx * stride;

#ifdef BB_TCNN_DEBUG_MODE
	float grad_avg = 0.0f;
	float grad_min = 0.0f;
	float grad_rrs = 0.0f;
#endif

	for (int i = 1; i < stride; i++) {
		values[prediction_idx + i]	  = 0;
		gradients[prediction_idx + i] = 0;
	}

	if (step == 3) {

#ifdef BB_TCNN_DEBUG_MODE
		uint32_t c_pixelId = pixelID[thread_idx];
		{
			float c_error			= error[thread_idx];
			float c_error_per_pixel = error_per_pixel[c_pixelId];

			if (c_error != c_error_per_pixel) {
				printf("error[%d]: %g, error_per_pixel[%d]: %g\n", thread_idx, c_error, c_pixelId,
					   c_error_per_pixel);
			}
		}
#endif
		// now the output is rrs
		const float var_sum = *error_sum;

		const float rrs_loss_scale = 1e0f;
		const uint32_t n_total	   = n_elements;

		float rrs = (float) predictions[prediction_idx];

#ifdef BB_TCNN_DEBUG_MODE
		// constexpr float sMaxRRS = 1e2f;
		// if (isinf(rrs) || rrs >= sMaxRRS) {
		// 	printf("rrs [%d]: rrs = %g\n", thread_idx, rrs);
		// 	rrs = fminf(sMaxRRS, rrs);
		// }
#endif

		const float var		 = error[thread_idx];
		const float path_pdf = fminf(pdf[thread_idx], 1.0f);
		// const float path_pdf		 = pdf[thread_idx];
		const float pixel_err_weight = sample_weight[thread_idx];

		float dactivate_drrs = 1.0f;
		bb_activation_and_gradient(rrs, dactivate_drrs);

		{ // k = 1
			const uint32_t ll2_idx = thread_idx * LL2_STRIDE;

			const float r  = max((float) ll2[ll2_idx + 0], 0.0f);
			const float g  = max((float) ll2[ll2_idx + 1], 0.0f);
			const float b  = max((float) ll2[ll2_idx + 2], 0.0f);
			const float ex = (r + g + b) / 3.0f;

			const uint32_t thp_idx = thread_idx * 3;

			const float rp = thp[thp_idx + 0];
			const float gp = thp[thp_idx + 1];
			const float bp = thp[thp_idx + 2];

			// const float path_pdf = (rp + gp + bp) / 3.0f;

			const float rrs_gt_step2 = (r * rp + g * gp + b * bp) / 3.0f;
			const float t_ref_mean	 = ref_mean[thread_idx];

			// const float rrs_center = t_ref_mean;
			// const float rrs_center = (rrs_gt_step2 + t_ref_mean) / 2.0f;
			// const float rrs_center = 0.0f;
			const float rrs_center = 1.0f;

			const float net_data_pdf = data_pdf ? data_pdf[thread_idx] : 1.0f;

			// loss
			const float e1 = var - var_sum / pixels_num;

#define BB_L1_L2_k 1

#if BB_L1_L2_k == 1
			float loss_value =
				rrs_loss_scale * (pixel_err_weight * (gamma1 * abs(e1) + gamma2 * var) +
								  (gamma3 * (rrs - rrs_center) * (rrs - rrs_center)));
#elif BB_L1_L2_k == 2
			float loss_value =
				rrs_loss_scale * (pixel_err_weight * (gamma1 * e1 * e1 + gamma2 * var * var) +
								  (gamma3 * (rrs - rrs_center) * (rrs - rrs_center)));
#endif

			loss_value /= net_data_pdf * n_total;

			values[prediction_idx] = loss_value;

			// gradient

#if BB_L1_L2_k == 1
			float dE_dvar =
				gamma1 * ((e1 > 0 ? 1 : -1) * (float(pixels_num - 1) / float(pixels_num))) + gamma2;
#elif BB_L1_L2_k == 2
			float dE_dvar =
				gamma1 * (2 * e1 * (float(pixels_num - 1) / float(pixels_num))) + gamma2 * 2 * var;
#endif

			// dE_dvar /= (var + 1); // var = log(var + 1)

			// dE_dvar *= expf(var) - 1.0f; // log scale

			// rel var
			// float rel_inv = ref_mean[thread_idx];
			// rel_inv		  = max(rel_inv * rel_inv, 1e-4f);
			// rel_inv		  = rel_inv > 0 ? 1.0f / rel_inv : 0;
			// TODO: this may not needed as we divided by it when use it
			// constexpr float rel_inv = 1.0f;

			// output is rrs
			float rel_inv = 1.0f / fmax(t_ref_mean * t_ref_mean, NRRS_EPSILON);

			// output is rrs*ref_mean
			// float rel_inv = 1.0f / fmax(t_ref_mean * t_ref_mean * t_ref_mean, NRRS_EPSILON);

			float path_var = 0.0f;
			// {

			const float r2 = max((float) ll2[ll2_idx + 3], 0.0f);
			const float g2 = max((float) ll2[ll2_idx + 4], 0.0f);
			const float b2 = max((float) ll2[ll2_idx + 5], 0.0f);

			if (trainSigma) {
				const float sigma =
					(bb_activation(r2) + bb_activation(g2) + bb_activation(b2)) / 3.0f;
				// S : sigma*sigma
				// RR: ex2
				float ex_ex = (rrs >= 1.0f) ? 0.0f : ex * ex;
				path_var	= sigma * sigma + ex_ex;
				path_var	= max(path_var, 0.0f);
			} else {
				const float out2 = (r2 + g2 + b2) / 3.0f;
				float ex_ex		 = (rrs >= 1.0f) ? ex * ex : 0.0f;
				path_var		 = max(out2 - ex_ex, 0.0f);
			}
			// }
			const float dvar_drrs = -path_pdf * path_var / max(rrs * rrs, NRRS_EPSILON);
			float grad			  = loss_scale * rrs_loss_scale *
						 (pixel_err_weight * (dE_dvar * rel_inv * dvar_drrs) +
						  gamma3 * 2 * (rrs - rrs_center)) *
						 dactivate_drrs;

			// grad = fmaxf(-1e1f, fminf(1e1f, grad));
			grad /= net_data_pdf * n_total;

			// grad = fminf(1e-2f, fmax(grad, -1e-2f));

			gradients[prediction_idx] = (T) (grad);

#ifdef BB_TCNN_DEBUG_MODE
			// atomicMax(grad_max, abs(grad));

			grad_avg =
				loss_scale * rrs_loss_scale *
				(pixel_err_weight *
				 ((gamma1 * ((e1 > 0 ? 1 : -1) * (float(pixels_num - 1) / float(pixels_num)))) *
				  rel_inv * dvar_drrs)) *
				dactivate_drrs / (net_data_pdf * n_total);
			grad_min = loss_scale * rrs_loss_scale *
					   (pixel_err_weight * (gamma2 * rel_inv * dvar_drrs)) * dactivate_drrs /
					   (net_data_pdf * n_total);
			grad_rrs = loss_scale * rrs_loss_scale * (gamma3 * 2 * (rrs - rrs_center)) *
					   dactivate_drrs / (net_data_pdf * n_total);
			if (path_pdf > pdf_lower_bound) {
				atomicAdd(pixel_debug_buffer + c_pixelId, 1u);
			}
#endif

#ifdef BB_TCNN_DEBUG_MODE
			const bool should_debug =
				(debugPixel != -1) && (pixelID != nullptr) && (pixelID[thread_idx] == debugPixel);
			// check nan
			if (should_debug || isnan(grad) || isinf(grad) || isnan(rrs) || abs(grad) >= 1e1 ||
				rrs < 0) {
				printf("rrs [%d]: rrs = %g, grad(avg + min + rrs) = %g = %g + %g + %g\n" // 1

					   "var = %g, path_pdf = %g, path_var = %g, "
					   "dvar_drrs = %g, dE_dvar = %g, rel_inv = %g, e1 = %g, var_sum = %g, "
					   "pixel_err_weight = %g, ref_mean = %g, ex = %g, rrs_gt_step2 = %g, "
					   "net_data_pdf = %g, loss_value = %g, pixelId: %d\n\n",

					   thread_idx, rrs, grad, grad_avg, grad_min, grad_rrs, // 1

					   var, path_pdf, path_var, dvar_drrs, dE_dvar, rel_inv, e1, var_sum,
					   pixel_err_weight, t_ref_mean, ex, rrs_gt_step2, net_data_pdf, loss_value,
					   c_pixelId);
			}
#endif
		}

		// values[prediction_idx]	  = 0;
		// gradients[prediction_idx] = 0;

	} else {
		const uint32_t thp_idx	  = thread_idx * 3; // thp: dim = 3
		const uint32_t target_idx = thread_idx;		// dim3 = 1

		float rrs_gt			   = 1.0f;
		const float rrs_loss_scale = 1.0f;
		// float luminance = 1.0f;
		if (step == 1) {
			// set all rrs = 1
			// rrs_gt = ref_mean[thread_idx];
		} else if (step == 2) {
			// step 2: ADRRS
			const uint32_t ll2_idx = thread_idx * LL2_STRIDE;

			const float r  = max((float) ll2[ll2_idx + 0], 0.0f);
			const float g  = max((float) ll2[ll2_idx + 1], 0.0f);
			const float b  = max((float) ll2[ll2_idx + 2], 0.0f);
			const float rp = thp[thp_idx + 0];
			const float gp = thp[thp_idx + 1];
			const float bp = thp[thp_idx + 2];

			rrs_gt = (r * rp + g * gp + b * bp) / 3.0f;

			const float refI = ref_mean[thread_idx];
			rrs_gt			 = rrs_gt / (refI + NRRS_EPSILON);

			// clamp
			// rrs_gt = fmaxf(0.5f, fminf(20.0f, rrs_gt));
			// luminance = (0.299f * r + 0.587f * g + 0.114f * b);
		}

		const float prediction_ori = (float) predictions[prediction_idx];
		// activation: sigmoid
		// const float prediction = 1.0f / (1.0f + expf(-prediction_ori));
		// activation: softplus
		// const float prediction_ori_exp = expf(prediction_ori);
		// const float prediction = log1pf(prediction_ori_exp);
		float prediction	 = prediction_ori;
		float dactivate_drrs = 1.0f;
		bb_activation_and_gradient(prediction, dactivate_drrs);

		// prediction	   = 1.0f / (1.0f + expf(-prediction_ori));
		// dactivate_drrs = prediction * (1.0f - prediction) * BB_SIGMOID_SCALE;
		// prediction *= BB_SIGMOID_SCALE;

		const float difference = rrs_loss_scale * dactivate_drrs * (prediction - rrs_gt);

		const uint32_t n_total = n_elements; // /stride; // why error?
		const float pdf		   = data_pdf ? data_pdf[target_idx] : 1;

		// const float prediction_sq_plus_epsilon = luminance * luminance + NRRS_EPSILON;
		const float prediction_sq_plus_epsilon = prediction * prediction + NRRS_EPSILON;

		float v = difference * difference / prediction_sq_plus_epsilon / pdf / n_total;

		float scale = 1.0f;
		if (clampOn) {
			scale = v > clampMax ? clampMax / v : 1.0f;
		}
		values[prediction_idx] = BB_RRS_LOSS_SCALE_STEP2 * scale * v;

		float gradient = 2 * difference / prediction_sq_plus_epsilon / pdf;
		// sigmoid
		// gradient *= prediction * (1.0f - prediction);
		// softplus
		// gradient *= prediction_ori_exp / (1.0f + prediction_ori_exp);
		gradients[prediction_idx] =
			(T) (BB_RRS_LOSS_SCALE_STEP2 * scale * loss_scale * gradient / n_total);

		// if (isnan(v) || isinf(v) || isnan(gradient) || isinf(gradient)) {
		// 	printf("rrs [%d]: v = %g, gradient = %g, rrs_gt = %g, prediction = %g, L = %g\n",
		// 		   thread_idx, v, gradient, rrs_gt, prediction,
		// 		   (float) predictions[prediction_idx + 0]);
		// }

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
	if (showLossIndex >= 4 && showLossIndex <= 7) {
		float sum = 0;
		if (showLossIndex == 4) {
			// grad all
			sum = (float) gradients[IDX(0)];
		} else if (showLossIndex == 5) {
			// grad avg
			sum = grad_avg;
		} else if (showLossIndex == 6) {
			// grad min
			sum = grad_min;
		} else if (showLossIndex == 7) {
			// grad rrs
			sum = grad_rrs;
		}
		values[IDX(0)] = sum;
#undef IDX
	}
#endif
}

template <typename T> class NRRSRRSLoss : public Loss<T> {
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

		CHECK_THROW(stride == 16);

		const float *thpPtr			 = mThp + mOffset * 3;			// 3 float
		const float *pdfPtr			 = mPdf + mOffset * 1;			// 1 float
		const float *errorPtr		 = mError + mOffset * 1;		// 1 float
		const float *sampleWeightPtr = mSampleWeight + mOffset * 1; // 1 float
		const float *refPtr			 = mRefMean + mOffset * 1;		// 1 float
		const uint32_t *pixelIDPtr	 = mPixelID + mOffset * 1;		// 1 uint32_t

		const __half *ll2Ptr = mLL2; // update each training batch

#ifdef BB_TCNN_DEBUG_MODE
		// if (mDebugPixel != -1) {
		// 	printf("debug pixel: %d, offset: %d\n", mDebugPixel, mOffset);
		// }
		// linear_kernel(prinf_grad, 0, stream, 1, mGradMax, mOffset, mStep);
#endif

		linear_kernel(nrrs_rrs_loss<T>, 0, stream, prediction.n_elements() / stride, dims,
					  loss_scale, mStep, thpPtr, pdfPtr, errorPtr, refPtr, mLossSumErrorGPUPtr,
					  sampleWeightPtr, ll2Ptr, mPixels, prediction.data(), target.data(),
					  values.data(), gradients.data(), data_pdf ? data_pdf->data() : nullptr,
					  mClampOn, mClampMax, mTrainSigma, mGamma1, mGamma2, mGamma3

#ifdef BB_TCNN_DEBUG_MODE
					  ,
					  mErrorPerPixel, mShowLossIndex, pixelIDPtr, mDebugPixel, mPixelDebugBuffer,
					  mPdfLoweBound, mGradMax
#endif
		);
	}

	void update_hyperparams(const json &params) override {
		// frequently update offset
		if (params.size() == 1 && params.contains("offset")) {
			mOffset = params.value("offset", mOffset);
			return;
		}

		mDebugPixel	   = params.value("debug_pixel_id", mDebugPixel);
		mClampMax	   = params.value("clamp_max", mClampMax);
		mClampOn	   = params.value("clamp_on", mClampOn);
		mTrainSigma	   = params.value("train_sigma", mTrainSigma);
		mStep		   = params.value("step", mStep);
		mShowLossIndex = params.value("show_loss_index", mShowLossIndex);
		mPixels		   = params.value("pixels", mPixels);

		mThp		  = (float *) params.value("thp", (uint64_t) mThp);
		mPdf		  = (float *) params.value("pdf", (uint64_t) mPdf);
		mError		  = (float *) params.value("error", (uint64_t) mError);
		mRefMean	  = (float *) params.value("ref_mean", (uint64_t) mRefMean);
		mSampleWeight = (float *) params.value("sample_weight", (uint64_t) mSampleWeight);
		mLL2		  = (__half *) params.value("ll2", (uint64_t) mLL2);
		mLossSumErrorGPUPtr =
			(float *) params.value("error_sum_ptr", (uint64_t) mLossSumErrorGPUPtr);
		mPixelID = (uint32_t *) params.value("pixel_id", (uint64_t) mPixelID);

		mErrorPerPixel = (float *) params.value("error_per_pixel", (uint64_t) mErrorPerPixel);
		mPixelDebugBuffer =
			(uint32_t *) params.value("pixel_debug_buffer", (uint64_t) mPixelDebugBuffer);

		mGamma1 = params.value("gamma1", mGamma1);
		mGamma2 = params.value("gamma2", mGamma2);
		mGamma3 = params.value("gamma3", mGamma3);

		mPdfLoweBound = params.value("pdf_lower_bound", mPdfLoweBound);

		printf("[NRRS_RRS Loss] update hyperparams: %s\n", params.dump().c_str());

		if (mGradMax == nullptr) {
			cudaMalloc(&mGradMax, sizeof(float));
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "NeuralRRSLoss"}, {"clamp_max", mClampMax}, {"clamp_on", mClampOn},
			// {"step", mStep},
			// {"show_loss_index", mShowLossIndex},
		};
	}

	virtual ~NRRSRRSLoss() {
		if (mGradMax) {
			cudaFree(mGradMax);
			mGradMax = nullptr;
		}
	}

	float *mGradMax{nullptr};

	bool mClampOn{false};
	float mClampMax{500.0f};
	uint32_t mStep{1};
	uint32_t mShowLossIndex{0};

	int32_t mDebugPixel{-1}; // debug training data
	bool mTrainSigma{true};	 // train sigma or X2

	uint32_t mPixels{1u}; // the number of pixels
	uint32_t mOffset{0};  // offset for read the following data
	float *mThp;
	float *mPdf;
	float *mError;
	float *mSampleWeight;
	__half *mLL2;		// the l,l2 for each element
	float *mRefMean;	// this is 1 floats for each element
	uint32_t *mPixelID; // the pixel id for each element

	float *mErrorPerPixel;		 // the error per pixel [Debug]
	uint32_t *mPixelDebugBuffer; // [Debug]

	float mGamma1{1.0f};
	float mGamma2{1.0f};
	float mGamma3{1.0f};

	float mPdfLoweBound{0.01f};

	float *mLossSumErrorGPUPtr; // the GPU address of the sum of error
};

} // namespace tcnn

#undef BB_TCNN_DEBUG_MODE
