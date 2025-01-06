#pragma once

#define BB_TCNN_DEBUG_MODE

#define BB_RRS_LOSS_SCALE_STEP2 1e0f
#define BB_SIGMOID_SCALE 20.0f
#define BB_L2_OFFSET 3
#define BB_SIGMA_OFFSET 3
#define OUTPUT_RRS_OFFSET 0

#define NRRS_EPSILON 1e-2f

__device__ inline float bb_activation(float x) {
	// x < 0: softplus(x) = log(1 + exp(x))
	// x >= 0: 0.5 * x + ln2
	return x < 0 ? logf(1 + expf(x)) : 0.5f * x + 0.6931471805599453f;
}

__device__ inline void bb_activation_and_gradient(float &x, float &dx) {
	// activation: softplus & y = 0.5x + ln2
	const float x_exp	= expf(x);
	const bool x_is_neg = x < 0;
	const float ln2		= 0.6931471805599453f;
	x					= x_is_neg ? (logf(x_exp + 1.0f)) : (0.5f * x + ln2);
	dx					= x_is_neg ? (x_exp / (1.0f + x_exp)) : 0.5f;
}