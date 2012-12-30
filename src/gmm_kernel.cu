/* *
 * Copyright 2012.  All rights reserved.
 *
 * Please refer to the author end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

/*
 GMM (Gaussian Mixture Model) Estimation Kernels
 */
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// includes, project
#include "gmm_types.h"

/**
 *  Subtract the vectors X and U, for the calculation of g(x) function
 */
__device__ void sub_vectors(float *x, float *u, float result[DATA_SIZE]) {
	for (short int i = 0; i < DATA_SIZE; i++)
		result[i] = x[i] - u[i];
}

/**
 *  Multiply the vector X for the covariance matrix S^-1 and for the X',
 *  for the calculation of g(x) function
 */
__device__ float mul_vector_matrix2(float x[DATA_SIZE], float *inv_cov) {
	float aux;
	float result = 0;
	for (short int i = 0; i < DATA_SIZE; i++) {
		aux = 0;
		for (short int j = 0; j < DATA_SIZE; j++)
			aux += x[j] * inv_cov[j * DATA_SIZE + i];
		result += aux * x[i];
	}
	return result;
}

/**
 *  Calculate the likelihood function g(x), given ...
 */
__device__ float gxFunction(float *data, float *means, float *inv_cov,
		float det, float fact_pi, int gausIdx, int dataIdx) {
	float vector_sub[DATA_SIZE];
	sub_vectors(data, means, vector_sub);
	float exp_factor = -0.5f * mul_vector_matrix2(vector_sub, inv_cov);
	float factor_a = 1 / (fact_pi * sqrtf(det));
	return (factor_a * __expf(exp_factor));
}

/**
 *               <<< FIRST PHASE OF ALGORITHM >>>
 * Computes the likelihood of all samples to each gaussian from mixture.
 * Each thread works with a sample at time, but is able to compute for
 * any numbers of threads, in case that.
 */
__global__ void p_kernel_old(float *likelihood, float *data, float *means,
		float *inv_cov, float *det, float *pi, float fact_pi) {

	int dataIdx = blockIdx.x;
	int gausIdx = blockIdx.y;

	// While the data index is valid, calculate yours likelihoods
	while (dataIdx < DB_SIZE) {
		// Calculate likelihood for each gaussian
		likelihood[gausIdx * DB_SIZE + dataIdx] = gxFunction(
				(data + dataIdx * DATA_SIZE), (means + gausIdx * DATA_SIZE),
				(inv_cov + gausIdx * (DATA_SIZE * DATA_SIZE)), det[gausIdx],
				fact_pi, gausIdx, dataIdx) * pi[gausIdx]; // + 1.e-29f;
		// Retrieve the next data index
		dataIdx += gridDim.x;
	}
}

/**
 *  Multiply the vector X for the inverse covariance matrix S^-1 and for the X',
 *  for the calculation of g(x) function
 */
__global__ void mul(DeviceData d, const unsigned char DT_N, const unsigned int DB_N, int shared_a, int shared_b) {
	extern __shared__ float *shared;
	float *cache1 = &shared[0];
	float *cache2 = &shared[shared_a]; //4 * DT_N * sizeof(float)];
	float *matrix = &shared[shared_b]; // 4 * 16 * sizeof(float)];

	/*
	__shared__ float cache1[4][DT_N];
	__shared__ float cache2[4][16];
	__shared__ float matrix[4][DT_N][DT_N];
	*/

	int sampleIdx = threadIdx.y + blockIdx.x * blockDim.y;
	int gausIdx = blockIdx.y;
	int tidy = threadIdx.y; // Which data position of block
	int tidx = threadIdx.x; // which component of the data

	while (sampleIdx < DB_N) {
		//Initializing cache 2 (for reduction)
		cache2[tidy][tidx] = 0;

		if (tidx < DT_N) {
			// loading sample on the shared memory
			cache1[tidy][tidx] = d.samples[tidx + sampleIdx * DT_N];
			// subtracting the mean on the shared memory
			cache1[tidy][tidx] -= d.means[tidx + gausIdx * DT_N];

			// loading inverse covariance matrix
			for (int i = 0; i < DT_N; i++)
				matrix[tidy][tidx][i] =
						d.inv_covariance_matrices[(gausIdx * (DT_N * DT_N))
								+ (tidx * DT_N + i)];

			__syncthreads();

			// calculating (x - u) * S^-1
			for (int i = 0; i < DT_N; i++)
				cache2[tidy][tidx] += cache1[tidy][i] * matrix[tidy][i][tidx];

			cache2[tidy][tidx] *= cache1[tidy][tidx];
		}

		__syncthreads();

		// Execute reduction to sum the likelihoods from this data.
		// The result sum will be stored in the position 0 in the cache.
		int i = (blockDim.x / 2);
		while (i != 0) {
			if (tidx < i) {
				cache2[tidy][tidx] += cache2[tidy][tidx + i];
				// Waiting all threads accomplish its sum
				__syncthreads();
			}
			i /= 2;
		}

		// Stores the result value in the likelihood matrix
		if (tidx == 0)
			d.likelihood_matrix[gausIdx * DB_N + sampleIdx] = cache2[tidy][0];

		sampleIdx += gridDim.x * blockDim.y;
	}
}

/*
 * Computes the likelihood of all samples to each gaussian from mixture.
 * Each thread works with a sample at time, but is able to compute for
 * any numbers of threads, in case that.
 */
__global__ void p_kernel(DeviceData d) {

	int sampleIdx = threadIdx.x + threadIdx.y * blockDim.x
			+ blockIdx.x * blockDim.x * blockDim.y;
	int gausIdx = blockIdx.y;

	// While the data index is valid, calculate yours likelihoods
	while (sampleIdx < DB_SIZE) {

		// Calculate the likelihood value
		float factor_b = -0.5f
				* d.likelihood_matrix[gausIdx * DB_SIZE + sampleIdx];
		float factor_a = 1 / (d.factor_pi * sqrtf(d.determinants[gausIdx]));
		float result = (factor_a * __expf(factor_b)) * d.weights[gausIdx];

		d.likelihood_matrix[gausIdx * DB_SIZE + sampleIdx] = result;

		// Retrieve the next data index
		sampleIdx += gridDim.x * blockDim.x * blockDim.y;
	}
}

/*
 *
 */
__global__ void pn_kernel(DeviceData d) {

}

void gmm(DeviceData d_data, ofstream *myfile, double *timeTotal,
		unsigned char dtsize, unsigned int dbsize) {

	float duration, duration_T = 0;
	cudaEvent_t start, stop;
	cudaEvent_t start_T, stop_T;
	cudaEventCreate(&start_T);
	cudaEventCreate(&stop_T);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 dimGrid(1, GMM_SIZE, 1);
	dim3 dimBlock(16, 4, 1);

	dim3 dimGrid2(4, GMM_SIZE, 1);
	dim3 dimBlock2(8, 8, 1);

//	dim3 dimGrid3(2048, GMM_SIZE);

	cudaDeviceSynchronize();
	cudaEventRecord(start_T, 0);
	cudaEventRecord(start, 0);
	int count_a = 4 * DATA_SIZE * sizeof(float);
	int count_b = 4 * 16 * sizeof(float);
	int shared_size = (4 * DATA_SIZE * DATA_SIZE * sizeof(float) ) + count_a + count_b;
	mul<<<dimGrid, dimBlock, shared_size>>>(d_data, DATA_SIZE, DB_SIZE, count_a,count_b);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	*myfile << "mul , " << duration << "\n";

	cudaDeviceSynchronize();
	cudaEventRecord(start, 0);
	p_kernel<<<dimGrid2, dimBlock2>>>(d_data);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	*myfile << "p_kernel , " << duration << "\n";

//	p_kernel_old<<<dimGrid3, 1>>>(d_data.likelihood_matrix, d_data.samples, d_data.means,
//			d_data.inv_covariance_matrices, d_data.determinants, d_data.weights,
//			d_data.factor_pi);

	cudaDeviceSynchronize();
	cudaEventRecord(stop_T, 0);
	cudaEventSynchronize(stop_T);
	cudaEventElapsedTime(&duration_T, start_T, stop_T);
	*timeTotal = (double) duration_T;
}
