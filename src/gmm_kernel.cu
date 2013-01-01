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

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}
};

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
template <class T>
__global__ void mul(DeviceData d, const unsigned char DT_N,
		const unsigned int DB_N) {

	int pt_a = DT_N * blockDim.y;
	int pt_b = pt_a + blockDim.x * blockDim.y;

	T *sdata = SharedMemory<T>();
	T *cache1 = &sdata[0];
	T *cache2 = &sdata[pt_a];
	T *matrix = &sdata[pt_b];

	int sampleIdx = threadIdx.y + blockIdx.x * blockDim.y;
	int gausIdx = blockIdx.y;
	int tidy = threadIdx.y; // Which data position of block
	int tidx = threadIdx.x; // which component of the data

	// loading matrix
	int lineId = threadIdx.y;
	while (lineId < DT_N && tidx < DT_N) {
		matrix[lineId * DT_N + tidx] = d.inv_covariance_matrices[(gausIdx
				* (DT_N * DT_N)) + (lineId * DT_N + tidx)];
		lineId += blockDim.y;
	}
	__syncthreads();

	while (sampleIdx < DB_N) {
		//Initializing cache 2 (for reduction)
		cache2[tidy * blockDim.x + tidx] = 0;

		if (tidx < DT_N) {
			// loading sample on the shared memory
			cache1[tidy * DT_N + tidx] = d.samples[tidx + sampleIdx * DT_N];
			// subtracting the mean on the shared memory
			cache1[tidy * DT_N + tidx] -= d.means[tidx + gausIdx * DT_N];

			__syncthreads();

			// calculating (x - u) * S^-1
			for (int i = 0; i < DT_N; i++)
				cache2[tidy * blockDim.x + tidx] += cache1[tidy * DT_N + i]
						* matrix[i * DT_N + tidx];

			cache2[tidy * blockDim.x + tidx] *= cache1[tidy * DT_N + tidx];
		}

		__syncthreads();

		// Execute reduction to sum the likelihoods from this data.
		// The result sum will be stored in the position 0 in the cache.
		int i = (blockDim.x / 2);
		while (i != 0) {
			if (tidx < i) {
				cache2[tidy * blockDim.x + tidx] += cache2[tidy * blockDim.x
						+ tidx + i];
				// Waiting all threads accomplish its sum
				__syncthreads();
			}
			i /= 2;
		}

		// Stores the result value in the likelihood matrix
		if (tidx == 0)
			d.likelihood_matrix[gausIdx * DB_N + sampleIdx] = cache2[tidy
					* blockDim.x];

		sampleIdx += gridDim.x * blockDim.y;
	}
}

/*
 * Computes the likelihood of all samples to each gaussian from mixture.
 * Each thread works with a sample at time, but is able to compute for
 * any numbers of threads, in case that.
 */
template <class T>
__global__ void p_kernel(DeviceData d) {

	int sampleIdx = threadIdx.x + threadIdx.y * blockDim.x
			+ blockIdx.x * blockDim.x * blockDim.y;
	int gausIdx = blockIdx.y;

	// While the data index is valid, calculate yours likelihoods
	while (sampleIdx < DB_SIZE) {

		// Calculate the likelihood value
		T factor_b = -0.5f
				* d.likelihood_matrix[gausIdx * DB_SIZE + sampleIdx];
		T factor_a = 1 / (d.factor_pi * sqrtf(d.determinants[gausIdx]));
		T result = (factor_a * __expf(factor_b)) * d.weights[gausIdx];

		d.likelihood_matrix[gausIdx * DB_SIZE + sampleIdx] = result;

		// Retrieve the next data index
		sampleIdx += gridDim.x * blockDim.x * blockDim.y;
	}
}

/*
 *
 */
template<class T>
__global__ void pn_kernel(DeviceData d, const unsigned char GMM_N,
		const unsigned int DB_N) {

	unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x; // sample index
	T mySum = 0;

	while (tidx < DB_N) {

		// First reduction phase
		for (int tidy = 0; tidy < GMM_N; tidy++)
			mySum += d.likelihood_matrix[tidx + tidy * DB_N];

		__syncthreads();

		// Normalize each likelihood value
		for (int tidy = 0; tidy < GMM_N; tidy++)
			d.likelihood_matrix[tidx + tidy * DB_N] /= mySum;

		// Retrieve the next data index
		tidx += gridDim.x * blockDim.x;
	}
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
	cudaDeviceSynchronize();
	cudaEventRecord(start_T, 0);
	cudaEventRecord(start, 0);
	int count_a = 4 * DATA_SIZE;
	int count_b = 4 * 16;
	int shared_size = (DATA_SIZE * DATA_SIZE) + count_a + count_b;
	mul<float> <<<dimGrid, dimBlock, shared_size * sizeof(float)>>>(d_data, DATA_SIZE,
			DB_SIZE);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	*myfile << "mul , " << duration << "\n";

	dim3 dimGrid2(1, GMM_SIZE, 1);
	// 128 threads allows an effective occupancy
	// optimization for a coalesced access to global memory
	dim3 dimBlock2(128, 1, 1);
	cudaDeviceSynchronize();
	cudaEventRecord(start, 0);
	p_kernel<float> <<<dimGrid2, dimBlock2>>>(d_data);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	*myfile << "p_kernel , " << duration << "\n";

	dim3 dimGrid3(2, 1, 1); // 2 = number of multiprocessors
	cudaDeviceSynchronize();
	cudaEventRecord(start, 0);
	pn_kernel<float> <<<dimGrid3, dimBlock2>>>(d_data, GMM_SIZE, DB_SIZE);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	*myfile << "pn_kernel , " << duration << "\n";

	cudaDeviceSynchronize();
	cudaEventRecord(stop_T, 0);
	cudaEventSynchronize(stop_T);
	cudaEventElapsedTime(&duration_T, start_T, stop_T);
	*timeTotal = (double) duration_T;
}
