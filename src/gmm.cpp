/*
 * gmm.cpp
 *
 *  Created on: Dec 9, 2012
 *      Author: gabriel
 */

// C libraries
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>

using namespace std;

// CUDA Runtime
#include <cuda_runtime.h>

// includes, project
#include "gmm.h"
#include "gmm_cpu.h"
#include "gmm_types.h"

const char* OUTPUT = "outputTimes.txt";
const char* FILENAME = "base_0.txt";
#define PI 3.14159265

bool run(int argc, char **argv, float *host_samples);
void readDatabase(float *data);

int main(int argc, char **argv) {

	printf("Estimating GMM with %d gaussians\n\n", GMM_SIZE);

	bool bResult = false;
	float *input_samples = new float[DB_SIZE * DATA_SIZE];
	readDatabase(input_samples);
	cudaDeviceReset();

	run(argc, argv, input_samples);
}

void readDatabase(float *data) {
	FILE *matriz;

	float valor;
	int i, j;

	if ((matriz = fopen(FILENAME, "r")) == NULL) {
		fprintf(stderr,
				"Nao foi possivel abrir arquivo Train_Arabic_Digit_0.txt\n");
		exit(1);
	}
	/* carregando os elementos do arquivo                 */
	for (i = 0; i < DB_SIZE; i++) {
		for (j = 0; j < DATA_SIZE; j++) {
			int r = fscanf(matriz, "%f", &valor);
			if (r > 0)
				data[i * DATA_SIZE + j] = valor;
		}
	}

}

void initializeData(HostData data) {
	// Initialize the convariances, means, determinants and weights
	for (int i = 0; i < GMM_SIZE; i++) {
		data.determinants[i] = 1.0e26;
		data.weights[i] = 1 / (float) GMM_SIZE;
		for (int j = 0; j < DATA_SIZE; j++) {
			int index = (DB_SIZE / GMM_SIZE) * i;
			data.means[i * DATA_SIZE + j] = data.samples[index * DATA_SIZE + j];
			for (int k = 0; k < DATA_SIZE; k++)
				if (j == k) {
					data.covariance_matrices[i * (DATA_SIZE * DATA_SIZE)
							+ j * DATA_SIZE + k] = 100;
					data.inv_covariance_matrices[i * (DATA_SIZE * DATA_SIZE)
							+ j * DATA_SIZE + k] = 0.01f;
				} else {
					data.covariance_matrices[i * (DATA_SIZE * DATA_SIZE)
							+ j * DATA_SIZE + k] = 0;
					data.inv_covariance_matrices[i * (DATA_SIZE * DATA_SIZE)
							+ j * DATA_SIZE + k] = 0;
				}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
		int maxThreads, int &blocks, int &threads) {

}

bool run(int argc, char **argv, float *host_samples) {

	// create random input data on CPU
	HostData h_data;
	h_data.samples = host_samples;

	h_data.likelihood_matrix = (float*) malloc(
			GMM_SIZE * DB_SIZE * sizeof(float));
	h_data.covariance_matrices = (float*) malloc(
			GMM_SIZE * DATA_SIZE * DATA_SIZE * sizeof(float));
	h_data.inv_covariance_matrices = (float*) malloc(
			GMM_SIZE * DATA_SIZE * DATA_SIZE * sizeof(float));
	h_data.means = (float*) malloc(GMM_SIZE * DATA_SIZE * sizeof(float));
	h_data.determinants = (float*) malloc(GMM_SIZE * sizeof(float));
	h_data.weights = (float*) malloc(GMM_SIZE * sizeof(float));
	h_data.factor_pi = pow(2 * PI, DATA_SIZE / 2.0);

	initializeData(h_data);

	/*int numBlocks = 0;
	 int numThreads = 0;
	 //getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
	 //		numThreads);
	 printf("%d fblocks\n\n", numBlocks);*/

	// allocate mem for the result on host side
	float *h_covariance_matrices = (float*) malloc(
			GMM_SIZE * DATA_SIZE * DATA_SIZE * sizeof(float));

	// allocate device memory and data
	DeviceData d_data;
	d_data.samples = NULL;
	d_data.likelihood_matrix = NULL;
	d_data.covariance_matrices = NULL;
	d_data.inv_covariance_matrices = NULL;
	d_data.means = NULL;
	d_data.determinants = NULL;
	d_data.weights = NULL;
	d_data.marginals = NULL;

	// Allocate memory on the GPU

	cudaMalloc((void**) &d_data.samples, DB_SIZE * DATA_SIZE * sizeof(float));

	cudaMalloc((void**) &d_data.likelihood_matrix,
			GMM_SIZE * DB_SIZE * sizeof(float));

	cudaMalloc((void**) &d_data.covariance_matrices,
			GMM_SIZE * DATA_SIZE * DATA_SIZE * sizeof(float));

	cudaMalloc((void**) &d_data.inv_covariance_matrices,
			GMM_SIZE * DATA_SIZE * DATA_SIZE * sizeof(float));

	cudaMalloc((void**) &d_data.means, GMM_SIZE * DATA_SIZE * sizeof(float));

	cudaMalloc((void**) &d_data.determinants, GMM_SIZE * sizeof(float));
	cudaMalloc((void**) &d_data.factor_pi, sizeof(float));

	cudaMalloc((void**) &d_data.weights, GMM_SIZE * sizeof(float));

	cudaMalloc((void**) &d_data.marginals, GMM_SIZE * sizeof(float));

	// copy data directly to device memory

	cudaMemcpy(d_data.samples, h_data.samples,
			DB_SIZE * DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_data.covariance_matrices, h_data.covariance_matrices,
			GMM_SIZE * DATA_SIZE * DATA_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);

	cudaMemcpy(d_data.inv_covariance_matrices, h_data.inv_covariance_matrices,
			GMM_SIZE * DATA_SIZE * DATA_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);

	cudaMemcpy(d_data.means, h_data.means, GMM_SIZE * DATA_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);

	cudaMemcpy(d_data.determinants, h_data.determinants,
			GMM_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_data.weights, h_data.weights, GMM_SIZE * sizeof(float),
			cudaMemcpyHostToDevice);
	d_data.factor_pi = pow(2 * PI, DATA_SIZE / 2.0);

	// Declaring StopWatch
	double timerGPU, timerCPU;

	// Output file for the times execution
	ofstream myfile;
	// Opening output file
	myfile.open(OUTPUT);

	// call cuda kernel
	gmm(d_data, &myfile, &timerGPU, DATA_SIZE, DB_SIZE);

	// Gets the elapsed time in the kernel execution (in milliseconds)
	printf("CUDA GMM Estimation, Time = %g milliseconds\n", timerGPU);

	gmmCPU(h_data, &timerCPU);

	// Gets the elapsed time in the kernel execution (in milliseconds)
	printf("CPU GMM Estimation, Time Clib = %g milliseconds\n", timerCPU);

	myfile.close();

	// Copying back the results from GPU
	float *host_likelihood = (float*) malloc(
			GMM_SIZE * DB_SIZE * sizeof(float));
	cudaMemcpy(host_likelihood, d_data.likelihood_matrix,
			GMM_SIZE * DB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	/** Begin Print result **/

	printf("\nGPU result\n");
	for (int i = 0; i < GMM_SIZE; i++) {
		for (int j = 0; j < 10; j++) {
			printf("%G, ", host_likelihood[j + i * DB_SIZE]);
		}
		printf("\n");
	}
	printf("\nCPU result\n");
	for (int i = 0; i < GMM_SIZE; i++) {
		for (int j = 0; j < 10; j++) {
			printf("%G, ", h_data.likelihood_matrix[j + i * DB_SIZE]);
		}
		printf("\n");
	}
	/** End Printer result **/

	// cleanup
	free(host_samples);
	free(host_likelihood);
	free(h_data.samples);
	free(h_data.likelihood_matrix);
	free(h_data.covariance_matrices);
	free(h_data.inv_covariance_matrices);
	free(h_data.means);
	free(h_data.determinants);
	free(h_data.weights);

	cudaFree(d_data.samples);
	cudaFree(d_data.likelihood_matrix);
	cudaFree(d_data.covariance_matrices);
	cudaFree(d_data.inv_covariance_matrices);
	cudaFree(d_data.means);
	cudaFree(d_data.determinants);
	cudaFree(d_data.weights);
	cudaFree(d_data.marginals);

	float threshold = 1e-8;
	float diff = 0;
	for (int i = 0; i < GMM_SIZE; i++) {
		for (int j = 0; j < DB_SIZE; j++) {
			diff = fabs(
					host_likelihood[j + i * DB_SIZE]
							- h_data.likelihood_matrix[j + i * DB_SIZE]);
			if (diff > threshold)
				return false;
		}
	}

	return true;
}
