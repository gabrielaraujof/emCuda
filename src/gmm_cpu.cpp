/*
 * gmm_cpu.cpp
 *
 *  Created on: Dec 10, 2012
 *      Author: gabriel
 */

/*
 GMM (Gaussian Mixture Model) Estimation Kernels
 CPU version
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>

// includes, project
#include "gmm_types.h"

/**
 *  Subtract the vectors X and U, for the calculation of g(x) function
 */
float* sub_vectors(float *x, float *u) {
	float *result = new float[DATA_SIZE];
	for (short int i = 0; i < DATA_SIZE; i++)
		result[i] = x[i] - u[i];
	return result;
}

/**
 *  Multiply the vector X for the covariance matrix S^-1 and for the X',
 *  for the calculation of g(x) function
 */
float mul_vector_matrix(float *x, float *inv_cov) {
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
void gxFunction(float *data, float *means, float *inv_cov, float *likelihood) {

	for (int i = 0; i < GMM_SIZE; i++) {
		for (int j = 0; j < DB_SIZE; j++) {
			float* vector_sub = sub_vectors((data + j * DATA_SIZE),
					(means + i * DATA_SIZE));
			likelihood[j + i * DB_SIZE] = mul_vector_matrix(vector_sub,
					(inv_cov + i * (DATA_SIZE * DATA_SIZE)));
			free(vector_sub);
		}
	}
}

/**
 *               <<< FIRST PHASE OF ALGORITHM >>>
 * Computes the likelihood of all samples to each gaussian from mixture.
 * Each thread works with a sample at time, but is able to compute for
 * any numbers of threads, in case that.
 */
void p_kernel(HostData h) {

	// Calculate likelihood for sample for each each gaussian
	for (int i = 0; i < GMM_SIZE; i++) {
		for (int j = 0; j < DB_SIZE; j++) {
			float factor_b = -0.5f * h.likelihood_matrix[i * DB_SIZE + j];
			float factor_a = 1 / (h.factor_pi * sqrt(h.determinants[i]));
			float result = (factor_a * exp(factor_b)) * h.weights[i];

			h.likelihood_matrix[i * DB_SIZE + j] = result;
		}
	}
}

/**
 *              <<< SECOND PHASE OF ALGORITHM >>>
 * Normalizes all the likelihoods values for each sample. Each block thread
 * has K threads (K = number of gaussianas), that works collaboratively
 * to normalize all the likelihoods from a sample.
 */
void pn_kernel(HostData h){

    for (int i=0; i < DB_SIZE; i++){
        float sum = 0;
        for (int j=0; j < GMM_SIZE; j++)
            sum += h.likelihood_matrix[j * DB_SIZE + i];
        for (int j=0; j < GMM_SIZE; j++)
            h.likelihood_matrix[j * DB_SIZE + i] /= sum;
    }
}

void gmmCPU(HostData h_data, double *timer2) {

	clock_t start_d = clock();

	gxFunction(h_data.samples, h_data.means, h_data.inv_covariance_matrices,
			h_data.likelihood_matrix);
	p_kernel(h_data);
	pn_kernel(h_data);

	clock_t end_d = clock();

	*timer2 = ((end_d - start_d) / (double) CLOCKS_PER_SEC) * 1.0e3;
}
