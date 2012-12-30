/*
 * gmm_types.h
 *
 *  Created on: Dec 10, 2012
 *      Author: gabriel
 */

#ifndef GMM_TYPES_H_
#define GMM_TYPES_H_

struct DeviceData {
	float *samples;
	float *likelihood_matrix;
	float *covariance_matrices;
	float *inv_covariance_matrices;
	float *means;
	float *determinants;
	float *weights;
	float *marginals;
	float factor_pi;
};

struct HostData {
	float *samples;
	float *likelihood_matrix;
	float *covariance_matrices;
	float *inv_covariance_matrices;
	float *means;
	float *determinants;
	float *weights;
	float factor_pi;
};

const int DATA_SIZE = 13;
const int DB_SIZE = 23344;
const int GMM_SIZE = 2;

#endif /* GMM_TYPES_H_ */
