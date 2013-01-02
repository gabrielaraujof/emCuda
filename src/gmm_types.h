/*
 * Copyright 2012. Gabriel Araujo
 * 
 * This file is part of EMCuda.
 *
 * EMCuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * EMCuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with EMCuda.  If not, see <http://www.gnu.org/licenses/>.
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
