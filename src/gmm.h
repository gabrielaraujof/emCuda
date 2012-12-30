/*
 * All rights reserved.
 *
 * Please refer to the author and user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * gmm.h
 *
 *  Created on: Dec 8, 2012
 *      Author: gabriel ( gabrielfa@dcomp.ufs.br )
 */

#include "gmm_types.h"

#ifndef EM_H_
#define EM_H_

void gmm(DeviceData d_data, ofstream *myfile, double *time,
		unsigned char dtsize, unsigned int dbsize);

#endif /* EM_H_ */
