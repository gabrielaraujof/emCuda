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

#include "gmm_types.h"

#ifndef EM_H_
#define EM_H_

void gmm(DeviceData d_data, ofstream *myfile, double *time,
		unsigned char dtsize, unsigned int dbsize);

#endif /* EM_H_ */
