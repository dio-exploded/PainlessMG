///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2015, Huamin Wang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////
//  Class CLOTHING
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_CLOTHING_H__
#define __WHMIN_CLOTHING_H__
#include "../lib/PROJECTIVE_MESH.h"


template <class TYPE>
class CLOTHING: public PROJECTIVE_MESH<TYPE>
{
public:

	CLOTHING()
	{	
		int n=101;
		Make_A_Plane(n, n, -0.5, 0.5, 0);
		Rotate_X(-3.14159/2);

		// Set up fixed nodes
		memset(fixed, 0, sizeof(int)*number);
		fixed[  0]=100000;	//1000000000
		fixed[n-1]=100000;
		
		// Adjust variables
		rho			= 0.9999;
		spring_k	= 3000000; //3000000
		bending_k	= 10000; //10000
		air_damping	= 0.9999;

		printf("Vertices: %d; triangles: %d\n", number, t_number);
	}	
};


#endif
