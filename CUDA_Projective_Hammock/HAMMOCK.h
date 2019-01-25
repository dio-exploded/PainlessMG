///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2014, Huamin Wang
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
//  Class HAMMOCK
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef	__WHMIN_HAMMOCK_H__
#define __WHMIN_HAMMOCK_H__
#include "../lib/CUDA_PROJECTIVE_TET_MESH.h"


template <class TYPE>
class HAMMOCK: public CUDA_PROJECTIVE_TET_MESH<TYPE> 
{
public:
	HAMMOCK()
	{
		Read_Original_File("hammock.1");
		Scale(0.5, 0.3, 0.4);
		Centralize();
		printf("N: %d, %d\n", number, tet_number);

		// Set fixed nodes
		for(int v=0; v<number; v++)
			if(X[v*3+1]>0.42)
				fixed[v]=1000000;	//10000000

		// Set variables
		control_mag	= 600;
		elasticity	= 30000000; //5000000
		rho			= 0.9996;
	}
	
};


#endif