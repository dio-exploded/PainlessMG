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
//  Class CLOTHING
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_CLOTHING_H__
#define __WHMIN_CLOTHING_H__
#include "../lib/CUDA_PROJECTIVE_MESH.h"

template <class TYPE>
class CLOTHING : public CUDA_PROJECTIVE_MESH<TYPE>
{
public:

	CLOTHING()
	{	
		Make_A_Plane(101,101, -0.5, 0, 0);
		Rotate_X(-3.14159/2);
		
		// Set two fixed corners
		memset(fixed, 0, sizeof(int)*number);
		fixed[  0]	= 1;
		fixed[100]	= 1;

		// Set variables
		rho			= 0.9998;
		control_mag	= 1000;
		spring_k	= 100;
		//bending_k	= 0.00001;
		//air_damping	= 1.0/*0.99999*/;
		//lap_damping = 4;
#ifdef SETTING1
		layer = 0;
		handles_num = new int[layer+1];
		handles_num[0] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
#endif
#ifdef SETTING2
		layer = 1;
		handles_num = new int[layer + 1];
		handles_num[0] = 250;
		handles_num[1] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
#endif
#ifdef SETTING3
		layer = 2;
		handles_num = new int[layer + 1];
		handles_num[0] = 25;
		handles_num[1] = 1000;
		handles_num[2] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
		stored_as_dense[2] = false;
#endif
#ifdef SETTING4
		layer = 3;
		handles_num = new int[layer + 1];
		handles_num[0] = 25;
		handles_num[1] = 250;
		handles_num[2] = 1500;
		handles_num[3] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
		stored_as_dense[2] = false;
		stored_as_dense[3] = false;
#endif
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Utility functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Clear_Velocity()
    {
        memset(V, 0, sizeof(TYPE)*number*3);
    }

///////////////////////////////////////////////////////////////////////////////////////////
//  Render function
///////////////////////////////////////////////////////////////////////////////////////////
	void Render(int render_mode)	
	{
		float diffuse_color[3]={0.6, 0.6, 1.0};
		glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_color);
		CUDA_PROJECTIVE_MESH<TYPE>::Render(render_mode);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  IO functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Write(std::fstream &output)
	{
		CUDA_PROJECTIVE_MESH<TYPE>::Write(output);
	}

	bool Write_File(const char *file_name)
	{
		std::fstream output; 
		output.open(file_name,std::ios::out|std::ios::binary);
		if(!output.is_open())	{printf("Error, file not open.\n"); return false;}
		Write(output);
		output.close();
		return true;
	}
	
	void Read(std::fstream &input)
	{
		CUDA_PROJECTIVE_MESH<TYPE>::Read(input);
	}

	bool Read_File(const char *file_name)
	{
		std::fstream input; 
		input.open(file_name,std::ios::in|std::ios::binary);
		if(!input.is_open())	{printf("Error, file not open.\n");	return false;}
		Read(input);
		input.close();
		return true;
	}

};




#endif