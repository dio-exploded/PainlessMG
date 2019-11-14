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
		FILE *f = fopen("setting.txt", "r");
		int plane_size;
		char output[256];
#ifdef SETTINGF
		fscanf(f, "%d", &plane_size);
#else
		plane_size = 51;
#endif
#ifdef SETTINGF
		fscanf(f, "%s", output);
		benchmark = fopen(output, "w");
#endif
		fscanf(f, "%d", &enable_benchmark);

		if (enable_benchmark)
		{
			printf("enable benchmark\n");
			if (!benchmark)
				printf("no benchmark file\n");
		}

		Make_A_Plane(plane_size, plane_size, -0.5, 0, 0);
		Rotate_X(-3.14159/2);
		
		// Set two fixed corners
		memset(fixed, 0, sizeof(int)*number);
		fixed[  0]	= 1;
		fixed[plane_size - 1] = 1;

		// Set variables
		rho			= 0.9998;
		control_mag	= 100*1;
		spring_k	= 50*2;
		bending_k = 10 * 2;
		collision_mag = 1.0;
		//bending_k	= 0.00001;
		//air_damping	= 1.0/*0.99999*/;
		//lap_damping = 4;

#ifdef SETTINGF
		fscanf(f, "%d %d %d %d", &pd_iters, &use_Hessian, &use_WHM, &enable_PD);
		fscanf(f, "%d", &layer);
		handles_num = new int[layer + 1];
		for (int i = 0; i < layer; i++)
			fscanf(f, "%d", &handles_num[i]);
		handles_num[layer] = number;
		stored_as_dense = new int[layer + 1];
		for (int i = 0; i <= layer; i++)
			fscanf(f, "%d", &stored_as_dense[i]);
		stored_as_LDU = new int[layer + 1];
		for (int i = 0; i <= layer; i++)
			fscanf(f, "%d", &stored_as_LDU[i]);
		int iters;
		fscanf(f, "%d", &iters);
		char buf[256];
		int iter_num;
		for (int i = 0; i < iters; i++)
		{
			fscanf(f, "%s", buf);
			if (!strcmp(buf, "Jacobi"))
			{
				fscanf(f, "%d", &iter_num);
				settings.push_back(iterationSetting{ Jacobi,iter_num });
			}
			else if (!strcmp(buf, "GS"))
			{
				fscanf(f, "%d", &iter_num);
				settings.push_back(iterationSetting{ GaussSeidel,iter_num });
			}
			else if (!strcmp(buf, "CG"))
			{
				fscanf(f, "%d", &iter_num);
				settings.push_back(iterationSetting{ CG,iter_num });
			}
			else if (!strcmp(buf, "Direct"))
				settings.push_back(iterationSetting{ Direct,-1 });
			else if (!strcmp(buf, "DS"))
				settings.push_back(iterationSetting{ DownSample,-1 });
			else if (!strcmp(buf, "US"))
				settings.push_back(iterationSetting{ UpSample,-1 });
		}
		fscanf(f, "%f", &rho);
		fscanf(f, "%d", &objects_num);
		if (objects_num)
		{
			objects = (object*)malloc(sizeof(object)*objects_num);
			for (int i = 0; i < objects_num; i++)
			{
				fscanf(f, "%s", buf);
				if (!strcmp(buf, "Plane"))
				{
					objects[i].type = objectType::Plane;
					fscanf(f, "%f %f %f %f", &objects[i].p_nx, &objects[i].p_ny, &objects[i].p_nz, &objects[i].p_c);
				}
				else if (!strcmp(buf, "Sphere"))
				{
					objects[i].type = objectType::Sphere;
					fscanf(f, "%f %f %f %f", &objects[i].s_cx, &objects[i].s_cy, &objects[i].s_cz, &objects[i].s_r);
				}
				else if (!strcmp(buf, "Cylinder"))
				{
					objects[i].type = objectType::Cylinder;
					fscanf(f, "%f %f %f", &objects[i].s_cx, &objects[i].s_cy, &objects[i].s_r);
				}
			}
		}
#endif
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
		handles_num[0] = 10;
		handles_num[1] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
		stored_as_LDU = new bool[layer + 1];
		stored_as_LDU[0] = false;
		stored_as_LDU[1] = true;
#endif
#ifdef SETTING3
		layer = 2;
		handles_num = new int[layer + 1];
		handles_num[0] = 25;
		handles_num[1] = 250;
		handles_num[2] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
		stored_as_dense[2] = false;
		stored_as_LDU = new bool[layer + 1];
		stored_as_LDU[0] = true;
		stored_as_LDU[1] = true;
		stored_as_LDU[2] = true;
#endif
#ifdef SETTING4
		layer = 4;
		handles_num = new int[layer + 1];
		handles_num[0] = 25;
		handles_num[1] = 50;
		handles_num[2] = 100;
		handles_num[3] = 200;
		handles_num[4] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
		stored_as_dense[2] = false;
		stored_as_dense[3] = false;
		stored_as_dense[4] = false;
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