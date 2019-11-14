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
//  Class ARMADILLO
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef	__WHMIN_ARMADILLO_H__
#define __WHMIN_ARMADILLO_H__
//#define N_REST_POSE
#define _FIXED
//#define DEBUG
#include "../lib/CUDA_PROJECTIVE_TET_MESH.h"

template <class TYPE>
class ARMADILLO : public CUDA_PROJECTIVE_TET_MESH<TYPE>
{
public:
	ARMADILLO()
	{
		FILE *f = fopen("setting.txt", "r");
		char filename[256];
		char output[256];
#ifdef SETTINGF
		fscanf(f, "%s", filename);
#else
		filename = "armadillo_10k.1";
#endif
#ifdef SETTINGF
		fscanf(f, "%s", output);
#else
		output = "default_name.txt"
#endif
		benchmark = fopen(output, "w");
		fscanf(f, "%d", &enable_benchmark);
		if (enable_benchmark)
		{
			printf("enable benchmark\n");
			if (!benchmark)
				printf("no benchmark file\n");
		}

		//Read_Original_File("sorted_armadillo");
		Read_Original_File(filename);
		//Read_Original_File("armadillo_100K");
		//Read_Original_File("longbar");
		float temp;
		fscanf(f, "%f", &temp);
		Scale(temp);
		Centralize();
		printf("N: %d, %d\n", number, tet_number);

		//Set fixed nodes
		fscanf(f, "%f", &temp);
		Rotate_X(temp);
#ifdef _FIXED
		//// for armadillo
		//for (int v = 0; v < number; v++)
		//	if(X[v*3+1]>0.465)		
		//		fixed[v] = 1;

		//for armadillo 2
		for (int v = 0; v < number; v++)
			if (fabsf(X[v * 3 + 1] + 0.01) < 2 * (X[v * 3 + 2] - 0.2))
				fixed[v] = 1;

		////for octopus
		//for (int v = 0; v < number; v++)
		//	if (X[v * 3 + 1] > 0.2)
		//		fixed[v] = 1;

		//for dragon
		//for (int v = 0; v < number; v++)
		//	if (X[v * 3 + 1] > 0.35)
		//		fixed[v] = 1;

		//for longbar
		//for (int v = 0; v < number; v++)
		//	if (X[v * 3 + 0]<-0.79 || X[v * 3 + 0]>0.79)
		//		fixed[v] = 1;

		//for bunny
		//for (int v = 0; v < number; v++)
		//	if (X[v * 3 + 1] < -0.25)
		//		fixed[v] = 1;


		//for test
		//fixed[0] = 1;
		//for (int v = 0; v < number; v++)
		//	if (X[v * 3 + 1] > 0.1)
		//		fixed[v] = 1;
#endif
		fscanf(f, "%f", &temp);
		Rotate_X(temp);

		

		//Translation(0.0, 0.65, 0.0);

		lambda = 0;
		mu = 250;

		//elasticity = 2.0*mu + lambda;/*18000000*/ //5000000
		elasticity = 500*1.0;
		control_mag = 1.0;		//500
		collision_mag = 1.0;
		damping = 1;

		fscanf(f, "%d %d %d %d", &pd_iters, &use_Hessian, &use_WHM, &enable_PD);
		fscanf(f, "%f", relaxation);

//		Read_Original_File("armadillo_10k.1");
//		Scale(0.008);
//		Centralize();
//		printf("N: %d, %d\n", number, tet_number);
//		Rotate_X(-0.2);
//#ifdef _FIXED
//				//for armadillo
//				for (int v = 0; v < number; v++)
//					//if(X[v*3+1]>-0.04 && X[v*3+1]<0)		
//					//	if(fabsf(X[v*3+1]+0.01)<1*(X[v*3+2]-0.1))
//					if (fabsf(X[v * 3 + 1] + 0.01) < 2 * (X[v * 3 + 2] - 0.2))
//						fixed[v] = 1;
//		
//				//for octopus
//				//for (int v = 0; v < number; v++)
//				//	if (X[v * 3 + 1] > 0.2)
//				//		fixed[v] = 1;
//#endif
//		Rotate_X(1.2);
//
//		elasticity = 500/*18000000*/; //5000000
//		control_mag = 1;		//500
//		damping = 1;


#ifdef SETTINGF
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
			else if (!strcmp(buf, "PCG"))
			{
				fscanf(f, "%d", &iter_num);
				settings.push_back(iterationSetting{ PCG,iter_num });
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

#else
#ifdef SETTING1
		layer = 0;
		handles_num = new int[layer + 1];
		handles_num[layer] = number;
		stored_as_LDU = new bool[layer + 1];
		stored_as_LDU[0] = false;
#else
#ifdef SETTING2
		layer = 1;
		handles_num = new int[layer + 1];
		handles_num[0] = 100;
		handles_num[1] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
#else
#ifdef SETTING3
		layer = 2;
		handles_num = new int[layer + 1];
		handles_num[0] = 10;
		handles_num[1] = 100;
		handles_num[2] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
		stored_as_dense[2] = false;
#else
#ifdef SETTING4
		layer = 3;
		handles_num = new int[layer + 1];
		handles_num[0] = 10;
		handles_num[1] = 100;
		handles_num[2] = 1000;
		handles_num[3] = number;
		stored_as_dense = new bool[layer + 1];
		stored_as_dense[0] = false;
		stored_as_dense[1] = false;
		stored_as_dense[2] = false;
		stored_as_dense[3] = false;
		stored_as_LDU = new bool[layer + 1];
		stored_as_LDU[0] = false;
		stored_as_LDU[1] = true;
		stored_as_LDU[2] = true;
		stored_as_LDU[3] = true;
#endif
#endif
#endif
#endif
#endif
	}

};


#endif
