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
//  Class CUDA_PROJECTIVE_TET_MESH
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef	__WHMIN_CUDA_PROJECTIVE_TET_MESH_H__
#define __WHMIN_CUDA_PROJECTIVE_TET_MESH_H__
#include "TET_MESH.h"
#include "TIMER.h"
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>
#include<algorithm>
#include<map>
#include<math.h>
#include<queue>
#include<vector>
#include "helper_cuda.h"
#include "helper_cusolver.h"


#define GRAVITY			-9.8
#define RADIUS_SQUARED	0.002//0.01

//#define NO_ORDER
#define MATRIX_OUTPUT

//#define PRECOMPUTE_DENSE_UPDATE
#define DEFAULT_UPDATE
//#define NO_UPDATE
//#define VECTOR_UPDATE

const float zero = 0.0;
const float one = 1.0;
const float minus_one = -1.0;
const float inf = INFINITY;
const float tol = 1e-5;
const int threadsPerBlock = 64;
const int tet_threadsPerBlock = 64;

///////////////////////////////////////////////////////////////////////////////////////////
//  math kernels
///////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_Matrix_Product_3(const float *A, const float *B, float *R)				//R=A*B
{
	R[0]=A[0]*B[0]+A[1]*B[3]+A[2]*B[6];
	R[1]=A[0]*B[1]+A[1]*B[4]+A[2]*B[7];
	R[2]=A[0]*B[2]+A[1]*B[5]+A[2]*B[8];
	R[3]=A[3]*B[0]+A[4]*B[3]+A[5]*B[6];
	R[4]=A[3]*B[1]+A[4]*B[4]+A[5]*B[7];
	R[5]=A[3]*B[2]+A[4]*B[5]+A[5]*B[8];
	R[6]=A[6]*B[0]+A[7]*B[3]+A[8]*B[6];
	R[7]=A[6]*B[1]+A[7]*B[4]+A[8]*B[7];
	R[8]=A[6]*B[2]+A[7]*B[5]+A[8]*B[8];
}

__device__ void dev_Matrix_Substract_3(float *A, float *B, float *R)						//R=A-B
{
	for(int i=0; i<9; i++)	R[i]=A[i]-B[i];
}

__device__ void dev_Matrix_Product(float *A, float *B, float *R, int nx, int ny, int nz)	//R=A*B
{
	memset(R, 0, sizeof(float)*nx*nz);
	for(int i=0; i<nx; i++)
	for(int j=0; j<nz; j++)
	for(int k=0; k<ny; k++)
		R[i*nz+j]+=A[i*ny+k]*B[k*nz+j];
}

__device__ void Get_Rotation(float F[3][3], float R[3][3])
{
    float C[3][3];
    memset(&C[0][0], 0, sizeof(float)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        C[i][j]+=F[k][i]*F[k][j];
    
    float C2[3][3];
    memset(&C2[0][0], 0, sizeof(float)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        C2[i][j]+=C[i][k]*C[j][k];
    
    float det    =   F[0][0]*F[1][1]*F[2][2]+
                    F[0][1]*F[1][2]*F[2][0]+
                    F[1][0]*F[2][1]*F[0][2]-
                    F[0][2]*F[1][1]*F[2][0]-
                    F[0][1]*F[1][0]*F[2][2]-
                    F[0][0]*F[1][2]*F[2][1];
    
    float I_c    =   C[0][0]+C[1][1]+C[2][2];
    float I_c2   =   I_c*I_c;
    float II_c   =   0.5*(I_c2-C2[0][0]-C2[1][1]-C2[2][2]);
    float III_c  =   det*det;
    float k      =   I_c2-3*II_c;
    
    float inv_U[3][3];
    if(k<1e-10f)
    {
        float inv_lambda=1/sqrt(I_c/3);
        memset(inv_U, 0, sizeof(float)*9);
        inv_U[0][0]=inv_lambda;
        inv_U[1][1]=inv_lambda;
        inv_U[2][2]=inv_lambda;
    }
    else
    {
        float l = I_c*(I_c*I_c-4.5*II_c)+13.5*III_c;
        float k_root = sqrt(k);
        float value=l/(k*k_root);
        if(value<-1.0) value=-1.0;
        if(value> 1.0) value= 1.0;
        float phi = acos(value);
        float lambda2=(I_c+2*k_root*cos(phi/3))/3.0;
        float lambda=sqrt(lambda2);
        
        float III_u = sqrt(III_c);
        if(det<0)   III_u=-III_u;
        float I_u = lambda + sqrt(-lambda2 + I_c + 2*III_u/lambda);
        float II_u=(I_u*I_u-I_c)*0.5;
        
        float U[3][3];
        float inv_rate, factor;
        
        inv_rate=1/(I_u*II_u-III_u);
        factor=I_u*III_u*inv_rate;
        
        memset(U, 0, sizeof(float)*9);
        U[0][0]=factor;
        U[1][1]=factor;
        U[2][2]=factor;
        
        factor=(I_u*I_u-II_u)*inv_rate;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            U[i][j]+=factor*C[i][j]-inv_rate*C2[i][j];
        
        inv_rate=1/III_u;
        factor=II_u*inv_rate;
        memset(inv_U, 0, sizeof(float)*9);
        inv_U[0][0]=factor;
        inv_U[1][1]=factor;
        inv_U[2][2]=factor;
        
        factor=-I_u*inv_rate;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            inv_U[i][j]+=factor*U[i][j]+inv_rate*C[i][j];
    }
    
    memset(&R[0][0], 0, sizeof(float)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        R[i][j]+=F[i][k]*inv_U[k][j];    
}

__device__ void solve4x4(const float *A,const float *dA, const float *b,float *x)
{
	float r[4];
	float p[4];
	float Ap[4];
	float alpha, beta;
	float old_r_norm, r_norm;
	float dot;

	for (int i = 0; i < 4; i++)
		r[i] = b[i];

	r_norm = 0;
	for (int i = 0; i < 4; i++)
		r_norm += r[i] * r[i];

	if (r_norm < EPSILON) return;

	for (int i = 0; i < 4; i++)
		p[i] = r[i];

	for (int l = 0; l < 4; l++)
	{
		for (int i = 0; i < 4; i++)
			Ap[i] = 0;
		for (int k = 0; k < 4; k++)
			for (int i = 0; i < 4; i++)
				Ap[i] += (A[k * 4 + i] + dA[k * 4 + i]) * p[k];
		dot = 0;
		for (int i = 0; i < 4; i++)
			dot += Ap[i] * p[i];
		alpha = r_norm / dot;
		for (int i = 0; i < 4; i++)
		{
			x[i] += alpha * p[i];
			r[i] -= alpha * Ap[i];
		}
		old_r_norm = r_norm;
		r_norm = 0;
		for (int i = 0; i < 4; i++)
			r_norm += r[i] * r[i];

		if (r_norm < EPSILON) return;

		beta = r_norm / old_r_norm;
		for (int i = 0; i < 4; i++)
			p[i] = r[i] + beta * p[i];
	}

}

__device__ void collision_detection(const float *X, float *fixed_X,int *fixed)
{
	*fixed = 0;
	//if (X[1] < -0.5)
	//{
	//	*fixed = 1;
	//	fixed_X[0] = X[0];
	//	fixed_X[1] = -0.5;
	//	fixed_X[2] = X[2];
	//}
}


///////////////////////////////////////////////////////////////////////////////////////////
//  Control kernel
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Control_Kernel(float* X, int *more_fixed, const int number, const int select_v)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=number)	return;

	more_fixed[i]=0;
	if(select_v!=-1)
	{
		float dist2=0;
		dist2+=(X[i*3+0]-X[select_v*3+0])*(X[i*3+0]-X[select_v*3+0]);
		dist2+=(X[i*3+1]-X[select_v*3+1])*(X[i*3+1]-X[select_v*3+1]);
		dist2+=(X[i*3+2]-X[select_v*3+2])*(X[i*3+2]-X[select_v*3+2]);
		if(dist2<RADIUS_SQUARED)	more_fixed[i]=1;		
	}
}

//__global__ void Update_Flag_Kernel(const int *lo, int *hi, const int *r, const int number)
//{
//	int i = blockDim.x * blockIdx.x + threadIdx.x;
//	if (i >= number)	return;
//	if (lo[i]==1)
//		hi[r[i]] = 1;
//}

///////////////////////////////////////////////////////////////////////////////////////////
//  Basic update kernel
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Fixed_Update_Kernel(float* X, int *collision_fixed, const int *fixed, const int *more_fixed, float *fixed_X, const int number, const float dir_x, const float dir_y, const float dir_z)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;

	collision_detection(X + i * 3, fixed_X + i * 3, collision_fixed + i);

	if (fixed[i] == 0 && collision_fixed[i] == 0 && more_fixed[i] != 0)
	{
		fixed_X[i * 3 + 0] = X[i * 3 + 0] + dir_x;
		fixed_X[i * 3 + 1] = X[i * 3 + 1] + dir_y;
		fixed_X[i * 3 + 2] = X[i * 3 + 2] + dir_z;
	}
}

__global__ void Basic_Update_Kernel(float* X, float* V, const float damping, const float t, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;

	//Apply damping
	V[i * 3 + 0] *= damping;
	V[i * 3 + 1] *= damping;
	V[i * 3 + 2] *= damping;
	//Apply gravity
	V[i * 3 + 1] += GRAVITY * t;
	//Position update
	X[i * 3 + 0] += V[i * 3 + 0] * t;
	X[i * 3 + 1] += V[i * 3 + 1] * t;
	X[i * 3 + 2] += V[i * 3 + 2] * t;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Tet Constraint Kernel
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Tet_Constraint_Kernel(const float* X, const int* Tet, const float* inv_Dm, const float* Vol, float* Tet_Temp, const float elasticity, const int tet_number, const int l)
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	if(t>=tet_number)	return;
	
	int p0=Tet[t*4+0]*3;
	int p1=Tet[t*4+1]*3;
	int p2=Tet[t*4+2]*3;
	int p3=Tet[t*4+3]*3;

	const float* idm=&inv_Dm[t*9];

	float Ds[9];
	Ds[0]=X[p1+0]-X[p0+0];
	Ds[3]=X[p1+1]-X[p0+1];
	Ds[6]=X[p1+2]-X[p0+2];
	Ds[1]=X[p2+0]-X[p0+0];
	Ds[4]=X[p2+1]-X[p0+1];
	Ds[7]=X[p2+2]-X[p0+2];
	Ds[2]=X[p3+0]-X[p0+0];
	Ds[5]=X[p3+1]-X[p0+1];
	Ds[8]=X[p3+2]-X[p0+2];

	float F[9], R[9], B[3], C[9];
	float new_R[9];
	dev_Matrix_Product_3(Ds, idm, F);
	
	Get_Rotation((float (*)[3])F, (float (*)[3])new_R);
	
	float half_matrix[3][4], result_matrix[3][4];
	half_matrix[0][0]=-idm[0]-idm[3]-idm[6];
	half_matrix[0][1]= idm[0];
	half_matrix[0][2]= idm[3];
	half_matrix[0][3]= idm[6];
	half_matrix[1][0]=-idm[1]-idm[4]-idm[7];
	half_matrix[1][1]= idm[1];
	half_matrix[1][2]= idm[4];
	half_matrix[1][3]= idm[7];
	half_matrix[2][0]=-idm[2]-idm[5]-idm[8];
	half_matrix[2][1]= idm[2];
	half_matrix[2][2]= idm[5];
	half_matrix[2][3]= idm[8];

	dev_Matrix_Substract_3(new_R, F, new_R);
	dev_Matrix_Product(new_R, &half_matrix[0][0], &result_matrix[0][0], 3, 3, 4);
			
	float rate=Vol[t]*elasticity;
	Tet_Temp[t*12+ 0]=result_matrix[0][0]*rate;
	Tet_Temp[t*12+ 1]=result_matrix[1][0]*rate;
	Tet_Temp[t*12+ 2]=result_matrix[2][0]*rate;
	Tet_Temp[t*12+ 3]=result_matrix[0][1]*rate;
	Tet_Temp[t*12+ 4]=result_matrix[1][1]*rate;
	Tet_Temp[t*12+ 5]=result_matrix[2][1]*rate;
	Tet_Temp[t*12+ 6]=result_matrix[0][2]*rate;
	Tet_Temp[t*12+ 7]=result_matrix[1][2]*rate;
	Tet_Temp[t*12+ 8]=result_matrix[2][2]*rate;
	Tet_Temp[t*12+ 9]=result_matrix[0][3]*rate;
	Tet_Temp[t*12+10]=result_matrix[1][3]*rate;
	Tet_Temp[t*12+11]=result_matrix[2][3]*rate;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint Kernel 0
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Diag_Update_Kernel(float* MF_Diag, const int* more_fixed,const int *collision_fixed, const int *vertex2index, const float control_mag,\
	const float collision_mag, const int *where_to_update, const float *precomputed_diag,float *UtAUs_diag, const int *handles_num, const int number,const int layer)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
	int i = vertex2index[v];
	if (more_fixed[v] || collision_fixed[v])
	{
		float mag = collision_fixed[v] ? collision_mag : control_mag;
		MF_Diag[i] = mag;
		int base = 0;
		for (int l = 0; l < layer; l++)
		{
			int t = where_to_update[layer*v + l];
			for (int dj = 0; dj < 4; dj++)
				for (int di = 0; di < 4; di++)
					atomicAdd(&UtAUs_diag[base + t * 16 + dj * 4 + di], mag*precomputed_diag[v * 16 + dj * 4 + di]);
			base += handles_num[l] * 16;
		}
	}
	
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint Kernel 1
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Constraint_1_Kernel(const float* X, const float* init_B, const float* VC, float* next_X, const float* Tet_Temp, const float* MD, const int* VTT, const int* vtt_num, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=number)	return;

	double b[3];
	b[0]=init_B[i*3+0]+MD[i]*X[i*3+0];
	b[1]=init_B[i*3+1]+MD[i]*X[i*3+1];
	b[2]=init_B[i*3+2]+MD[i]*X[i*3+2];
			
	for(int index=vtt_num[i]; index<vtt_num[i+1]; index++)
	{
		b[0]+=Tet_Temp[VTT[index]*3+0];
		b[1]+=Tet_Temp[VTT[index]*3+1];
		b[2]+=Tet_Temp[VTT[index]*3+2];
	}

	next_X[i*3+0]=b[0]/(VC[i]+MD[i]);
	next_X[i*3+1]=b[1]/(VC[i]+MD[i]);
	next_X[i*3+2]=b[2]/(VC[i]+MD[i]);
}

__global__ void Energy_Gradient_Kernel(float *G, const float* Tet_Temp, const int* VTT, const int* vtt_num, \
	const int *fixed,const int *more_fixed, const int *collision_fixed, const float *fixed_X,const float *X,\
	const int *vertex2index, const float control_mag,const float collision_mag, const int number)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
	//G[i] = 0;
	//G[number + i] = 0;
	//G[number * 2 + i] = 0;
	int i = vertex2index[v];
	for (int index = vtt_num[v]; index < vtt_num[v + 1]; index++)
	{
		G[i] += Tet_Temp[VTT[index] * 3 + 0];
		G[number + i] += Tet_Temp[VTT[index] * 3 + 1];
		G[number * 2 + i] += Tet_Temp[VTT[index] * 3 + 2];
	}
	if (fixed[v] || more_fixed[v] || collision_fixed[v])
	{
		float mag = collision_fixed[v] ? collision_mag : control_mag;
		G[i] += mag * (fixed_X[3 * v + 0] - X[3 * v + 0]);
		G[number + i] += mag * (fixed_X[3 * v + 1] - X[3 * v + 1]);
		G[number * 2 + i] += mag * (fixed_X[3 * v + 2] - X[3 * v + 2]);
	}
}

__global__ void Inertia_Gradient_Kernel(float *G, const float *inertia, const float *X, const int *vertex2index, const float *M, const float inv_t, int number)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
	float c = M[v] * inv_t*inv_t;
	int i = vertex2index[v];
	G[i] += c * (inertia[3 * i + 0] - X[3 * i + 0]);
	G[number + i] += c * (inertia[3 * i + 1] - X[3 * i + 1]);
	G[2 * number + i] += c * (inertia[3 * i + 2] - X[3 * i + 2]);
}

__global__ void MF_Diag_Update_Kernel(float *Y,const float *X, const float *Diag, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	Y[i] = Diag[i] * X[i];
}

__global__ void UtAUs_Diag_Update_Kernel(float *Y, const float *Diag, const float *X, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	Y[i * 4 + 0] = 0;
	Y[i * 4 + 1] = 0;
	Y[i * 4 + 2] = 0;
	Y[i * 4 + 3] = 0;
	for (int di = 0; di < 4; di++)
		for (int dj = 0; dj < 4; dj++)
			Y[i * 4 + di] += Diag[16 * i + dj * 4 + di] * X[i * 4 + dj];
}

__global__ void Colored_GS_MF_Kernel(float *X, const float *MF_Diag,const float *MF_Diag_addition,const float *b, const int base, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int t = base + i;
	X[t] = b[t] / (MF_Diag[t] + MF_Diag_addition[t]);
}

__global__ void Colored_GS_UtAU_Kernel(float *X, const float *UtAU_Diag,const float *UtAU_Diag_addition, const float *b, const int base, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int t = base + i;
	solve4x4(UtAU_Diag + 16 * t, UtAU_Diag_addition + 16 * t, b + 4 * t, X + 4 * t);
}


__global__ void Update_DeltaX_Kernel(float *X, const float *deltaX, const int *index2vertex, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int v = index2vertex[i];
	X[3 * v + 0] += deltaX[i];
	X[3 * v + 1] += deltaX[i + number];
	X[3 * v + 2] += deltaX[i + number * 2];
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint Kernel 2
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Constraint_2_Kernel(float* prev_X, float* X, float* next_X, float omega, int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=number)	return;	

	next_X[i*3+0]=(next_X[i*3+0]-X[i*3+0])*0.666+X[i*3+0];
	next_X[i*3+1]=(next_X[i*3+1]-X[i*3+1])*0.666+X[i*3+1];
	next_X[i*3+2]=(next_X[i*3+2]-X[i*3+2])*0.666+X[i*3+2];

	next_X[i*3+0]=omega*(next_X[i*3+0]-prev_X[i*3+0])+prev_X[i*3+0];
	next_X[i*3+1]=omega*(next_X[i*3+1]-prev_X[i*3+1])+prev_X[i*3+1];
	next_X[i*3+2]=omega*(next_X[i*3+2]-prev_X[i*3+2])+prev_X[i*3+2];
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint Kernel 3
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Constraint_3_Kernel(float* X, float* init_B, float* V, const float *M,const float *fixed, const float *more_fixed, float inv_t, int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=number)	return;

	float c=(M[i]+fixed[i]+more_fixed[i])*inv_t*inv_t;
	V[i*3+0]+=(X[i*3+0]-init_B[i*3+0]/c)*inv_t;
	V[i*3+1]+=(X[i*3+1]-init_B[i*3+1]/c)*inv_t;
	V[i*3+2]+=(X[i*3+2]-init_B[i*3+2]/c)*inv_t;
}



///////////////////////////////////////////////////////////////////////////////////////////
//  class CUDA_PROJECTIVE_TET_MESH
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
class CUDA_PROJECTIVE_TET_MESH: public TET_MESH<TYPE> 
{
	using TET_MESH<TYPE>::M;
	using TET_MESH<TYPE>::Tet;
public:
	TYPE	cost[8];
	int		cost_ptr;
	TYPE	fps;
	TYPE*	old_X;
	TYPE*	V;
	int*	fixed;
	int* more_fixed;
	TYPE*   fixed_X;

	TYPE	rho;
	TYPE	elasticity;
	TYPE	control_mag;
	TYPE    collision_mag;
	TYPE	damping;

	TYPE*	MD;			//matrix diagonal
	//matrix full
	int *MF_rowInd;
	int *MF_colInd;
	TYPE *MF_Val;
	int MF_nnz;
	TYPE *MF_Diag;

	int *MF_GS_U_Ptr;
	int *MF_GS_U_rowInd;
	int *MF_GS_U_colInd;
	TYPE *MF_GS_U_Val;
	int MF_GS_U_nnz;


	int**   UtAUs_rowInd;
	int** UtAUs_rowPtr;
	int** UtAUs_colInd;
	TYPE** UtAUs_Val;
	int* UtAUs_nnz;
	TYPE** UtAUs_Diag;

	int** UtAUs_GS_U_Ptr;
	int** UtAUs_GS_U_rowInd;
	int** UtAUs_GS_U_rowPtr;
	int** UtAUs_GS_U_colInd;
	TYPE** UtAUs_GS_U_Val;
	int *UtAUs_GS_U_nnz;

	TYPE *precomputed_Diag_addition;
	int **handle_child_num;
	int **handle_child;

	//reduction
	std::vector<int> handles;
	int** handle;
	int** vertex2index;
	int** index2vertex;
	int** color_vertices_num;
	int* colors_num;
	int *where_to_update;

	int** Us_rowInd;
	int** Us_rowPtr;
	int** Us_colInd;
	TYPE** Us_Val;
	int* Us_nnz;
	int* dims;

	//reduction setting
	int layer;
	bool *stored_as_dense;
	int *handles_num;
	int *iterations_num;

	TYPE*	TQ;
	TYPE*	Tet_Temp;
	int*	VTT;		//The index list mapping to Tet_Temp
	int*	vtt_num;

	int* e_num;
	int* e_to;
	TYPE* e_dist;

	//CUDA data
	TYPE*	dev_X;
	TYPE*	dev_old_X;
	TYPE* dev_inertia_X;
	TYPE*	dev_E;
	TYPE*	dev_V;
	TYPE*	dev_next_X;		// next X		(for temporary storage)
	TYPE*	dev_prev_X;		// previous X	(for Chebyshev)
	int*	dev_fixed;
	TYPE*   dev_fixed_X;
	int*	dev_more_fixed;
	int* dev_collision_fixed;
	int** dev_update_flag;
	TYPE*	dev_init_B;		// Initialized momentum condition in B
	
	//for conjugated gradient
	TYPE **dev_temp_X;
	TYPE **dev_deltaX;
	TYPE **dev_R;
	TYPE **dev_P;
	TYPE **dev_AP;

	//for reduction
	int **dev_Us_rowPtr;
	int **dev_Us_colInd;
	TYPE **dev_Us_Val;

	int *sub_U_rowPtr;
	int *dev_sub_U_rowPtr;
	int *dev_sub_U_colInd;
	TYPE *dev_sub_U_Val;
	int sub_U_non_zero_row_number;
	int *sub_U_non_zero_row_index;
	int sub_U_nnz;
	TYPE *result_coeff;

	int **dev_Uts_rowPtr;
	int **dev_Uts_colInd;
	TYPE **dev_Uts_Val;

	int **dev_UtAUs_rowInd;
	int **dev_UtAUs_rowPtr;
	int **dev_UtAUs_colInd;
	TYPE **dev_UtAUs_Val;
	TYPE **dev_UtAUs_Val_Dense;
	TYPE **dev_UtAUs_Diag;
	TYPE **dev_UtAUs_Diag_addition;

	int **dev_UtAUs_GS_U_rowInd;
	int **dev_UtAUs_GS_U_rowPtr;
	int **dev_UtAUs_GS_U_colInd;
	TYPE **dev_UtAUs_GS_U_Val;

	TYPE *dev_precomputed_Diag_addition;
	int **dev_handle_child_num;
	int **dev_handle_child;

	TYPE **dev_delta_UtAUs_Val_Dense;
	TYPE *dev_temp_deltaA_U;


	TYPE*	dev_Dm;
	TYPE*	dev_inv_Dm;
	TYPE*	dev_Vol;
	int*	dev_Tet;
	TYPE*	dev_TQ;
	TYPE*	dev_Tet_Temp;
	TYPE*	dev_M;
	TYPE*	dev_VC;
	TYPE*	dev_MD;

	TYPE*   dev_MF;
	TYPE* dev_MF_Diag;
	TYPE* dev_MF_Diag_addition;

	int* dev_MF_rowInd;
	int* dev_MF_rowPtr;
	int* dev_MF_colInd;
	TYPE* dev_MF_Val;

	int* dev_MF_GS_U_rowInd;
	int* dev_MF_GS_U_rowPtr;
	int* dev_MF_GS_U_colInd;
	TYPE* dev_MF_GS_U_Val;

	int *dev_where_to_update;
	int *dev_handles_num;

	int** dev_vertex2index;
	int** dev_index2vertex;

	int* dev_colors;

	int*	dev_VTT;
	int*	dev_vtt_num;

	TYPE*	error;
	TYPE*	dev_error;

	cublasHandle_t cublasHandle;
	cusparseHandle_t cusparseHandle;
	cusparseMatDescr_t descr;
	cusparseMatDescr_t descrU;
	cusparseMatDescr_t descrL;
	cusparseSolveAnalysisInfo_t infoU;
	cusparseSolveAnalysisInfo_t infoL;

	cudaStream_t stream1, stream2, stream3;

	CUDA_PROJECTIVE_TET_MESH()
	{
		cost_ptr= 0;

		old_X	= new TYPE	[max_number*3];
		V		= new TYPE	[max_number*3];
		fixed	= new int	[max_number  ];
		more_fixed = new int[max_number];
		fixed_X = new TYPE	[max_number*3];

		MD		= new TYPE	[max_number  ];
		TQ		= new TYPE	[max_number*4];
		Tet_Temp= new TYPE	[max_number*24];

		VTT		= new int	[max_number*4];
		vtt_num	= new int	[max_number  ];

		error	= new TYPE	[max_number*3];

		fps			= 0;
		elasticity	= 3000000; //5000000
		control_mag	= 10;
		collision_mag = 1;
		rho			= 0.9992;
		damping		= 0.9995;

		memset(		V, 0, sizeof(TYPE)*max_number*3);
		memset(	fixed, 0, sizeof(int )*max_number  );

		// GPU data
		dev_X			= 0;
		dev_E			= 0;
		dev_V			= 0;
		dev_next_X		= 0;
		dev_prev_X		= 0;
		dev_fixed		= 0;
		dev_more_fixed	= 0;
		dev_init_B		= 0;

		dev_Dm			= 0;
		dev_inv_Dm		= 0;
		dev_Vol			= 0;
		dev_Tet			= 0;
		dev_TQ			= 0;
		dev_Tet_Temp	= 0;
		dev_VC			= 0;
		dev_MD			= 0;
		dev_MF = 0;
		dev_VTT			= 0;
		dev_vtt_num		= 0;

		dev_error		= 0;
		cublasHandle = 0;
		cusparseHandle = 0;
		descr = 0;
	}
	
	~CUDA_PROJECTIVE_TET_MESH()
	{
		if (old_X)			delete[] old_X;
		if (V)				delete[] V;
		if (fixed)			delete[] fixed;
		if (MD)				delete[] MD;
		if (MF_rowInd)	delete[] MF_rowInd;
		if (MF_colInd) delete[] MF_colInd;
		if (MF_Val) delete[] MF_Val;
		if (TQ)				delete[] TQ;
		if (Tet_Temp)		delete[] Tet_Temp;
		if (VTT)				delete[] VTT;
		if (vtt_num)			delete[] vtt_num;
		if (error)			delete[] error;

		//GPU Data
		if (dev_X)			cudaFree(dev_X);
		if (dev_E)			cudaFree(dev_E);
		if (dev_V)			cudaFree(dev_V);
		if (dev_next_X)		cudaFree(dev_next_X);
		if (dev_prev_X)		cudaFree(dev_prev_X);
		if (dev_fixed)		cudaFree(dev_fixed);
		if (dev_more_fixed)	cudaFree(dev_more_fixed);
		if (dev_init_B)		cudaFree(dev_init_B);

		if (dev_Dm)			cudaFree(dev_Dm);
		if (dev_inv_Dm)		cudaFree(dev_inv_Dm);
		if (dev_Vol)			cudaFree(dev_Vol);
		if (dev_Tet)			cudaFree(dev_Tet);
		if (dev_TQ)			cudaFree(dev_TQ);
		if (dev_Tet_Temp)	cudaFree(dev_Tet_Temp);
		if (dev_VC)			cudaFree(dev_VC);
		if (dev_MD)			cudaFree(dev_MD);
		if (dev_MF)			cudaFree(dev_MF);
		if (dev_VTT)			cudaFree(dev_VTT);
		if (dev_vtt_num)		cudaFree(dev_vtt_num);

		if (dev_error)		cudaFree(dev_error);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Initialize functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(TYPE t)
	{
		printf("\n");
		TET_MESH<TYPE>::Initialize();
		printf(".");
		//Initialize_MD();
		InitializeContext();
		printf(".");
		//Coloring();
		//Initialize_MF(t);
		Build_VTT();
		printf(".");
		for(int t=0; t<tet_number; t++)
		{
			TQ[t*4+0]=0;
			TQ[t*4+1]=0;
			TQ[t*4+2]=0;
			TQ[t*4+3]=1;
		}

		dims = (int*)malloc(sizeof(int)*(layer + 1));
		dims[layer] = number;
		for (int l = 0; l < layer; l++)
			dims[l] = 4 * handles_num[l];
		
		memset(fixed_X, 0, sizeof(TYPE)*3*number);
		for(int v=0;v<number;v++)
			if (fixed[v])
			{
				fixed_X[3 * v + 0] = X[3 * v + 0];
				fixed_X[3 * v + 1] = X[3 * v + 1];
				fixed_X[3 * v + 2] = X[3 * v + 2];
			}
		printf(".");
#ifdef N_REST_POSE
		for (int v = 0; v < number; v++)
			X[v * 3 + 1] *= 2;
#endif
		Allocate_GPU_Memory();
		printf(".");
		build_Graph();
		printf(".");
		select_handles();
		printf(".");
		reOrder();
		printf(".");
#ifdef MATRIX_OUTPUT
		outputA(t);
		outputU();
#endif
		compute_U_Matrix();
		printf(".");
		compute_A_Matrix(t);
		printf(".");
		CPUCoo2GPUCsr();
		printf(".");
		prepare_Diag_Part();
		checkCudaErrors(cudaGetLastError());
		printf(".");
		printf("\n");
	}
	
	void Initialize_MD()
	{
		memset(MD, 0, sizeof(TYPE)*number);
		for(int t=0; t<tet_number; t++)
		{
			int*	v=&Tet[t*4];

			TYPE	idm[12];			
			memcpy(idm, &inv_Dm[t*9], sizeof(TYPE)*9);
			idm[ 9]=-(idm[0]+idm[3]+idm[6]);
			idm[10]=-(idm[1]+idm[4]+idm[7]);
			idm[11]=-(idm[2]+idm[5]+idm[8]);

			TYPE	M[12][12];
			for(int i=0; i<12; i++)
			{
				TYPE sum=0;
				for(int j=0; j<12; j++)
				{
					M[i][j]=idm[i]*idm[j];
					if(i!=j)	sum+=fabs(M[i][j]);
				}				
			}

			MD[v[0]]+=(idm[0]+idm[3]+idm[6])*(idm[0]+idm[3]+idm[6])*Vol[t]*elasticity;
			MD[v[0]]+=(idm[1]+idm[4]+idm[7])*(idm[1]+idm[4]+idm[7])*Vol[t]*elasticity;
			MD[v[0]]+=(idm[2]+idm[5]+idm[8])*(idm[2]+idm[5]+idm[8])*Vol[t]*elasticity;

			MD[v[1]]+=idm[0]*idm[0]*Vol[t]*elasticity;
			MD[v[1]]+=idm[1]*idm[1]*Vol[t]*elasticity;
			MD[v[1]]+=idm[2]*idm[2]*Vol[t]*elasticity;

			MD[v[2]]+=idm[3]*idm[3]*Vol[t]*elasticity;
			MD[v[2]]+=idm[4]*idm[4]*Vol[t]*elasticity;
			MD[v[2]]+=idm[5]*idm[5]*Vol[t]*elasticity;

			MD[v[3]]+=idm[6]*idm[6]*Vol[t]*elasticity;
			MD[v[3]]+=idm[7]*idm[7]*Vol[t]*elasticity;
			MD[v[3]]+=idm[8]*idm[8]*Vol[t]*elasticity;
		}		
	}

	void Build_VTT()
	{
		int* _VTT=new int[tet_number*8];
		for(int t=0; t<tet_number*4; t++)
		{
			_VTT[t*2+0]=Tet[t];
			_VTT[t*2+1]=t;
		}
		Quick_Sort_VTT(_VTT, 0, tet_number*4-1);
		
		for(int i=0, v=-1; i<tet_number*4; i++)
		{
			if(_VTT[i*2+0]!=v)	//start a new vertex
			{
				v=_VTT[i*2+0];
				vtt_num[v]=i;
			}
			VTT[i]=_VTT[i*2+1];
		}
		vtt_num[number]=tet_number*4;		
		delete[] _VTT;
	}	

	void Quick_Sort_VTT(int a[], int l, int r)
	{				
		if(l>=r)	return;
		int j=Quick_Sort_Partition_VTT(a, l, r);
		Quick_Sort_VTT(a, l, j-1);
		Quick_Sort_VTT(a, j+1, r);		
	}
	
	int Quick_Sort_Partition_VTT(int a[], int l, int r) 
	{
		int pivot[2], i, j;
		pivot[0] = a[l*2+0];
		pivot[1] = a[l*2+1];
		i = l; j = r+1;		
		while( 1)
		{
			do ++i; while( (a[i*2]<pivot[0] || a[i*2]==pivot[0] && a[i*2+1]<=pivot[1]) && i <= r );
			do --j; while(  a[j*2]>pivot[0] || a[j*2]==pivot[0] && a[j*2+1]> pivot[1] );
			if( i >= j ) break;
			//Swap i and j
			Swap(a[i*2+0], a[j*2+0]);
			Swap(a[i*2+1], a[j*2+1]);
		}
		//Swap l and j
		Swap(a[l*2+0], a[j*2+0]);
		Swap(a[l*2+1], a[j*2+1]);
		return j;
	}

	void InitializeContext()
	{
		cublasCreate(&cublasHandle);
		cusparseCreate(&cusparseHandle);

		cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

		cusparseCreateMatDescr(&descrU);
		cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
		cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
		cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);

		cusparseCreateMatDescr(&descrL);
		cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
		cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);
		cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);

		cusparseCreateSolveAnalysisInfo(&infoU);
		cusparseCreateSolveAnalysisInfo(&infoL);
	}

	void Allocate_GPU_Memory()
	{
		cudaError_t err;
		//Allocate CUDA memory
		err = cudaMalloc((void**)&dev_X,			sizeof(TYPE)*3*number);
		err = cudaMalloc((void**)&dev_old_X,		sizeof(TYPE)*3*number);
		err = cudaMalloc((void**)&dev_inertia_X,	sizeof(TYPE)*3*number);
		err = cudaMalloc((void**)&dev_E,			sizeof(int )*3*number);
		err = cudaMalloc((void**)&dev_V,			sizeof(TYPE)*3*number);
		err = cudaMalloc((void**)&dev_next_X,		sizeof(TYPE)*3*number);
		err = cudaMalloc((void**)&dev_prev_X,		sizeof(TYPE)*3*number);
		err = cudaMalloc((void**)&dev_fixed,		sizeof(int )*  number);
		err = cudaMalloc((void**)&dev_more_fixed,	sizeof(int )*  number);
		err = cudaMalloc((void**)&dev_collision_fixed, sizeof(int)*number);
		err = cudaMalloc((void**)&dev_fixed_X,		sizeof(TYPE)*3*number);
		err = cudaMalloc((void**)&dev_init_B,		sizeof(TYPE)*3*number);

		
		dev_temp_X = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_deltaX = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_R = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_P = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_AP = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));

		for (int l = 0; l <= layer; l++)
		{
			err = cudaMalloc((void**)&dev_temp_X[l],	sizeof(TYPE)*3*dims[l]);
			err = cudaMalloc((void**)&dev_deltaX[l],	sizeof(TYPE)*3*dims[l]);
			err = cudaMalloc((void**)&dev_R[l],			sizeof(TYPE)*3*dims[l]);
			err = cudaMalloc((void**)&dev_P[l],			sizeof(TYPE)*3*dims[l]);
			err = cudaMalloc((void**)&dev_AP[l],		sizeof(TYPE)*3*dims[l]);
		}

		err = cudaMalloc((void**)&dev_Dm,			sizeof(TYPE)*tet_number*9);
		err = cudaMalloc((void**)&dev_inv_Dm,		sizeof(TYPE)*tet_number*9);
		err = cudaMalloc((void**)&dev_Vol,			sizeof(TYPE)*tet_number);
		err = cudaMalloc((void**)&dev_Tet,			sizeof(TYPE)*tet_number*4);
		err = cudaMalloc((void**)&dev_TQ,			sizeof(TYPE)*tet_number*4);
		err = cudaMalloc((void**)&dev_Tet_Temp,		sizeof(TYPE)*tet_number*12);
		
		err = cudaMalloc((void**)&dev_VC,			sizeof(TYPE)*number);
		err = cudaMalloc((void**)&dev_M,			sizeof(TYPE)*number);
// warning
		err = cudaMalloc((void**)&dev_MD,			sizeof(TYPE)*number);
// end 

		err = cudaMalloc((void**)&dev_VTT,			sizeof(int )*tet_number*4);
		err = cudaMalloc((void**)&dev_vtt_num,		sizeof(int )*(number+1));

		err = cudaMalloc((void**)&dev_colors,		sizeof(int)*number);

		err = cudaMalloc((void**)&dev_error,		sizeof(TYPE)*3*number);

		//Copy data into CUDA memory
		err = cudaMemcpy(dev_X,				X,			sizeof(TYPE)*3*number,		cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_V,				V,			sizeof(TYPE)*3*number,		cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_prev_X,		X,			sizeof(TYPE)*3*number,		cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_next_X,		X,			sizeof(TYPE)*3*number,		cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_fixed,			fixed,		sizeof(TYPE)*number,		cudaMemcpyHostToDevice);
		err = cudaMemset(dev_more_fixed,	0,			sizeof(TYPE)*number);
		err = cudaMemcpy(dev_fixed_X,		fixed_X,	sizeof(TYPE)*3*number,		cudaMemcpyHostToDevice);

		err = cudaMemcpy(dev_Dm,			Dm,			sizeof(int)*tet_number*9,	cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_inv_Dm,		inv_Dm,		sizeof(int)*tet_number*9,	cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_Vol,			Vol,		sizeof(int)*tet_number,		cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_Tet,			Tet,		sizeof(int)*tet_number*4,	cudaMemcpyHostToDevice);

		err = cudaMemcpy(dev_M,				M,			sizeof(TYPE)*number,		cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_MD,			MD,			sizeof(TYPE)*number,		cudaMemcpyHostToDevice);

		
		err = cudaMemcpy(dev_VTT,			VTT,		sizeof(int )*tet_number*4,	cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_TQ,			TQ,			sizeof(TYPE)*tet_number*4,	cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_vtt_num,		vtt_num,	sizeof(int )*(number+1),	cudaMemcpyHostToDevice);
	}

	void build_Graph()
	{
		std::pair<int, int>* _E = new std::pair<int, int>[tet_number * 12];
		int t1, t2;
		t1 = 0;
		for (int i = 0; i < tet_number; i++)
			for (int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
					if(j!=k)
						_E[t1++] = std::make_pair(Tet[4 * i + j], Tet[4 * i + k]);
		sort(_E, _E + tet_number * 12);
		
		e_num = new int[number + 1];
		e_to = new int[tet_number * 12];
		e_dist = new TYPE[tet_number * 12];
		t1 = 0; t2 = 0;
		int pre_i = -1, pre_to = -1;
		for (int i = 0; i < tet_number * 12; i++)
		{
			std::pair<int, int> e = _E[i];
			if (e.first == pre_i && e.second == pre_to) continue;
			if (e.first != pre_i)
				e_num[t1++] = t2;
			e_to[t2] = e.second;
			e_dist[t2] = dist(e.first, e.second);
			t2++;
			pre_i = e.first; pre_to = e.second;
		}
		e_num[number] = t2;
		delete _E;
	}


	TYPE dist(int v1, int v2)
	{
		TYPE d = 0;
		for (int k = 0; k < 3; k++)
			d += (X[3 * v1 + k] - X[3 * v2 + k])*(X[3 * v1 + k] - X[3 * v2 + k]);
		return sqrt(d);
	}

	std::vector<TYPE> computeShortestPath(int v)
	{
		struct temp
		{
			TYPE d;
			int i;
			bool operator<(const temp& right) const { return d > right.d; }
		};
		std::vector<TYPE> s(number);
		std::fill(s.begin(), s.end(), inf);
		std::priority_queue<temp> pq;
		pq.push(temp{ 0,v });
		s[v] = 0;
		while (!pq.empty())
		{
			temp top = pq.top();
			pq.pop();
			if (top.d > s[top.i]) continue;
			for (int i = e_num[top.i]; i < e_num[top.i + 1]; i++)
			{
				int to = e_to[i];
				TYPE nd = top.d + e_dist[i];
				if (nd < s[to])
				{
					s[to] = nd;
					pq.push(temp{ nd,to });
				}
			}
		}
		return s;
	}

	void select_handles()
	{
		if (!layer) return;
		handle = (int**)malloc(sizeof(int*)*(layer + 1));
		std::vector<std::vector<TYPE>> acc_nd;
		std::vector<TYPE> pre_nd;
		int init_handle = 0;
		std::vector<TYPE> nd = computeShortestPath(init_handle);
		handles.push_back(init_handle);
		acc_nd.push_back(nd);
		for (int l = 0; l < layer; l++)
		{
			while (handles.size() < handles_num[l])
			{
				int next_handle = std::distance(nd.begin(), std::max_element(nd.begin(), nd.end()));
				handles.push_back(next_handle);
				std::vector<TYPE> nnd = computeShortestPath(next_handle);
				acc_nd.push_back(nnd);
				std::transform(nd.begin(), nd.end(), nnd.begin(), nd.begin(), [](TYPE a, TYPE b)->TYPE {return std::min(a, b); });
			}
			if (l)
			{
				handle[l] = new int[handles_num[l]];
				for (int i = 0; i < handles_num[l]; i++)
				{
					TYPE mini_dist = inf;
					int arg_min = -1;
					for (int j = 0; j < handles_num[l - 1]; j++)
						if (acc_nd[j][handles[i]] < mini_dist)
						{
							mini_dist = acc_nd[j][handles[i]];
							arg_min = j;
						}
					handle[l][i] = arg_min;
				}
			}
		}
		handle[layer] = new int[number];
		for (int i = 0; i < number; i++)
		{
			TYPE mini_dist = inf;
			int arg_min = -1;
			for (int j = 0; j < handles_num[layer-1]; j++)
				if (acc_nd[j][i] < mini_dist)
				{
					mini_dist = acc_nd[j][i];
					arg_min = j;
				}
			handle[layer][i] = arg_min;
		}
	}

	void Coloring(int* t_e_num,int* t_e_to,int* t_coloring,int t_number,int t_init_color_num)
	{
		srand(0);
		bool* fixed = new bool[t_number];
		memset(fixed, 0, sizeof(bool)*t_number);
		int unknown = t_number;
		int pre_unknown = t_number;
		int c_num = t_init_color_num;/* should be something related to degree */
		bool expand_flag;
		int stuck = 0;
		while (unknown)
		{
			expand_flag = false;
			bool *used = new bool[c_num];
			for (int i = 0; i < t_number; i++)
				if (!fixed[i])
				{
					memset(used, 0, sizeof(bool)*c_num);
					for (int j = t_e_num[i]; j < t_e_num[i + 1]; j++)
					{
						int t = t_e_to[j];
						if (fixed[t])
							used[t_coloring[t]] = true;
					}
					int available = 0;
					for (int j = 0; j < c_num; j++)
						if (!used[j]) available++;
					if (available == 0)
					{
						expand_flag = true;
						t_coloring[i] = c_num;
					}
					else
					{
						int index = rand() % available;
						for (int j = 0; j < c_num; j++)
							if (!used[j])
								if (!(index--))
								{
									t_coloring[i] = j;
									break;
								}
					}
				}
			for (int i = 0; i < t_number; i++)
				if (!fixed[i])
				{
					bool valid = true;
					for (int j = t_e_num[i]; j < t_e_num[i + 1]; j++)
						valid &= (t_coloring[i] != t_coloring[t_e_to[j]]);
					if (valid)
					{
						fixed[i] = true;
						unknown--;
					}
				}
			delete used;
			if (expand_flag)
				c_num++;
			else
			{
				if (pre_unknown == unknown)
					stuck++;
				else
					stuck = 0;
				if (stuck == 4)
				{
					c_num++;
					stuck = 0;
				}
			}
			pre_unknown = unknown;
		}
		delete fixed;
		//int num = 0;
		//for (int i = 0; i < number; i++)
		//	colors_num = colors[i] > colors_num ? colors[i] : colors_num;
		//cudaErr = cudaMemcpy(dev_colors, colors, sizeof(int)*number, cudaMemcpyHostToDevice);
	}

	void reOrder()
	{
		cudaError_t cudaErr;

		vertex2index = (int**)malloc(sizeof(int*)*(layer+1));
		index2vertex = (int**)malloc(sizeof(int*)*(layer + 1));
		color_vertices_num = (int**)malloc(sizeof(int*)*(layer + 1));
		colors_num = (int*)malloc(sizeof(int)*(layer + 1));
		dev_vertex2index = (int**)malloc(sizeof(int*)*(layer + 1));
		dev_index2vertex = (int**)malloc(sizeof(int*)*(layer + 1));

		int *coloring = new int[number];
		std::pair<int, int> *temp = new std::pair<int, int>[e_num[number] * 2];
		int *temp_e_num=new int[number+1], *temp_e_to=new int[e_num[number]];

		for (int l = layer; l >= 0; l--)
		{
			if (l == layer)
			{
				memcpy(temp_e_num, e_num, sizeof(int)*(number + 1));
				memcpy(temp_e_to, e_to, sizeof(int)*e_num[number]);
			}
			else
			{
				// build graph
				int *_temp_e_num = new int[handles_num[l] + 1], *_temp_e_to = new int[temp_e_num[handles_num[l + 1]] * 2];
				int t = 0;
				for (int i = 0; i < handles_num[l + 1]; i++)
					for (int j = temp_e_num[i]; j < temp_e_num[i + 1]; j++)
						if (handle[l + 1][i] != handle[l + 1][temp_e_to[j]])
						{
							temp[t++] = std::make_pair(handle[l + 1][i], handle[l + 1][temp_e_to[j]]);
							temp[t++] = std::make_pair(handle[l + 1][temp_e_to[j]], handle[l + 1][i]);
						}
				sort(temp, temp + t);
				int pre_i = -1, pre_to = -1;
				int t1 = 0, t2 = 0;
				for (int i = 0; i < t; i++)
				{
					if (pre_i == temp[i].first && pre_to == temp[i].second) continue;
					if (pre_i != temp[i].first)
					{
						_temp_e_num[t1++] = t2;
						pre_i = temp[i].first;
					}
					_temp_e_to[t2++] = temp[i].second;
					pre_i = temp[i].first; pre_to = temp[i].second;
				}
				_temp_e_num[t1] = t2;

				delete temp_e_num; delete temp_e_to;
				temp_e_num = _temp_e_num; temp_e_to = _temp_e_to;
			}
#ifdef MATRIX_OUTPUT
			if (l < layer)
			{
				char file_name[256];
				sprintf(file_name, "benchmark\\topo-%d.txt", l);
				FILE *file = fopen(file_name, "w");
				fprintf(file,"%d\n", handles_num[l]);
				for (int i = 0; i < handles_num[l]; i++)
				{
					int h = handles[i];
					fprintf(file, "%f %f %f\n", X[3 * h + 0], X[3 * h + 1], X[3 * h + 2]);
				}
				fprintf(file,"%d\n", temp_e_num[handles_num[l]]);
				for (int i = 0; i < handles_num[l]; i++)
					for (int j = temp_e_num[i]; j < temp_e_num[i + 1]; j++)
						fprintf(file, "%d %d\n", i, temp_e_to[j]);
				fclose(file);
			}
#endif
			// coloring
			vertex2index[l] = new int[handles_num[l]];
			index2vertex[l] = new int[handles_num[l]];
			color_vertices_num[l] = new int[handles_num[l]+1];
			//int init_guess = 0.5* temp_e_num[handles_num[l]] / handles_num[l];
			Coloring(temp_e_num, temp_e_to, coloring, handles_num[l], 5);
			for (int i = 0; i < handles_num[l]; i++)
				temp[i] = std::make_pair(coloring[i], i);
			sort(temp, temp + handles_num[l]);
			int pre = -1;
			colors_num[l] = 0;
			for (int i = 0; i < handles_num[l]; i++)
			{
				if (temp[i].first != pre)
				{
					color_vertices_num[l][colors_num[l]++] = i;
					pre = temp[i].first;
				}
				vertex2index[l][temp[i].second] = i;
				index2vertex[l][i] = temp[i].second;
			}
			color_vertices_num[l][colors_num[l]] = handles_num[l];


			cudaErr = cudaMalloc(&dev_vertex2index[l], sizeof(int)*handles_num[l]);
			cudaErr = cudaMalloc(&dev_index2vertex[l], sizeof(int)*handles_num[l]);
			cudaErr = cudaMemcpy(dev_vertex2index[l], vertex2index[l], sizeof(int)*handles_num[l], cudaMemcpyHostToDevice);
			cudaErr = cudaMemcpy(dev_index2vertex[l], index2vertex[l], sizeof(int)*handles_num[l], cudaMemcpyHostToDevice);
		}
	}

	void outputA(TYPE t)
	{
		TYPE inv_t = 1 / t;
		std::map<std::pair<int, int>, TYPE> cooMap;
		for (int t = 0; t < tet_number; t++)
		{
			int*	v = &Tet[t * 4];

			TYPE	idm[12];
			memcpy(&idm[3], &inv_Dm[t * 9], sizeof(TYPE) * 9);
			idm[0] = -(idm[3] + idm[6] + idm[9]);
			idm[1] = -(idm[4] + idm[7] + idm[10]);
			idm[2] = -(idm[5] + idm[8] + idm[11]);

			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
				{
					TYPE dot = 0;
					for (int k = 0; k < 3; k++)
						dot += idm[i * 3 + k] * idm[j * 3 + k];
#ifdef NO_ORDER
					cooMap[std::make_pair(v[i],v[j])]+= dot * Vol[t] * elasticity;
#else
					cooMap[std::make_pair(vertex2index[layer][v[i]], vertex2index[layer][v[j]])] += dot * Vol[t] * elasticity;
#endif
				}
		}
		for (int i = 0; i < number; i++)
		{
			TYPE& entry = cooMap[std::make_pair(i, i)];
#ifdef NO_ORDER
			int v = i;
#else
			int v = index2vertex[layer][i];
#endif
			entry += M[v] * inv_t*inv_t;
			if (fixed[v])
				entry += control_mag;
		}

		FILE *f = fopen("benchmark\\A.txt", "w");
		fprintf(f, "%d\n%d\n%d\n", number, number, cooMap.size());
		for (auto iter : cooMap)
		{
			int r = iter.first.first, c = iter.first.second;
			TYPE v = iter.second;
			fprintf(f, "%d %d %f\n", r, c, v);
		}
		fclose(f);
	}

	void outputU()
	{
		if (!layer)
			return;
		for (int l = 0; l < layer - 1; l++)
		{
			char file_name[256];
			sprintf(file_name, "benchmark\\U%d.txt", l);
			FILE *f = fopen(file_name, "w");
			fprintf(f, "%d\n%d\n%d\n", dims[l + 1], dims[l], handles_num[l + 1] * 4);
			for (int i = 0; i < handles_num[l + 1]; i++)
			{
				int j = handle[l + 1][i];
#ifdef NO_ORDER
				int r = i;
				int c = j;
#else
				int r = vertex2index[l + 1][i];
				int c = vertex2index[l][j];
#endif
				for (int t = 0; t < 4; t++)
					fprintf(f, "%d %d %f\n", 4 * r + t, 4 * c + t, 1.0);
			}
			fclose(f);
		}
		char file_name[256];
		sprintf(file_name, "benchmark\\U%d.txt", layer-1);
		FILE *f = fopen(file_name, "w");
		fprintf(f, "%d\n%d\n%d\n", dims[layer], dims[layer - 1], number * 4);
		for (int i = 0; i < number; i++)
		{
			int j = handle[layer][i];
#ifdef NO_ORDER
			int r = i;
			int c = j;
#else
			int r = vertex2index[layer][i];
			int c = vertex2index[layer - 1][j];
#endif
			fprintf(f, "%d %d %f\n", r, 4 * c + 0, X[3 * i + 0]);
			fprintf(f, "%d %d %f\n", r, 4 * c + 1, X[3 * i + 1]);
			fprintf(f, "%d %d %f\n", r, 4 * c + 2, X[3 * i + 2]);
			fprintf(f, "%d %d %f\n", r, 4 * c + 3, 1.0);
		}
		fclose(f);
	}

	void compute_U_Matrix()
	{
		if (!layer) return;
		Us_rowPtr = (int**)malloc(sizeof(int*)*layer);
		Us_colInd = (int**)malloc(sizeof(int*)*layer);
		Us_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		Us_nnz= (int*)malloc(sizeof(int)*layer);
		for (int l = 0; l < layer - 1; l++)
		{
			Us_rowPtr[l] = new int[dims[l+1] + 1];
			Us_colInd[l] = new int[4 * handles_num[l + 1]];
			Us_Val[l] = new TYPE[4 * handles_num[l + 1]];
			Us_nnz[l] = 4 * handles_num[l + 1];
			int i1 = 0;
			for (int r = 0; r < handles_num[l + 1]; r++)
			{
				int c = vertex2index[l][handle[l + 1][index2vertex[l + 1][r]]];
				for (int t = 0; t < 4; t++)
				{
					Us_rowPtr[l][4*r+t] = i1;
					Us_colInd[l][i1] = 4 * c + t;
					Us_Val[l][i1]=1.0;
					i1++;
				}
			}
			Us_rowPtr[l][dims[l+1]] = i1;
		}

		Us_rowPtr[layer-1] = new int[dims[layer] + 1];
		Us_colInd[layer-1] = new int[4 * handles_num[layer]];
		Us_Val[layer-1] = new TYPE[4 * handles_num[layer]];
		Us_nnz[layer - 1] = 4 * handles_num[layer];
		int i1 = 0;
		for (int r = 0; r < number; r++)
		{
			int v = index2vertex[layer][r];
			int c = vertex2index[layer - 1][handle[layer][v]];
			Us_rowPtr[layer-1][r] = i1;
			for (int t = 0; t < 4; t++)
			{
				Us_colInd[layer - 1][i1] = 4 * c + t;
				Us_Val[layer - 1][i1] = (t == 3) ? 1 : X[3 * v + t];
				i1++;
			}
		}
		Us_rowPtr[layer - 1][dims[layer]] = i1;

	}

	void sortCoo(int *rowInd, int *colInd, TYPE *Val, int number)
	{
		std::pair<int, std::pair<int, TYPE>> *temp = new std::pair<int, std::pair<int, TYPE>>[number];
		for (int i = 0; i < number; i++)
			temp[i] = std::make_pair(rowInd[i], std::make_pair(colInd[i], Val[i]));
		sort(temp, temp + number);
		for (int i = 0; i < number; i++)
		{
			rowInd[i] = temp[i].first;
			colInd[i] = temp[i].second.first;
			Val[i] = temp[i].second.second;
		}
	}

	void compute_A_Matrix(TYPE t)
	{
		// finest layer
		TYPE inv_t = 1 / t;
		std::map<std::pair<int, int>, TYPE> cooMap;
		for (int t = 0; t < tet_number; t++)
		{
			int*	v = &Tet[t * 4];

			TYPE	idm[12];
			memcpy(&idm[3], &inv_Dm[t * 9], sizeof(TYPE) * 9);
			idm[0] = -(idm[3] + idm[6] + idm[9]);
			idm[1] = -(idm[4] + idm[7] + idm[10]);
			idm[2] = -(idm[5] + idm[8] + idm[11]);

			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
				{
					TYPE dot = 0;
					for (int k = 0; k < 3; k++)
						dot += idm[i * 3 + k] * idm[j * 3 + k];
					cooMap[std::make_pair(vertex2index[layer][v[i]], vertex2index[layer][v[j]])] += dot * Vol[t] * elasticity;
				}
		}
		for (int i = 0; i < number; i++)
		{
			TYPE& entry = cooMap[std::make_pair(i, i)];
			int v = index2vertex[layer][i];
			entry += M[v] * inv_t*inv_t;
			if (fixed[v])
				entry += control_mag;
		}
		{
			MF_nnz = cooMap.size();
			MF_rowInd = new int[MF_nnz];
			MF_colInd = new int[MF_nnz];
			MF_Val = new TYPE[MF_nnz];
			MF_Diag = new TYPE[number];

			MF_GS_U_nnz = (MF_nnz - number) >> 1;
			MF_GS_U_Ptr = new int[colors_num[layer] + 1];
			MF_GS_U_rowInd = new int[MF_GS_U_nnz];
			MF_GS_U_colInd = new int[MF_GS_U_nnz];
			MF_GS_U_Val = new TYPE[MF_GS_U_nnz];

			int i1 = 0, i2 = 0;
			int p_base = -1;
			int r_base = -1, c_base = 0;
			for (auto iter : cooMap)
			{
				int r = iter.first.first, c = iter.first.second;
				TYPE v = iter.second;
				MF_rowInd[i1] = r;
				MF_colInd[i1] = c;
				MF_Val[i1] = v;
				i1++;

				if (r == c)
					MF_Diag[r] = v;
				if (r < c)
				{
					while (r >= c_base)
					{
						p_base++;
						MF_GS_U_Ptr[p_base] = i2;
						r_base = color_vertices_num[layer][p_base];
						c_base = color_vertices_num[layer][p_base + 1];
					}
					MF_GS_U_rowInd[i2] = r - r_base;
					MF_GS_U_colInd[i2] = c - c_base;
					MF_GS_U_Val[i2] = v;
					i2++;
				}
			}
			for (; p_base < colors_num[layer] - 1;)
				MF_GS_U_Ptr[++p_base] = i2;
		}
		//finest layer end
		if (!layer)
			return;

		UtAUs_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_colInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		UtAUs_nnz = (int*)malloc(sizeof(int)*layer);
		UtAUs_Diag = (TYPE**)malloc(sizeof(TYPE*)*layer);

		UtAUs_GS_U_Ptr = (int**)malloc(sizeof(int*)*layer);
		UtAUs_GS_U_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_GS_U_colInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_GS_U_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		UtAUs_GS_U_nnz = (int*)malloc(sizeof(int)*layer);

		//initial coo44Map from cooMap
		std::map<std::pair<int, int>, TYPE*> coo44Map;
		for (auto iter : cooMap)
		{
			int r = iter.first.first, c = iter.first.second;
			TYPE v = iter.second;
			int i = index2vertex[layer][r], j = index2vertex[layer][c];
			int rd = vertex2index[layer-1][handle[layer][i]];
			int cd = vertex2index[layer-1][handle[layer][j]];
			std::pair<int, int> key = std::make_pair(rd, cd);
			if (coo44Map.find(key) == coo44Map.end())
			{
				coo44Map[key] = new TYPE[16];
				memset(coo44Map[key], 0,sizeof(TYPE) * 16);
			}

			for (int dj = 0; dj < 4; dj++)
				for (int di = 0; di < 4; di++)
				{
					TYPE v1 = (di == 3) ? 1 : X[3 * i + di];
					TYPE v2 = (dj == 3) ? 1 : X[3 * j + dj];
					coo44Map[key][4 * dj + di] += v * v1*v2;
				}
		}

		//the rest case
		for (int l = layer - 1; l >= 0; l--)
		{
			UtAUs_nnz[l] = coo44Map.size() * 16;
			UtAUs_rowInd[l] = new int[UtAUs_nnz[l]];
			UtAUs_colInd[l] = new int[UtAUs_nnz[l]];
			UtAUs_Val[l] = new TYPE[UtAUs_nnz[l]];
			UtAUs_Diag[l] = new TYPE[16 * handles_num[l]];

			int i1 = 0, i2 = 0;
			int p_base = -1;
			int r_base = -1, c_base = 0;
			for (auto iter : coo44Map)
			{
				int r = iter.first.first, c = iter.first.second;
				TYPE *v = iter.second;
				for (int dj = 0; dj < 4; dj++)
					for (int di = 0; di < 4; di++)
					{
						UtAUs_rowInd[l][i1] = r * 4 + di;
						UtAUs_colInd[l][i1] = c * 4 + dj;
						UtAUs_Val[l][i1] = v[4 * dj + di];
						i1++;
					}
				if (r == c)
				{
					for (int dj = 0; dj < 4; dj++)
						for (int di = 0; di < 4; di++)
							UtAUs_Diag[l][16 * r + dj * 4 + di] = v[4 * dj + di];
				}
			}
			sortCoo(UtAUs_rowInd[l], UtAUs_colInd[l], UtAUs_Val[l], UtAUs_nnz[l]);

			if (!stored_as_dense[l])
			{
				UtAUs_GS_U_nnz[l] = (UtAUs_nnz[l] - 16 * handles_num[l]) >> 1;
				UtAUs_GS_U_Ptr[l] = new int[colors_num[l] + 1];
				UtAUs_GS_U_rowInd[l] = new int[UtAUs_GS_U_nnz[l]];
				UtAUs_GS_U_colInd[l] = new int[UtAUs_GS_U_nnz[l]];
				UtAUs_GS_U_Val[l] = new TYPE[UtAUs_GS_U_nnz[l]];

				for (auto iter : coo44Map)
				{
					int r = iter.first.first, c = iter.first.second;
					TYPE *v = iter.second;

					if (r < c)
					{
						while (r * 4 >= c_base)
						{
							p_base++;
							UtAUs_GS_U_Ptr[l][p_base] = i2;
							r_base = color_vertices_num[l][p_base] * 4;
							c_base = color_vertices_num[l][p_base + 1] * 4;
						}
						for (int dj = 0; dj < 4; dj++)
							for (int di = 0; di < 4; di++)
							{
								UtAUs_GS_U_rowInd[l][i2] = r * 4 + di - r_base;
								UtAUs_GS_U_colInd[l][i2] = c * 4 + dj - c_base;;
								UtAUs_GS_U_Val[l][i2] = v[4 * dj + di];
								i2++;
							}
					}
				}
				for (; p_base < colors_num[l] - 1;)
					UtAUs_GS_U_Ptr[l][++p_base] = i2;

				for (int i = 0; i < colors_num[l] - 1; i++)
					sortCoo(UtAUs_GS_U_rowInd[l] + UtAUs_GS_U_Ptr[l][i], UtAUs_GS_U_colInd[l] + UtAUs_GS_U_Ptr[l][i], \
						UtAUs_GS_U_Val[l] + UtAUs_GS_U_Ptr[l][i], UtAUs_GS_U_Ptr[l][i + 1] - UtAUs_GS_U_Ptr[l][i]);
			}
			if (l)
			{
				std::map<std::pair<int, int>, TYPE*> _coo44Map;
				for (auto iter : coo44Map)
				{
					int r = iter.first.first, c = iter.first.second;
					int i = index2vertex[l][r], j = index2vertex[l][c];
					int rd = vertex2index[l-1][handle[l][i]];
					int cd = vertex2index[l-1][handle[l][j]];
					TYPE *v = iter.second;
					std::pair<int, int> key = std::make_pair(rd, cd);
					if (_coo44Map.find(key) == _coo44Map.end())
					{
						_coo44Map[key] = new TYPE[16];
						memset(_coo44Map[key], 0, sizeof(TYPE) * 16);
					}
					for (int dj = 0; dj < 4; dj++)
						for (int di = 0; di < 4; di++)
							_coo44Map[key][4 * dj + di] += v[4 * dj + di];
				}
				for (auto iter : coo44Map)
					delete iter.second;
				coo44Map = _coo44Map;
			}
		}
	}

	void CPUCoo2GPUCsr()
	{
		cudaError_t err;
		cusparseStatus_t cuspErr;

		// Us
		dev_Us_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_Us_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_Us_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);

		dev_Uts_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_Uts_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_Uts_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);

		for (int l = 0; l < layer; l++)
		{
			err = cudaMalloc((void**)&dev_Us_rowPtr[l], sizeof(int)*(dims[l + 1] + 1));
			err = cudaMalloc((void**)&dev_Us_colInd[l], sizeof(int)*Us_nnz[l]);
			err = cudaMalloc((void**)&dev_Us_Val[l], sizeof(TYPE)*Us_nnz[l]);

			err = cudaMemcpy(dev_Us_rowPtr[l], Us_rowPtr[l], sizeof(int)*(dims[l + 1] + 1), cudaMemcpyHostToDevice);
			err = cudaMemcpy(dev_Us_colInd[l], Us_colInd[l], sizeof(int)*Us_nnz[l], cudaMemcpyHostToDevice);
			err = cudaMemcpy(dev_Us_Val[l], Us_Val[l], sizeof(int)*Us_nnz[l], cudaMemcpyHostToDevice);

			err = cudaMalloc((void**)&dev_Uts_rowPtr[l], sizeof(int)*(dims[l] + 1));
			err = cudaMalloc((void**)&dev_Uts_colInd[l], sizeof(int)*Us_nnz[l]);
			err = cudaMalloc((void**)&dev_Uts_Val[l], sizeof(TYPE)*Us_nnz[l]);

			cuspErr = cusparseScsr2csc(cusparseHandle, dims[l + 1], dims[l], Us_nnz[l], dev_Us_Val[l], dev_Us_rowPtr[l], dev_Us_colInd[l], \
				dev_Uts_Val[l], dev_Uts_colInd[l], dev_Uts_rowPtr[l], CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

		}

		// MF

		err = cudaMalloc((void**)&dev_MF_rowInd, sizeof(int)*MF_nnz);
		err = cudaMalloc((void**)&dev_MF_rowPtr, sizeof(int)*(number + 1));
		err = cudaMalloc((void**)&dev_MF_colInd, sizeof(int)*MF_nnz);
		err = cudaMalloc((void**)&dev_MF_Val, sizeof(TYPE)*MF_nnz);
		err = cudaMalloc((void**)&dev_MF_Diag, sizeof(TYPE)*number);

		err = cudaMemcpy(dev_MF_rowInd, MF_rowInd, sizeof(int)*MF_nnz, cudaMemcpyHostToDevice);
		checkCudaErrors(cusparseXcoo2csr(cusparseHandle, dev_MF_rowInd, MF_nnz, number, dev_MF_rowPtr, CUSPARSE_INDEX_BASE_ZERO));
		err = cudaMemcpy(dev_MF_colInd, MF_colInd, sizeof(int)*MF_nnz, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_MF_Val, MF_Val, sizeof(TYPE)*MF_nnz, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_MF_Diag, MF_Diag, sizeof(TYPE)*number, cudaMemcpyHostToDevice);

		err = cudaMalloc((void**)&dev_MF_GS_U_rowInd, sizeof(int)*MF_GS_U_nnz);
		err = cudaMalloc((void**)&dev_MF_GS_U_rowPtr, sizeof(int)*(number + colors_num[layer] + 1));
		err = cudaMalloc((void**)&dev_MF_GS_U_colInd, sizeof(int)*MF_GS_U_nnz);
		err = cudaMalloc((void**)&dev_MF_GS_U_Val, sizeof(int)*MF_GS_U_nnz);

		err = cudaMemcpy(dev_MF_GS_U_rowInd, MF_GS_U_rowInd, sizeof(int)*MF_GS_U_nnz, cudaMemcpyHostToDevice);

		for (int i = 0; i < colors_num[layer] - 1; i++)
		{
			cuspErr=cusparseXcoo2csr(cusparseHandle, dev_MF_GS_U_rowInd + MF_GS_U_Ptr[i], MF_GS_U_Ptr[i + 1] - MF_GS_U_Ptr[i], \
				color_vertices_num[layer][i + 1] - color_vertices_num[layer][i], dev_MF_GS_U_rowPtr + color_vertices_num[layer][i] + i, CUSPARSE_INDEX_BASE_ZERO);
		}

		err = cudaMemcpy(dev_MF_GS_U_colInd, MF_GS_U_colInd, sizeof(int)*MF_GS_U_nnz, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_MF_GS_U_Val, MF_GS_U_Val, sizeof(TYPE)*MF_GS_U_nnz, cudaMemcpyHostToDevice);

		// UtAUs
		dev_UtAUs_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		dev_UtAUs_Val_Dense = (TYPE**)malloc(sizeof(TYPE*)*layer);
		dev_UtAUs_Diag = (TYPE**)malloc(sizeof(TYPE*)*layer);

		dev_UtAUs_GS_U_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_GS_U_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_GS_U_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_GS_U_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);

		for (int l = layer - 1; l >= 0; l--)
		{
			if (stored_as_dense[l])
			{
				TYPE *temp = new TYPE[dims[l] * dims[l]];
				memset(temp, 0, sizeof(TYPE)*dims[l] * dims[l]);
				for (int i = 0; i < UtAUs_nnz[l]; i++)
					temp[dims[l] * UtAUs_colInd[l][i] + UtAUs_rowInd[l][i]] = UtAUs_Val[l][i];
				err = cudaMalloc((void**)&dev_UtAUs_Val_Dense[l], sizeof(float)*dims[l]*dims[l]);
				err = cudaMemcpy(dev_UtAUs_Val_Dense[l], temp, sizeof(float)*dims[l] * dims[l], cudaMemcpyHostToDevice);
			}
			else
			{
				err = cudaMalloc((void**)&dev_UtAUs_rowInd[l], sizeof(int)*UtAUs_nnz[l]);
				err = cudaMalloc((void**)&dev_UtAUs_rowPtr[l], sizeof(int)*(dims[l] + 1));
				err = cudaMalloc((void**)&dev_UtAUs_colInd[l], sizeof(int)*UtAUs_nnz[l]);
				err = cudaMalloc((void**)&dev_UtAUs_Val[l], sizeof(TYPE)*UtAUs_nnz[l]);

				err = cudaMemcpy(dev_UtAUs_rowInd[l], UtAUs_rowInd[l], sizeof(int)*UtAUs_nnz[l], cudaMemcpyHostToDevice);
				checkCudaErrors(cusparseXcoo2csr(cusparseHandle, dev_UtAUs_rowInd[l], UtAUs_nnz[l], dims[l], dev_UtAUs_rowPtr[l], CUSPARSE_INDEX_BASE_ZERO));
				err = cudaMemcpy(dev_UtAUs_colInd[l], UtAUs_colInd[l], sizeof(int)*UtAUs_nnz[l], cudaMemcpyHostToDevice);
				err = cudaMemcpy(dev_UtAUs_Val[l], UtAUs_Val[l], sizeof(TYPE)*UtAUs_nnz[l], cudaMemcpyHostToDevice);

				err = cudaMalloc((void**)&dev_UtAUs_GS_U_rowInd[l], sizeof(int)*UtAUs_GS_U_nnz[l]);
				err = cudaMalloc((void**)&dev_UtAUs_GS_U_rowPtr[l], sizeof(int)*(4*handles_num[l] + colors_num[l] + 1));
				err = cudaMalloc((void**)&dev_UtAUs_GS_U_colInd[l], sizeof(int)*UtAUs_GS_U_nnz[l]);
				err = cudaMalloc((void**)&dev_UtAUs_GS_U_Val[l], sizeof(int)*UtAUs_GS_U_nnz[l]);

				err = cudaMemcpy(dev_UtAUs_GS_U_rowInd[l] , UtAUs_GS_U_rowInd[l], sizeof(int)*UtAUs_GS_U_nnz[l], cudaMemcpyHostToDevice);
				for (int i = 0; i < colors_num[l] - 1; i++)
				{
					cuspErr = cusparseXcoo2csr(cusparseHandle, dev_UtAUs_GS_U_rowInd[l] + UtAUs_GS_U_Ptr[l][i], UtAUs_GS_U_Ptr[l][i + 1] - UtAUs_GS_U_Ptr[l][i], \
						4 * (color_vertices_num[l][i + 1] - color_vertices_num[l][i]), dev_UtAUs_GS_U_rowPtr[l] + 4 * color_vertices_num[layer][i] + i, \
						CUSPARSE_INDEX_BASE_ZERO);
				}

				err = cudaMemcpy(dev_UtAUs_GS_U_colInd[l], UtAUs_GS_U_colInd[l], sizeof(int)*UtAUs_GS_U_nnz[l], cudaMemcpyHostToDevice);
				err = cudaMemcpy(dev_UtAUs_GS_U_Val[l], UtAUs_GS_U_Val[l], sizeof(TYPE)*UtAUs_GS_U_nnz[l], cudaMemcpyHostToDevice);
			}

			err = cudaMalloc(&dev_UtAUs_Diag[l], sizeof(float) * 16 * handles_num[l]);
			err = cudaMemcpy(dev_UtAUs_Diag[l], UtAUs_Diag[l], sizeof(float) * 16 * handles_num[l], cudaMemcpyHostToDevice);

		}
	}

	void prepare_Diag_Part()
	{
		cudaError_t cudaErr;
		//things to avoid racing condition
		//handle_child_num = (int**)malloc(sizeof(int*)*layer);
		//handle_child = (int**)malloc(sizeof(int*)*layer);
		//std::pair<int, int> *temp = new std::pair<int, int>[number];
		//for (int l = 0; l < layer; l++)
		//{
		//	for (int i = 0; i < handles_num[l + 1]; i++)
		//		temp[i] = std::make_pair(handle[l + 1][i], i);
		//	sort(temp, temp + handles_num[l + 1]);
		//	handle_child_num[l] = new int[handles_num[l] + 1];
		//	handle_child = new int[handles_num[l + 1]];
		//	int pre = -1;
		//	int i1 = 0;
		//	for (int i = 0; i < handles_num[l + 1]; i++)
		//	{
		//		if (pre != temp[i].first)
		//		{
		//			pre = temp[i].first;
		//			handle_child_num[i1++] = i;
		//		}
		//		handle_child[i] = temp[i].second;
		//	}
		//}
		//delete temp;

		//dev_handle_child_num = (int**)malloc(sizeof(int*)*layer);
		//dev_handle_child = (int**)malloc(sizeof(int*)*layer);

		//for (int l = 0; l < layer; l++)
		//{
		//	cudaErr = cudaMalloc(&dev_handle_child_num[l], sizeof(int)*(handles_num[l] + 1));
		//	cudaErr = cudaMalloc(&dev_handle_child[l], sizeof(int)*handles_num[l + 1]);
		//	cudaErr = cudaMemcpy(dev_handle_child_num[l], handle_child_num[l], sizeof(int)*(handles_num[l] + 1), cudaMemcpyHostToDevice);
		//	cudaErr = cudaMemcpy(dev_handle_child[l], handle_child[l], sizeof(int)*(handles_num[l + 1]), cudaMemcpyHostToDevice);
		//}

		//dev_update_flag = (int**)malloc(sizeof(int*)*layer);
		//for (int l = 0; l < layer; l++)
		//	cudaErr = cudaMalloc(&dev_update_flag[l], sizeof(int)*handles_num[l]);

		// precomputed diag
		precomputed_Diag_addition = new TYPE[number * 16];

		for(int i=0;i<number;i++)
			for (int dj = 0; dj < 4; dj++)
				for (int di = 0; di < 4; di++)
				{
					TYPE v1 = di == 3 ? 1:X[3 * i + di];
					TYPE v2 = dj == 3 ? 1:X[3 * i + dj];
					precomputed_Diag_addition[i * 16 + 4 * dj + di] = v1 * v2;
				}

		cudaErr = cudaMalloc(&dev_precomputed_Diag_addition, sizeof(float)*number * 16);
		cudaErr = cudaMemcpy(dev_precomputed_Diag_addition, precomputed_Diag_addition, sizeof(float)*number * 16, cudaMemcpyHostToDevice);

		cudaErr = cudaMalloc(&dev_MF_Diag_addition, sizeof(float)*number);

		dev_UtAUs_Diag_addition = (TYPE**)malloc(sizeof(TYPE*)*layer);
		

		// things to use atomic add
		if (layer)
		{
			//weird part
			int sum = 0;
			for (int l = 0; l < layer; l++)
				sum += handles_num[l];
			TYPE *temp;
			cudaErr = cudaMalloc(&temp, sizeof(TYPE) * 16 * sum);
			sum = 0;
			for (int l = 0; l < layer; l++)
			{
				//cudaErr = cudaMalloc(&dev_UtAUs_Diag_addition[l], sizeof(float) * 16 * handles_num[l]);
				dev_UtAUs_Diag_addition[l] = temp + sum;
				sum += handles_num[l] * 16;
			}

			where_to_update = new int[layer*number];
			for (int i = 0; i < number; i++)
			{
				int now = i;
				for (int l = layer - 1; l >= 0; l--)
				{
					where_to_update[i*layer + l] = vertex2index[l][handle[l + 1][now]];
					now = handle[l + 1][now];
				}
			}
			cudaErr = cudaMalloc(&dev_where_to_update, sizeof(int)*layer*number);
			cudaErr = cudaMemcpy(dev_where_to_update, where_to_update, sizeof(int)*layer*number, cudaMemcpyHostToDevice);
		}
		cudaErr = cudaMalloc(&dev_handles_num, sizeof(int)*layer);
		cudaErr = cudaMemcpy(dev_handles_num, handles_num, sizeof(int)*layer, cudaMemcpyHostToDevice);
		
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Control functions
///////////////////////////////////////////////////////////////////////////////////////////
 	void Reset_More_Fixed(int select_v)
	{
		int threadsPerBlock = 64;
		int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;
		Control_Kernel << <blocksPerGrid, threadsPerBlock>> >(dev_X, dev_more_fixed, number, select_v);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Update functions
///////////////////////////////////////////////////////////////////////////////////////////
	//some functions
	void Diag_Update(int l, float* AP, const float* alpha, const float* P)
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;
		TIMER timer;
		//cublasStatus=cublasScopy(cublasHandle, dims[l], P, 1, dev_temp_X[l], 1);
#ifdef PRECOMPUTE_DENSE_UPDATE	
		if (l == layer)
		{
			MG_Diag_Update_Kernel << <((dims[layer] + threadsPerBlock - 1) / threadsPerBlock), threadsPerBlock >> > (dev_temp_X[layer], dev_VC, dims[layer]);
			cublasStatus = cublasSaxpy(cublasHandle, dims[l], alpha, dev_temp_X[l], 1, AP, 1);
		}
		else
		{
			//timer.Start();
			//cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, dims[l], dims[l], &one, dev_delta_UtAUs_Val_Dense[l], dims[l], \
			//	dev_temp_X[l], 1, &zero, &dev_temp_X[l][dims[l]], 1);
			//cudaDeviceSynchronize();
			//printf("layer %d:%f\n", l, timer.Get_Time());
			//cublasStatus = cublasSaxpy(cublasHandle, dims[l], alpha, &dev_temp_X[l][dims[l]], 1, AP, 1);

		}
#endif
#ifdef DEFAULT_UPDATE
		// default
		//cublasStatus=cublasScopy(cublasHandle, dims[l], P, 1, dev_temp_X[l], 1);
		//for (int i = l; i < layer ; i++)
		//	{
		//		cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[i + 1], dims[i], Us_nnz[i], \
		//			&one, descr, dev_Us_Val[i], dev_Us_rowPtr[i], dev_Us_colInd[i], dev_temp_X[i], &zero, dev_temp_X[i + 1]);
		//	}
		//MF_Diag_Update_Kernel << <((dims[layer] + threadsPerBlock - 1) / threadsPerBlock), threadsPerBlock >> > (dev_temp_X[layer], dev_temp_X[layer], dev_MF_Diag_addition, dims[layer]);
		//for (int i = layer - 1; i >= l; i--)
		//	{
		//		cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[i], dims[i + 1], Us_nnz[i], \
		//			&one, descr, dev_Uts_Val[i], dev_Uts_rowPtr[i], dev_Uts_colInd[i], dev_temp_X[i + 1], &zero, dev_temp_X[i]);
		//	}
		if (l == layer)
			MF_Diag_Update_Kernel << <((number + threadsPerBlock - 1) / threadsPerBlock), threadsPerBlock >> > (dev_temp_X[l], dev_MF_Diag_addition, P, number);
		else
			UtAUs_Diag_Update_Kernel << <((handles_num[l] + threadsPerBlock - 1) / threadsPerBlock), threadsPerBlock >> > (dev_temp_X[l], dev_UtAUs_Diag_addition[l], P, handles_num[l]);
		cublasStatus = cublasSaxpy(cublasHandle, dims[l], alpha, dev_temp_X[l], 1, AP, 1);
#endif
#ifdef VECTOR_UPDATE
		if (l == layer)
		{
			MG_Diag_Update_Kernel << <((dims[layer] + threadsPerBlock - 1) / threadsPerBlock), threadsPerBlock >> > (dev_temp_X[layer], dev_VC, dims[layer]);
			cublasStatus = cublasSaxpy(cublasHandle, dims[l], alpha, dev_temp_X[l], 1, AP, 1);
		}
		else
		{
			if (sub_U_non_zero_row_number)
			{
				for (int i = l; i < layer - 1; i++)
				{
					cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[i + 1], dims[i], Us_nnz[i], \
						&one, descr, dev_Us_Val[i], dev_Us_rowPtr[i], dev_Us_colInd[i], dev_temp_X[i], &zero, dev_temp_X[i + 1]);
				}

				cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, sub_U_non_zero_row_number, dims[layer - 1], sub_U_nnz, &control_mag, descr, dev_sub_U_Val, dev_sub_U_rowPtr, \
					dev_sub_U_colInd, dev_temp_X[layer - 1], &zero, dev_temp_X[layer]);

				//cudaErr = cudaMemset(&dev_temp_X[layer - 1][dims[layer - 1]], 0, sizeof(float)*dims[layer - 1]);
				//cudaErr = cudaMemcpy(result_coeff, dev_temp_X[layer], sizeof(float)*(sub_U_non_zero_row_number), cudaMemcpyDeviceToHost);

				//for (int i = 0; i < sub_U_non_zero_row_number; i++)
				//{
				//	int v = sub_U_non_zero_row_index[i];
				//	int start = Us_rowPtr[layer - 1][v];
				//	cuspErr = cusparseSaxpyi(cusparseHandle, Us_rowPtr[layer - 1][v + 1] - start, &result_coeff[i], &dev_Us_Val[layer - 1][start], &dev_Us_colInd[layer - 1][start], \
				//		&dev_temp_X[layer - 1][dims[layer - 1]], CUSPARSE_INDEX_BASE_ZERO);
				//}

				//for (int i = layer - 2; i >= l; i--)
				//{
				//	cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[i], dims[i + 1], Us_nnz[i], \
				//		&one, descr, dev_Uts_Val[i], dev_Uts_rowPtr[i], dev_Uts_colInd[i], &dev_temp_X[i + 1][dims[i+1]], &zero, &dev_temp_X[i][dims[i]]);
				//}
				//cublasStatus = cublasSaxpy(cublasHandle, dims[l], alpha, &dev_temp_X[l][dims[l]], 1, AP, 1);

				cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, sub_U_non_zero_row_number, dims[layer - 1], sub_U_nnz, &one, descr, dev_sub_U_Val, dev_sub_U_rowPtr, \
					dev_sub_U_colInd, dev_temp_X[layer], &zero, dev_temp_X[layer - 1]);
				for (int i = layer - 2; i >= l; i--)
				{
					cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[i], dims[i + 1], Us_nnz[i], \
						&one, descr, dev_Uts_Val[i], dev_Uts_rowPtr[i], dev_Uts_colInd[i], dev_temp_X[i + 1], &zero, dev_temp_X[i]);
				}
				cublasStatus = cublasSaxpy(cublasHandle, dims[l], alpha, dev_temp_X[l], 1, AP, 1);
			}
		}
#endif
	}
	void calculateAP(int l, float *AP, const float *alpha,float *P)
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;
#ifdef PRECOMPUTE_DENSE_UPDATE
		if (l == layer)
		{
			cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l], dims[l], MF_nnz, alpha, descr, dev_MF_Val, dev_MF_rowPtr, dev_MF_colInd, P, &one, AP);
			Diag_Update(l, AP, alpha, P);
		}
		else
		{
			cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, dims[l], dims[l], alpha, dev_delta_UtAUs_Val_Dense[l], dims[l], \
				P, 1, &one, AP, 1);
		}
#endif
#ifdef DEFAULT_UPDATE
		if (l == layer)
			cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l], dims[l], MF_nnz, alpha, descr, \
				dev_MF_Val, dev_MF_rowPtr, dev_MF_colInd, P, &one, AP);
		else
		{
			if (stored_as_dense[l])
				cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, dims[l], dims[l], alpha, dev_UtAUs_Val_Dense[l], dims[l], P, 1, &one, AP, 1);
			else
				cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l], dims[l], UtAUs_nnz[l], alpha, descr, \
					dev_UtAUs_Val[l], dev_UtAUs_rowPtr[l], dev_UtAUs_colInd[l], P, &one, AP);
		}

		Diag_Update(l, AP, alpha, P);
#endif
#ifdef VECTOR_UPDATE
		if (l == layer)
			cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l], dims[l], MF_nnz, alpha, descr, dev_MF_Val, dev_MF_rowPtr, dev_MF_colInd, P, &one, AP);
		else
		{
			cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, dims[l], dims[l], alpha, dev_UtAUs_Val_Dense[l], dims[l], P, 1, &one, AP, 1);
		}

		Diag_Update(l, AP, alpha, P);
#endif
	}

	void performCGIteration(int l,int max_iter,float tol)
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;
		float r0, r1;
		float alpha, beta, neg_alpha;
		float dot;
		int k;
		for (int dim = 0; dim < 3; dim++)
		{
			float *P = &(dev_P[l][dims[l]*dim]);
			float *R = &(dev_R[l][dims[l]*dim]);
			float *AP = &(dev_AP[l][dims[l]*dim]);
			float *deltaX = &(dev_deltaX[l][dims[l]*dim]);
			cublasStatus = cublasSdot(cublasHandle, dims[l], R, 1, R, 1, &r1);
			if (r1 < EPSILON) continue;
			float r = r1;

			k = 1;
			while (r1 > tol*r && (max_iter < 0 || k <= max_iter))
			{
				if (k > 1)
				{
					beta = r1 / r0;
					cublasStatus = cublasSscal(cublasHandle, dims[l], &beta, P, 1);
					cublasStatus = cublasSaxpy(cublasHandle, dims[l], &one, R, 1, P, 1);
				}
				else
				{
					cublasStatus = cublasScopy(cublasHandle, dims[l], R, 1, P, 1);
				}
				//if (l == layer)
				//	cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l], dims[l], MF_nnz, &one, descr, dev_MF_Val, dev_MF_rowPtr, dev_MF_colInd, P, &zero, AP);
				//else
				//{
				//	cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, dims[l], dims[l], &one, dev_UtAUs_Val_Dense[l], dims[l], P, 1, &zero, AP, 1);
				//}

				//Diag_Update(l, AP, &one, P);
				cudaErr = cudaMemset(AP, 0, sizeof(float)*dims[l]); // this should be set by once

				calculateAP(l, AP, &one, P);

				cublasStatus = cublasSdot(cublasHandle, dims[l], P, 1, AP, 1, &dot);
				alpha = r1 / dot;

				cublasStatus = cublasSaxpy(cublasHandle, dims[l], &alpha, P, 1, deltaX, 1);
				neg_alpha = -alpha;
				cublasStatus = cublasSaxpy(cublasHandle, dims[l], &neg_alpha, AP, 1, R, 1);

				r0 = r1;
				cublasStatus = cublasSdot(cublasHandle, dims[l], R, 1, R, 1, &r1);
				cudaDeviceSynchronize();
				//printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
				k++;
			}
		}
	}

	void performGaussSeidelIteration(int l, int max_iter, float tol)
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasErr;
		float r1;
		if (l == layer)
		{
			//printf("this method is not supposed to run on finest level only.\n");
			//return;
			for (int dim = 0; dim < 3; dim++)
			{
				float *P = &(dev_P[l][dims[l] * dim]);
				float *R = &(dev_R[l][dims[l] * dim]);
				float *deltaX = &(dev_deltaX[l][dims[l] * dim]);
				for (int k = 0; k < max_iter; k++)
				{
					//cuspErr = cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, number, &one, descrU, \
					//	dev_MF_U_Val, dev_MF_U_rowPtr, dev_MF_U_colInd, infoU, R, P);

					cublasErr = cublasScopy(cublasHandle, dims[l], R, 1, dev_temp_X[l], 1);
					for (int c = colors_num[l] - 1; c >= 0; c--)
					{
						int base = color_vertices_num[l][c];
						Colored_GS_MF_Kernel << <(color_vertices_num[l][c+1] - base + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (P, dev_MF_Diag, dev_MF_Diag_addition, dev_temp_X[l], base, color_vertices_num[l][c+1] - base);
						cudaErr = cudaGetLastError();
						if (c)
						{
							cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, color_vertices_num[l][c] - color_vertices_num[l][c - 1], \
								color_vertices_num[l][colors_num[l]] - color_vertices_num[l][c], MF_GS_U_Ptr[c] - MF_GS_U_Ptr[c-1], &minus_one, descr, \
								dev_MF_GS_U_Val + MF_GS_U_Ptr[c - 1], dev_MF_GS_U_rowPtr + color_vertices_num[l][c - 1] + c - 1, dev_MF_GS_U_colInd + MF_GS_U_Ptr[c - 1], \
								P + base, &one, dev_temp_X[l] + color_vertices_num[l][c - 1]);
						}
						
					}
					cublasErr = cublasSaxpy(cublasHandle, dims[l], &one, P, 1, deltaX, 1);
					calculateAP(l, R, &minus_one, P);
				}
#ifdef Test
				cublasErr = cublasSdot(cublasHandle, dims[l], R, 1, R, 1, &r1);
				printf("%d: %f\n", dim, r1);
#endif
			}
#ifdef Test
			printf("--------------------\n");
#endif
		}
		else
		{
			for (int dim = 0; dim < 3; dim++)
			{
				float *P = &(dev_P[l][dims[l] * dim]);
				float *R = &(dev_R[l][dims[l] * dim]);
				float *deltaX = &(dev_deltaX[l][dims[l] * dim]);
				for (int k = 0; k < max_iter; k++)
				{
					cublasErr = cublasScopy(cublasHandle, dims[l], R, 1, dev_temp_X[l], 1);
					cudaErr = cudaMemset(P, 0, sizeof(float)*dims[l]);
					for (int c = colors_num[l] - 1; c >= 0; c--)
					{
						int base = color_vertices_num[l][c];
						Colored_GS_UtAU_Kernel << <(color_vertices_num[l][c + 1] - base + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (P, dev_UtAUs_Diag[l], dev_UtAUs_Diag_addition[l], dev_temp_X[l], base, color_vertices_num[l][c + 1] - base);
						cudaErr = cudaGetLastError();
						if (c)
						{
							if (stored_as_dense[l])
								cublasErr = cublasSgemv(cublasHandle, CUBLAS_OP_N, 4 * (color_vertices_num[l][c] - color_vertices_num[l][c - 1]), \
									4 * (color_vertices_num[l][colors_num[l]] - color_vertices_num[l][c]), &minus_one, \
									dev_UtAUs_Val_Dense[l] + dims[l] * 4 * color_vertices_num[l][c] + 4 * color_vertices_num[l][c - 1], dims[l], \
									P + 4 * base, 1, &one, dev_temp_X[l] + 4 * color_vertices_num[l][c - 1], 1);
							else
								cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4 * (color_vertices_num[l][c] - color_vertices_num[l][c - 1]), \
									4 * (color_vertices_num[l][colors_num[l]] - color_vertices_num[l][c]), UtAUs_GS_U_Ptr[l][c] - UtAUs_GS_U_Ptr[l][c - 1], &minus_one, descr, \
									dev_UtAUs_GS_U_Val[l] + 4 * UtAUs_GS_U_Ptr[l][c - 1], dev_UtAUs_GS_U_rowPtr[l] + 4 * color_vertices_num[l][c - 1] + c - 1, dev_UtAUs_GS_U_colInd[l] + UtAUs_GS_U_Ptr[l][c - 1], \
									P + 4 * base, &one, dev_temp_X[l] + 4 * color_vertices_num[l][c - 1]);
						}

					}
					cublasErr = cublasSaxpy(cublasHandle, dims[l], &one, P, 1, deltaX, 1);
					calculateAP(l, R, &minus_one, P);
				}
#ifdef Test
				cublasErr = cublasSdot(cublasHandle, dims[l], R, 1, R, 1, &r1);
				printf("%d: %f\n", dims, r1);
#endif
			}
#ifdef Test
			printf("--------------------\n");
#endif
		}
	}

	void downSample(int& l)
	{
		if (l == 0)
		{
			printf("down sampled in the coarsest layer.\n");
			return;
		}
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;
		for (int dim = 0; dim < 3; dim++)
		{
			float *now_R = &(dev_R[l][dim*dims[l]]);
			float *down_R = &(dev_R[l - 1][dim*dims[l - 1]]);
			//cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, dims[l], dims[l-1], \
				Us_nnz[l - 1], &one, descr, dev_Us_Val[l - 1], dev_Us_rowPtr[l - 1], dev_Us_colInd[l - 1], now_R, &zero, down_R);
			cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l - 1], dims[l], \
				Us_nnz[l - 1], &one, descr, dev_Uts_Val[l - 1], dev_Uts_rowPtr[l - 1], dev_Uts_colInd[l - 1], now_R, &zero, down_R);
		}
		cudaErr = cudaMemset(dev_deltaX[l - 1], 0, sizeof(float) * 3 * dims[l - 1]);
		l--;
	}

	void upSample(int& l)
	{
		if (l == layer)
		{
			printf("up sampled in the finest level.\n");
			return;
		}
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;
		for (int dim = 0; dim < 3; dim++)
		{
			float *up_temp_X = &(dev_temp_X[l + 1][dim*dims[l + 1]]);
			float *now_deltaX = &(dev_deltaX[l][dim*dims[l]]);
			cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l + 1], dims[l], \
				Us_nnz[l], &one, descr, dev_Us_Val[l], dev_Us_rowPtr[l], dev_Us_colInd[l], now_deltaX, &zero, up_temp_X);
		}
		cublasStatus = cublasSaxpy(cublasHandle, dims[l + 1] * 3, &one, dev_temp_X[l + 1], 1, dev_deltaX[l + 1], 1);
		for (int dim = 0; dim < 3; dim++)
		{
			float *up_temp_X = &(dev_temp_X[l + 1][dim*dims[l + 1]]);
			float *up_R = &(dev_R[l + 1][dim*dims[l + 1]]);
			calculateAP(l + 1, up_R, &minus_one, up_temp_X);
		}
		l++;
	}

	void solve(int l)
	{
		if (l != 0)
		{
			printf("solve is only available in the coarsest level.\n");
			return;
		}
	}

	void Update(TYPE t, int iterations, TYPE dir[])
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;

		int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;
		int tet_blocksPerGrid = (tet_number + tet_threadsPerBlock - 1) / tet_threadsPerBlock;

		TIMER fps_timer;
		TIMER timer;

		float now_time = 0;
		
		float g_norm;

		// Step 0: Set up Diag data
		Fixed_Update_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_X, dev_collision_fixed, dev_fixed, dev_more_fixed, dev_fixed_X, number, dir[0], dir[1], dir[2]);

		if (layer)
		{
			int sum = 0;
			for (int l = 0; l < layer; l++)
				sum += handles_num[l] * 16;
			cudaErr = cudaMemset(*dev_UtAUs_Diag_addition, 0, sizeof(float)*sum);
		}
		cudaErr = cudaMemset(dev_MF_Diag_addition, 0, sizeof(float)*number);
		Diag_Update_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_MF_Diag_addition, dev_more_fixed, dev_collision_fixed, dev_vertex2index[layer], control_mag, \
			collision_mag, dev_where_to_update, dev_precomputed_Diag_addition, *dev_UtAUs_Diag_addition, dev_handles_num, number, layer);

		for (int s = 0; s < sub_step; s++)
		{
			cublasStatus = cublasScopy(cublasHandle, number * 3, dev_X, 1, dev_old_X, 1);

			// Step 1: Basic update
			Basic_Update_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_X, dev_V, damping, t, number);
			cudaErr = cudaGetLastError();

			cublasStatus = cublasScopy(cublasHandle, number * 3, dev_X, 1, dev_inertia_X, 1);

			cudaErr = cudaGetLastError();

			// Our Step 2: MultiGrid Method
			timer.Start();

			for (int l = 0; l < pd_iters; l++)
			{
				Tet_Constraint_Kernel << <tet_blocksPerGrid, tet_threadsPerBlock >> > (dev_X, dev_Tet, dev_inv_Dm, dev_Vol, dev_Tet_Temp, elasticity, tet_number, l);
				cudaErr = cudaGetLastError();

				cudaErr = cudaMemset(dev_R[layer], 0, sizeof(float) * 3 * dims[layer]);
				Energy_Gradient_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_R[layer], dev_Tet_Temp, dev_VTT, dev_vtt_num, dev_fixed, dev_more_fixed, \
					dev_collision_fixed, dev_fixed_X, dev_X, dev_vertex2index[layer], control_mag, collision_mag, number);
				Inertia_Gradient_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_R[layer], dev_inertia_X, dev_X, dev_vertex2index[layer], dev_M, 1.0 / t, number);
	
				cudaErr = cudaGetLastError();
#ifdef BENCHMARK
				cublasStatus = cublasSdot(cublasHandle, 3 * number, dev_R[layer], 1, dev_R[layer], 1, &g_norm);
				fprintf(benchmark, "%f %f\n", timer.Get_Time(), g_norm);
#endif

				cudaErr = cudaMemset(dev_deltaX[layer], 0, sizeof(float) * 3 * dims[layer]);

				int now_Layer = layer;
#ifdef SETTING1
				//performGaussSeidelIteration(now_Layer, 30, tol, false);
				performGaussSeidelIteration(now_Layer, 30, tol);
#endif
#ifdef SETTING2

				//V-cycle

				//performGaussSeidelIteration(now_Layer, 5, tol);

				//downSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 5, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 5, tol);

				//FMG

				downSample(now_Layer);

				performGaussSeidelIteration(now_Layer, 5, tol);

				upSample(now_Layer);

				performGaussSeidelIteration(now_Layer, 5, tol);

#endif
#ifdef SETTING3
				//V-cycle

				//performGaussSeidelIteration(now_Layer, 3, tol);
	
				//downSample(now_Layer);
		
				//performGaussSeidelIteration(now_Layer, 3, tol);
		
				//downSample(now_Layer);
			
				//performGaussSeidelIteration(now_Layer, 3, tol);

				//downSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);
			
				//upSample(now_Layer);			

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//FMG

				//downSample(now_Layer);
				//downSample(now_Layer);
				//downSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//downSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//downSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//downSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//upSample(now_Layer);

				//performGaussSeidelIteration(now_Layer, 3, tol);

				//FMG-mogai

				if ((l & 3) == 0)
				{
					downSample(now_Layer);
					downSample(now_Layer);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					upSample(now_Layer);
				}
				if ((l & 3) == 1)
				{
					downSample(now_Layer);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
				}

				if ((l & 3) == 2)
				{
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
				}

				if ((l & 3) == 3)
				{
					performGaussSeidelIteration(now_Layer, 3, tol);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					downSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
					upSample(now_Layer);
					performGaussSeidelIteration(now_Layer, 3, tol);
				}

#endif

				Update_DeltaX_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_X, dev_deltaX[layer], dev_index2vertex[layer], number);
			}
			
			// Step 3: Finalizing update
			cublasStatus = cublasSaxpy(cublasHandle, 3 * number, &minus_one, dev_X, 1, dev_old_X, 1);
			float neg_inv_t = -1.0 / t;
			cublasStatus = cublasSscal(cublasHandle, 3 * number, &neg_inv_t, dev_old_X, 1);
			cublasStatus = cublasScopy(cublasHandle, 3 * number, dev_old_X, 1, dev_V, 1);

#ifdef BENCHMARK
			fprintf(benchmark, "\n");
#endif
		}
		//Output to main memory for rendering
		cudaMemcpy(X, dev_X, sizeof(TYPE) * 3 * number, cudaMemcpyDeviceToHost);

		cost[cost_ptr] = fps_timer.Get_Time();

		cost_ptr = (cost_ptr + 1) % 8;
		fps = 8 / (cost[0] + cost[1] + cost[2] + cost[3] + cost[4] + cost[5] + cost[6] + cost[7]);

	}

///////////////////////////////////////////////////////////////////////////////////////////
//  IO functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Write(std::fstream &output)
	{
		Write_Binaries(output, X, number*3);
		Write_Binaries(output, V, number*3);
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
		Read_Binaries(input, X, number*3);
		Read_Binaries(input, V, number*3);
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