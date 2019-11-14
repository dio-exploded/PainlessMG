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
#ifndef	__WHMIN_CUDA_PROJECTIVE_MESH_H__
#define __WHMIN_CUDA_PROJECTIVE_MESH_H__
#include "DYNAMIC_MESH.h"
#include "TIMER.h"
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <device_launch_parameters.h>
#include<algorithm>
#include<map>
#include<math.h>
#include<queue>
#include<vector>
#include "helper_cuda.h"
//#include "helper_cusolver.h"
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include "Output.h"
#include "cusolverDn.h"


#define GRAVITY			-9.8
#define RADIUS_SQUARED	0.002//0.01
#define CG_EPSILON 1e-10
//#define TST

//#define NO_ORDER
//#define MATRIX_OUTPUT

//#define PRECOMPUTE_DENSE_UPDATE
#define DEFAULT_UPDATE
//#define NO_UPDATE
//#define VECTOR_UPDATE

const float zero = 0.0;
const float one = 1.0;
const float minus_one = -1.0;
const float inf = INFINITY;
const float tol = 1e-5;
const float cg_tol = 1e-15;
const int threadsPerBlock = 64;
const int tet_threadsPerBlock = 64;
const int Reduction_threadsPerBlock = 8;
const int UtAUs_threadPerBlock = 1;

enum iterationType
{
	Jacobi,
	GaussSeidel,
	CG,
	Direct,
	DownSample,
	UpSample
};
typedef std::pair<iterationType, int> iterationSetting;
enum objectType
{
	Plane,
	Sphere
};
__device__ __host__ struct object
{
	objectType type;
	//Plane
	float p_nx;
	float p_ny;
	float p_nz;
	float p_c;
	//Sphere
	float s_cx;
	float s_cy;
	float s_cz;
	float s_r;
};

///////////////////////////////////////////////////////////////////////////////////////////
//  math kernels
///////////////////////////////////////////////////////////////////////////////////////////
__device__ void dev_Matrix_Product_3(const float *A, const float *B, float *R)				//R=A*B
{
	R[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
	R[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
	R[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
	R[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
	R[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
	R[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
	R[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
	R[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
	R[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

__device__ void dev_Matrix_Substract_3(float *A, float *B, float *R)						//R=A-B
{
	for (int i = 0; i < 9; i++)	R[i] = A[i] - B[i];
}

__device__ void dev_Matrix_Product(float *A, float *B, float *R, int nx, int ny, int nz)	//R=A*B
{
	memset(R, 0, sizeof(float)*nx*nz);
	for (int i = 0; i < nx; i++)
		for (int j = 0; j < nz; j++)
			for (int k = 0; k < ny; k++)
				R[i*nz + j] += A[i*ny + k] * B[k*nz + j];
}

__device__ void Get_Rotation(float F[3][3], float R[3][3])
{
	float C[3][3];
	memset(&C[0][0], 0, sizeof(float) * 9);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				C[i][j] += F[k][i] * F[k][j];

	float C2[3][3];
	memset(&C2[0][0], 0, sizeof(float) * 9);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				C2[i][j] += C[i][k] * C[j][k];

	float det = F[0][0] * F[1][1] * F[2][2] +
		F[0][1] * F[1][2] * F[2][0] +
		F[1][0] * F[2][1] * F[0][2] -
		F[0][2] * F[1][1] * F[2][0] -
		F[0][1] * F[1][0] * F[2][2] -
		F[0][0] * F[1][2] * F[2][1];

	float I_c = C[0][0] + C[1][1] + C[2][2];
	float I_c2 = I_c * I_c;
	float II_c = 0.5*(I_c2 - C2[0][0] - C2[1][1] - C2[2][2]);
	float III_c = det * det;
	float k = I_c2 - 3 * II_c;

	float inv_U[3][3];
	if (k < 1e-10f)
	{
		float inv_lambda = 1 / sqrt(I_c / 3);
		memset(inv_U, 0, sizeof(float) * 9);
		inv_U[0][0] = inv_lambda;
		inv_U[1][1] = inv_lambda;
		inv_U[2][2] = inv_lambda;
	}
	else
	{
		float l = I_c * (I_c*I_c - 4.5*II_c) + 13.5*III_c;
		float k_root = sqrt(k);
		float value = l / (k*k_root);
		if (value < -1.0) value = -1.0;
		if (value > 1.0) value = 1.0;
		float phi = acos(value);
		float lambda2 = (I_c + 2 * k_root*cos(phi / 3)) / 3.0;
		float lambda = sqrt(lambda2);

		float III_u = sqrt(III_c);
		if (det < 0)   III_u = -III_u;
		float I_u = lambda + sqrt(-lambda2 + I_c + 2 * III_u / lambda);
		float II_u = (I_u*I_u - I_c)*0.5;

		float U[3][3];
		float inv_rate, factor;

		inv_rate = 1 / (I_u*II_u - III_u);
		factor = I_u * III_u*inv_rate;

		memset(U, 0, sizeof(float) * 9);
		U[0][0] = factor;
		U[1][1] = factor;
		U[2][2] = factor;

		factor = (I_u*I_u - II_u)*inv_rate;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				U[i][j] += factor * C[i][j] - inv_rate * C2[i][j];

		inv_rate = 1 / III_u;
		factor = II_u * inv_rate;
		memset(inv_U, 0, sizeof(float) * 9);
		inv_U[0][0] = factor;
		inv_U[1][1] = factor;
		inv_U[2][2] = factor;

		factor = -I_u * inv_rate;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				inv_U[i][j] += factor * U[i][j] + inv_rate * C[i][j];
	}

	memset(&R[0][0], 0, sizeof(float) * 9);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				R[i][j] += F[i][k] * inv_U[k][j];
}

__device__ void solve3x3(const float *A, const float *b, float *x)
{
	float r[3];
	float p[3];
	float Ap[3];
	float alpha, beta;
	float old_r_norm, r_norm;
	float dot;

	for (int i = 0; i < 3; i++)
		r[i] = b[i];

	r_norm = 0;

	for (int i = 0; i < 3; i++)
		r_norm += r[i] * r[i];

	if (r_norm < CG_EPSILON) return;


	for (int i = 0; i < 3; i++)
		p[i] = r[i];


	for (int l = 0; l < 3; l++)
	{

		for (int i = 0; i < 3; i++)
			Ap[i] = 0;

#pragma unroll
		for (int k = 0; k < 3; k++)
			for (int i = 0; i < 3; i++)
				Ap[i] += (A[k * 3 + i]) * p[k];
		dot = 0;

		for (int i = 0; i < 3; i++)
			dot += Ap[i] * p[i];
		alpha = r_norm / dot;

		if (dot < CG_EPSILON) return;

		for (int i = 0; i < 3; i++)
		{
			x[i] += alpha * p[i];
			r[i] -= alpha * Ap[i];
		}
		old_r_norm = r_norm;
		r_norm = 0;

		for (int i = 0; i < 3; i++)
			r_norm += r[i] * r[i];

		if (r_norm < CG_EPSILON) return;

		beta = r_norm / old_r_norm;

		for (int i = 0; i < 3; i++)
			p[i] = r[i] + beta * p[i];
	}

}

__device__ void solve12x12(const float *A, const float *b, float *x,int f)
{
	float r[12];
	float p[12];
	float Ap[12];
	float alpha, beta;
	float old_r_norm, r_norm;
	float dot;

	for (int i = 0; i < 12; i++)
		r[i] = b[i];

	r_norm = 0;

	for (int i = 0; i < 12; i++)
		r_norm += r[i] * r[i];

	if (r_norm < CG_EPSILON) return;


	for (int i = 0; i < 12; i++)
		p[i] = r[i];

	for (int l = 0; l < 4; l++)
	{
		for (int i = 0; i < 12; i++)
			Ap[i] = 0;

#pragma unroll
		for (int k = 0; k < 12; k++)
			for (int i = 0; i < 12; i++)
				Ap[i] += (A[k * 12 + i]) * p[k];
		dot = 0;

		for (int i = 0; i < 12; i++)
			dot += Ap[i] * p[i];

		alpha = r_norm / dot;

		for (int i = 0; i < 12; i++)
		{
			x[i] += alpha * p[i];
			r[i] -= alpha * Ap[i];
		}
		old_r_norm = r_norm;
		r_norm = 0;

		for (int i = 0; i < 12; i++)
			r_norm += r[i] * r[i];

		if (r_norm < CG_EPSILON) return;

		beta = r_norm / old_r_norm;

		for (int i = 0; i < 12; i++)
			p[i] = r[i] + beta * p[i];
	}

}

__device__ void solve4x4(const float *A, const float *dA, const float *b, float *x)
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

__device__ void collision_detection(const float *X, float *fixed_X, int *fixed)
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
	if (i >= number)	return;

	more_fixed[i] = 0;
	if (select_v != -1)
	{
		float dist2 = 0;
		dist2 += (X[i * 3 + 0] - X[select_v * 3 + 0])*(X[i * 3 + 0] - X[select_v * 3 + 0]);
		dist2 += (X[i * 3 + 1] - X[select_v * 3 + 1])*(X[i * 3 + 1] - X[select_v * 3 + 1]);
		dist2 += (X[i * 3 + 2] - X[select_v * 3 + 2])*(X[i * 3 + 2] - X[select_v * 3 + 2]);
		if (dist2 < RADIUS_SQUARED)	more_fixed[i] = 1;
	}
}

__global__ void Hessian_Kernel(float *A_Val, const float *X, const int *spring_v1, const int *spring_v2, const float *spring_len, const int *update_offset, const float spring_k, int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int v1 = spring_v1[i], v2 = spring_v2[i];
	float len = spring_len[i];
	float dA[9];
	float d[3];
	d[0] = X[v1 * 3 + 0] - X[v2 * 3 + 0];
	d[1] = X[v1 * 3 + 1] - X[v2 * 3 + 1];
	d[2] = X[v1 * 3 + 2] - X[v2 * 3 + 2];
	float len_new = sqrt(DOT(d, d));
	float a = spring_k * len / len_new, b = a / len_new / len_new;

	for (int di = 0; di < 3; di++)
		for (int dj = 0; dj < 3; dj++)
			dA[di * 3 + dj] = b * d[di] * d[dj];
	

	if (len_new > len)
		for (int di = 0; di < 3; di++)
			dA[di * 4] += spring_k - a;

	const int *u = update_offset + 4 * i;

	for (int di = 0; di < 3; di++)
		for (int dj = 0; dj < 3; dj++)
		{
			int off = di * 3 + dj;
			float temp = dA[di * 3 + dj];
			atomicAdd(A_Val + u[0] * 9 + off, temp);
			atomicAdd(A_Val + u[1] * 9 + off, -temp);
			atomicAdd(A_Val + u[2] * 9 + off, -temp);
			atomicAdd(A_Val + u[3] * 9 + off, temp);
		}
}

__global__ void Hessian_Diag_Kernel(float *A_Diag_Val, const float *M, const float inv_t, const int *fixed, const int *more_fixed, const float control_mag, const int *vertex2index, const int *D_offset, int number)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
	float dA = M[v] * inv_t*inv_t;
	if (fixed[v] || more_fixed[v])
		dA += control_mag;
	const int off = 9 * D_offset[vertex2index[v]];
	A_Diag_Val[off] += dA;
	A_Diag_Val[off + 4] += dA;
	A_Diag_Val[off + 8] += dA;
}

__global__ void First_UtAUs_Val_Kernal(float *UtAUs_Val, const float *A_Val, const int *update_offset, const float *precomputed_4x4, int number)
{
	//unroll chu qi ji
	int o = blockDim.x * blockIdx.x + threadIdx.x;
	int tx = threadIdx.y, ty = threadIdx.z;
	if (o >= number)	return;
	float *UtAU = UtAUs_Val + update_offset[o] * 144;
	const float *A = A_Val + 9 * o;
	for (int vi = 0; vi < 4; vi++)
		for (int vj = 0; vj < 4; vj++)
			atomicAdd(UtAU + (tx * 4 + vi) * 12 + ty * 4 + vj, A[tx * 3 + ty] * precomputed_4x4[16 * o + 4 * vi + vj]);
			//UtAU[(tx * 4 + vi) * 12 + ty * 4 + vj] += A[tx * 3 + ty] * precomputed_4x4[16 * o + 4 * vi + vj];
			//UtAU[(tx * 4 + vi) * 12 + ty * 4 + vj] += 1;
}

__global__ void MF_Mapping_Kernel(float *temp_Val, const float *A_Val, const float *precomputed_4x4, const int *half_offset,int number)
{
	int i = blockIdx.x*blockDim.x / 16 + threadIdx.x / 16;
	if (i > number) return;
	int q = threadIdx.x % 16;
	int tx = q / 4, ty = q % 4;
	int po = i / 9, pt = i % 9;
	po = half_offset[po];
	i = po * 9 + pt;
	int px = pt / 3, py = pt % 3;
	//temp_Val[144 * po + (px * 4 + tx) * 12 + py * 4 + ty] = A_Val[i] * precomputed_4x4[16 * po + 4 * tx + ty];
	temp_Val[16 * i + 4 * tx + ty] = A_Val[i] * precomputed_4x4[16 * po + 4 * tx + ty];
}

__global__ void MF_2_UtAU_sparse_reduction_Kernel(float *UtAUs_Val, const float *temp_Val, const int *update_offset, const int *half_offset, int number)
{
	int i = blockIdx.x*blockDim.x / 16 + threadIdx.x / 16;
	if (i > number) return;
	int q = threadIdx.x % 16;
	int tx = q / 4, ty = q % 4;
	int po = i / 9, pt = i % 9;
	po = half_offset[po];
	i = po * 9 + pt;
	int px = pt / 3, py = pt % 3;
	atomicAdd(UtAUs_Val + 144 * update_offset[po] + (px * 4 + tx) * 12 + py * 4 + ty, temp_Val[16 * i + 4 * tx + ty]);
}

__global__ void MF_2_UtAU_dense_reduction_Kernel(float *UtAUs_Val, const float *temp_Val, const int *update_offset, const int *half_offset, const int ldim, int number)
{
	int i = blockIdx.x*blockDim.x / 16 + threadIdx.x / 16;
	if (i > number) return;
	int q = threadIdx.x % 16;
	int tx = q / 4, ty = q % 4;
	int po = i / 9, pt = i % 9;
	po = half_offset[po];
	i = po * 9 + pt;
	int px = pt / 3, py = pt % 3;
	int r = update_offset[po] / ldim, c = update_offset[po] % ldim;
	atomicAdd(UtAUs_Val + (c * 12 + 4 * py + ty) * 12 * ldim + r * 12 + 4 * px + tx, temp_Val[16 * i + 4 * tx + ty]);
}

__global__ void UtAU_2_UtAU_sparse_Kernel(float *UtAUs_Val, const float *temp_Val, const int *update_offset, const int *half_offset,int number)
{
	int i = blockIdx.x*blockDim.y + threadIdx.y;
	if (i > number) return;
	int off = threadIdx.x;
	atomicAdd(UtAUs_Val + 144 * update_offset[i] + off, temp_Val[144 * i + off]);
}

__global__ void UtAU_2_UtAU_dense_Kernel(float *UtAUs_Val, const float *temp_Val, const int *update_offset, const int ldim, int number)
{
	int i = blockIdx.x*blockDim.y + threadIdx.y;
	if (i > number) return;
	int off = threadIdx.x;
	int dx = off / 12, dy = off % 12;
	int r = update_offset[i] / ldim, c = update_offset[i] % ldim;
	atomicAdd(UtAUs_Val + (c * 12 + dy) * 12 * ldim + r * 12 + dx, temp_Val[144 * i + off]);
}


__global__ void UtAUs_sparse_Mirror_Kernel(float *UtAUs_Val,const int *L_offset,  const int *mirror_offset, int number)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x) / 144;
	if (i > number) return;
	int off1 = threadIdx.x % 144;
	int off2 = off1 / 12 + (off1 % 12) * 12;
	UtAUs_Val[144 * mirror_offset[i] + off1] = UtAUs_Val[144 * L_offset[i] + off2];
}

__global__ void UtAUs_dense_Mirror_Kernel(float *UtAUs_Val, const int *L_offset, const int *mirror_offset, int ldim, int number)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x) / 144;
	if (i > number) return;
	int off= threadIdx.x % 144;
	int dx = off / 12, dy = off % 12;
	int r = L_offset[i] / ldim, c = L_offset[i] % ldim;
	UtAUs_Val[(r * 12 + dx) * 12 * ldim + c * 12 + dy] = UtAUs_Val[(c * 12 + dy) * 12 * ldim + r * 12 + dx];
}

__global__ void UtAUs_Diag_Rank_Fix_Kernel(float *UtAUs_Val, const int *D_offset, int number)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x);
	if (i > number) return;
	int off = D_offset[i];
	UtAUs_Val[144 * off + 12 * 1 + 1] += 1.0;
	UtAUs_Val[144 * off + 12 * 5 + 5] += 1.0;
	UtAUs_Val[144 * off + 12 * 9 + 9] += 1.0;
}

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

///////////////////////////////////////////////////////////////////////////////////////////
//  Basic update kernel
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Basic_Update_Kernel(float* X, float* V, const float damping, const float t, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;

	//Apply damping
	V[i * 3 + 0] *= damping;
	V[i * 3 + 1] *= damping;
	V[i * 3 + 2] *= damping;
	//Apply gravity
	//V[i * 3 + 1] += GRAVITY * t;
	//Position update
	X[i * 3 + 0] += V[i * 3 + 0] * t;
	X[i * 3 + 1] += V[i * 3 + 1] * t;
	X[i * 3 + 2] += V[i * 3 + 2] * t;
}


///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint Kernel 0
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Diag_Update_Kernel(float* MF_Diag, const int* more_fixed, const int *collision_fixed, const int *vertex2index, const float control_mag, \
	const float collision_mag, const int *where_to_update, const float *precomputed_diag, float *UtAUs_diag, const int *handles_num, const int number, const int layer)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
#ifdef NO_ORDER
	int i = v;
#else
	int i = vertex2index[v];
#endif
	if (more_fixed[v] || collision_fixed[v])
	{
		float mag = collision_fixed[v] ? collision_mag : control_mag;
		MF_Diag[3 * i + 0] = mag;
		MF_Diag[3 * i + 1] = mag;
		MF_Diag[3 * i + 2] = mag;
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
	if (i >= number)	return;

	double b[3];
	b[0] = init_B[i * 3 + 0] + MD[i] * X[i * 3 + 0];
	b[1] = init_B[i * 3 + 1] + MD[i] * X[i * 3 + 1];
	b[2] = init_B[i * 3 + 2] + MD[i] * X[i * 3 + 2];

	for (int index = vtt_num[i]; index < vtt_num[i + 1]; index++)
	{
		b[0] += Tet_Temp[VTT[index] * 3 + 0];
		b[1] += Tet_Temp[VTT[index] * 3 + 1];
		b[2] += Tet_Temp[VTT[index] * 3 + 2];
	}

	next_X[i * 3 + 0] = b[0] / (VC[i] + MD[i]);
	next_X[i * 3 + 1] = b[1] / (VC[i] + MD[i]);
	next_X[i * 3 + 2] = b[2] / (VC[i] + MD[i]);
}

__global__ void Energy_Gradient_Kernel(float *G, const int* VV, const float *VL, const int* vv_num, float spring_k, \
	const int *fixed, const int *more_fixed, const int *collision_fixed, const float *fixed_X, const float *X, \
	const int *vertex2index, const float control_mag, const float collision_mag, const int number)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
#ifdef NO_ORDER
	int i = v;
#else
	int i = vertex2index[v];
#endif
	for (int index = vv_num[v]; index < vv_num[v + 1]; index++)
	{

		int j = VV[index];
		float d[3];
		d[0] = X[v * 3 + 0] - X[j * 3 + 0];
		d[1] = X[v * 3 + 1] - X[j * 3 + 1];
		d[2] = X[v * 3 + 2] - X[j * 3 + 2];

		float delta_L = spring_k * (VL[index] / sqrt(DOT(d, d)) - 1.0);

		G[3 * i + 0] += delta_L * d[0];
		G[3 * i + 1] += delta_L * d[1];
		G[3 * i + 2] += delta_L * d[2];
	}
	if (fixed[v] || more_fixed[v] || collision_fixed[v])
	{
		float mag = collision_fixed[v] ? collision_mag : control_mag;
		G[3 * i + 0] += mag * (fixed_X[3 * v + 0] - X[3 * v + 0]);
		G[3 * i + 1] += mag * (fixed_X[3 * v + 1] - X[3 * v + 1]);
		G[3 * i + 2] += mag * (fixed_X[3 * v + 2] - X[3 * v + 2]);
	}
}

__global__ void Inertia_Gradient_Kernel(float *G, const float *inertia, const float *X, const int *vertex2index, const float *M, const float inv_t, int number)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
	float c = M[v] * inv_t*inv_t;
#ifdef NO_ORDER
	int i = v;
#else
	int i = vertex2index[v];
#endif
	G[3 * i + 0] += c * (inertia[3 * v + 0] - X[3 * v + 0]);
	G[3 * i + 1] += c * (inertia[3 * v + 1] - X[3 * v + 1]);
	G[3 * i + 2] += c * (inertia[3 * v + 2] - X[3 * v + 2]);

	G[3 * i + 1] += GRAVITY * M[v];
}

__global__ void Energy_Kernel(float *E, const int* VV, const float *VL, const int* vv_num, float spring_k, \
	const int *fixed, const int *more_fixed, const int *collision_fixed, const float *fixed_X, const float *X, \
	const int *vertex2index, const float control_mag, const float collision_mag, const int number)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
#ifdef NO_ORDER
	int i = v;
#else
	int i = vertex2index[v];
#endif
	float e = 0;
	for (int index = vv_num[v]; index < vv_num[v + 1]; index++)
	{

		int j = VV[index];
		float d[3];
		d[0] = X[v * 3 + 0] - X[j * 3 + 0];
		d[1] = X[v * 3 + 1] - X[j * 3 + 1];
		d[2] = X[v * 3 + 2] - X[j * 3 + 2];

		float delta_L = VL[index] - sqrt(DOT(d, d));

		e += 0.5*spring_k * delta_L*delta_L;
	}
	if (fixed[v] || more_fixed[v] || collision_fixed[v])
	{
		float mag = collision_fixed[v] ? collision_mag : control_mag;
		float d[3];
		d[0] = (fixed_X[3 * v + 0] - X[3 * v + 0]);
		d[1] = (fixed_X[3 * v + 1] - X[3 * v + 1]);
		d[2] = (fixed_X[3 * v + 2] - X[3 * v + 2]);
		e += 0.5*mag * DOT(d, d);
	}
	atomicAdd(E, e);
}

__global__ void Inertia_Kernel(float *E, const float *inertia, const float *X, const int *vertex2index, const float *M, const float inv_t, int number)
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	if (v >= number)	return;
	float c = M[v] * inv_t*inv_t;
#ifdef NO_ORDER
	int i = v;
#else
	int i = vertex2index[v];
#endif
	float e = 0;
	float d[3];
	d[0] = (inertia[3 * v + 0] - X[3 * v + 0]);
	d[1] = (inertia[3 * v + 1] - X[3 * v + 1]);
	d[2] = (inertia[3 * v + 2] - X[3 * v + 2]);

	e += 0.5* c * DOT(d, d);

	e -= GRAVITY * M[v] * X[3 * v + 1];

	atomicAdd(E, e);
}

__global__ void MF_Diag_Update_Kernel(float *Y, const float *X, const float *Diag, const int number)
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

__global__ void Colored_GS_MF_Kernel(float *X, const float *A_Diag_Val, const float *b, const int base, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int t = base + i;
	solve3x3(A_Diag_Val + 9 * t, b + 3 * t, X + 3 * t);
}

__global__ void Jacobi_Kernel(float *X, const float *A_Val, const float *b, const int *D_offset, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number) return;
	solve3x3(A_Val + 9 * D_offset[i], b + 3 * i, X + 3 * i);
}

__global__ void Colored_GS_UtAU_Kernel(float *X, const float *UtAU_Diag_Val, const float *b, const int base, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int t = base + i;
	solve12x12(UtAU_Diag_Val + 144 * t, b + 12 * t, X + 12 * t,i);
}

__global__ void Mogai_2_Kernel(float *X, const float *UtAUs_Diag_Val, const float *b, const int base, const int number)
{
	const int i = blockIdx.x;
	const int o = threadIdx.x;
	const int oi = threadIdx.x / 12;
	int t = base + i * 2;
	const float *A = UtAUs_Diag_Val + 144 * t;
	float *x = X + 12 * t;
	__shared__ float r[24], p[24], Ap[24];
	__shared__ float temp[24];
	__shared__ float dot[2], alpha[2], beta[2], r_norm[2], old_r_norm[2];
	__shared__ bool flag[2];
	if (o == 0 || o == 12)
		flag[oi] = ((i * 2 + oi) >= number);
	__syncthreads();
	
	if (!flag[oi])
	{
		r[o] = b[12 * t + o];
		p[o] = r[o];
		temp[o] = r[o] * r[o];
		if (o == 0 || o == 12)
		{
			r_norm[oi] = 0;
			for (int s = 0; s < 12; s++)
				r_norm[oi] += temp[12 * oi + s];
			if (r_norm[oi] < CG_EPSILON)
				flag[oi] = true;
		}
	}
	__syncthreads();
#pragma unroll
	for (int l = 0; l < 4; l++)
	{
		if (!flag[oi])
		{
			Ap[o] = 0;
#pragma unroll
			for (int k = 0; k < 12; k++)
				Ap[o] += A[12 * o + k] * p[12 * oi + k];
			temp[o] = p[o] * Ap[o];
			if (o == 0 || o == 12)
			{
				dot[oi] = 0;
				for (int s = 0; s < 12; s++)
					dot[oi] += temp[12 * oi + s];
				alpha[oi] = r_norm[oi] / dot[oi];
			}
		}
		__syncthreads();
		if (!flag[oi])
		{
			x[o] += alpha[oi] * p[o];
			r[o] -= alpha[oi] * Ap[o];
			temp[o] = r[o] * r[o];
			if (o == 0 || o == 12)
			{
				old_r_norm[oi] = r_norm[oi];
				r_norm[oi] = 0;
				for (int s = 0; s < 12; s++)
					r_norm[oi] += temp[12 * oi + s];
				beta[oi] = r_norm[oi] / old_r_norm[oi];
				if (r_norm[oi] < CG_EPSILON)
					flag[oi] = true;
			}
		}
		__syncthreads();
		if (!flag[oi])
			p[o] = beta[oi] * p[o] + r[o];
	}
}

__global__ void Jacobi_Mogai_Kernel(float *X, const float *UtAU_Val, const float *b, const int *UtAU_D_offset, const int number)
{
	int i = blockIdx.x;
	int o = threadIdx.x;
	const float *A = UtAU_Val + 144 * UtAU_D_offset[i];
	float *x = X + 12 * i;
	__shared__ float r[12], p[12], Ap[12];
	__shared__ volatile float temp[18];
	__shared__ float dot, alpha, beta, r_norm, old_r_norm;
	__shared__ bool flag;
	r[o] = b[12 * i + o];
	p[o] = r[o];
	temp[o] = r[o] * r[o];
	temp[o] += temp[o + 6];
	temp[o] += temp[o + 3];
	//__syncthreads();
	if (o == 0)
	{
		r_norm = temp[0] + temp[1] + temp[2];
		//r_norm = 0;
		//for (int s = 0; s < 12; s++)
		//	r_norm += temp[s];
		flag = false;
		if (r_norm < CG_EPSILON)
			flag = true;
	}
	__syncthreads();
	if (flag) return;

	for (int l = 0; l < 12; l++)
	{
		Ap[o] = 0;
#pragma unroll
		for (int k = 0; k < 12; k++)
			Ap[o] += A[12 * o + k] * p[k];
		temp[o] = p[o] * Ap[o];
		temp[o] += temp[o + 6];
		temp[o] += temp[o + 3];
		//__syncthreads();
		if (o == 0)
		{
			dot = temp[0] + temp[1] + temp[2];
			alpha = r_norm / dot;
			if (dot < CG_EPSILON)
				flag = flag;
		}
		__syncthreads();
		if (flag) return;
		x[o] += alpha * p[o];
		r[o] -= alpha * Ap[o];
		temp[o] = r[o] * r[o];
		temp[o] += temp[o + 6];
		temp[o] += temp[o + 3];
		//__syncthreads();
		if (o == 0)
		{
			old_r_norm = r_norm;
			r_norm = temp[0] + temp[1] + temp[2];
			beta = r_norm / old_r_norm;
			if (r_norm < CG_EPSILON)
				flag = true;
		}
		__syncthreads();
		if (flag) return;
		p[o] = beta * p[o] + r[o];
		//__syncthreads();
	}
}


__global__ void Mogai_Kernel(float *X, const float *UtAU_Diag_Val, const float *b, const int base)
{
	int i = blockIdx.x;
	int o = threadIdx.x;
	int t = base + i;
	const float *A = UtAU_Diag_Val + 144 * t;
	float *x = X + 12 * t;
	__shared__ float r[12], p[12], Ap[12];
	__shared__ volatile float temp[18];
	__shared__ float dot, alpha, beta, r_norm, old_r_norm;
	__shared__ bool flag;
	r[o] = b[12 * t + o];
	p[o] = r[o];
	temp[o] = r[o] * r[o];
	temp[o] += temp[o + 6];
	temp[o] += temp[o + 3];
	//__syncthreads();
	if (o == 0)
	{
		r_norm = temp[0] + temp[1] + temp[2];
		//r_norm = 0;
		//for (int s = 0; s < 12; s++)
		//	r_norm += temp[s];
		flag = false;
		if (r_norm < CG_EPSILON)
			flag = true;
	}
	__syncthreads();
	if (flag) return;
	
	for (int l = 0; l < 4; l++)
	{
		Ap[o] = 0;
#pragma unroll
		for (int k = 0; k < 12; k++)
			Ap[o] += A[12 * o + k] * p[k];
		temp[o] = p[o] * Ap[o];
		temp[o] += temp[o + 6];
		temp[o] += temp[o + 3];
		//__syncthreads();
		if (o == 0)
		{
			dot = temp[0] + temp[1] + temp[2];
			alpha = r_norm / dot;
			if (dot < CG_EPSILON)
				flag = true;
		}
		__syncthreads();
		if (flag) return;
		x[o] += alpha * p[o];
		r[o] -= alpha * Ap[o];
		temp[o] = r[o] * r[o];
		temp[o] += temp[o + 6];
		temp[o] += temp[o + 3];
		//__syncthreads();
		if (o == 0)
		{
			old_r_norm = r_norm;
			r_norm = temp[0] + temp[1] + temp[2];
			beta = r_norm / old_r_norm;
			if (r_norm < CG_EPSILON)
				flag = true;
		}
		__syncthreads();
		if (flag) return;
		p[o] = beta * p[o] + r[o];
		//__syncthreads();
	}
}

__global__ void Update_DeltaX_Kernel(float *X, const float *deltaX, const int *index2vertex, const int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int v = index2vertex[i];
	X[3 * v + 0] += deltaX[3 * i + 0];
	X[3 * v + 1] += deltaX[3 * i + 1];
	X[3 * v + 2] += deltaX[3 * i + 2];
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint Kernel 2
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Constraint_2_Kernel(float* prev_X, float* X, float* next_X, float omega, int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;

	next_X[i * 3 + 0] = (next_X[i * 3 + 0] - X[i * 3 + 0])*0.666 + X[i * 3 + 0];
	next_X[i * 3 + 1] = (next_X[i * 3 + 1] - X[i * 3 + 1])*0.666 + X[i * 3 + 1];
	next_X[i * 3 + 2] = (next_X[i * 3 + 2] - X[i * 3 + 2])*0.666 + X[i * 3 + 2];

	next_X[i * 3 + 0] = omega * (next_X[i * 3 + 0] - prev_X[i * 3 + 0]) + prev_X[i * 3 + 0];
	next_X[i * 3 + 1] = omega * (next_X[i * 3 + 1] - prev_X[i * 3 + 1]) + prev_X[i * 3 + 1];
	next_X[i * 3 + 2] = omega * (next_X[i * 3 + 2] - prev_X[i * 3 + 2]) + prev_X[i * 3 + 2];
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint Kernel 3
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Constraint_3_Kernel(float* X, float* init_B, float* V, const float *M, const float *fixed, const float *more_fixed, float inv_t, int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;

	float c = (M[i] + fixed[i] + more_fixed[i])*inv_t*inv_t;
	V[i * 3 + 0] += (X[i * 3 + 0] - init_B[i * 3 + 0] / c)*inv_t;
	V[i * 3 + 1] += (X[i * 3 + 1] - init_B[i * 3 + 1] / c)*inv_t;
	V[i * 3 + 2] += (X[i * 3 + 2] - init_B[i * 3 + 2] / c)*inv_t;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  class CUDA_PROJECTIVE_TET_MESH
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
class CUDA_PROJECTIVE_MESH : public DYNAMIC_MESH<TYPE>
{
public:
	using   BASE_MESH<TYPE>::max_number;
	using   BASE_MESH<TYPE>::number;
	using   BASE_MESH<TYPE>::t_number;
	using   BASE_MESH<TYPE>::X;
	using   BASE_MESH<TYPE>::M;
	using   BASE_MESH<TYPE>::T;
	using   BASE_MESH<TYPE>::VN;
	using   MESH<TYPE>::e_number;
	using   MESH<TYPE>::E;
	using   MESH<TYPE>::ET;
public:
	FILE	*benchmark;
	TYPE	cost[8];
	int		cost_ptr;
	TYPE	fps;
	TYPE*	old_X;
	TYPE*	V;
	int*	fixed;
	int* more_fixed;
	TYPE*   fixed_X;
	TYPE*	target_X;

	TYPE	rho;
	TYPE	spring_k;
	TYPE	control_mag;
	TYPE    collision_mag;
	TYPE	damping;

	TYPE*	MD;			//matrix diagonal

	int*	all_VV;
	TYPE*	all_VL;
	int*	all_vv_num;

	int spring_num;
	int *spring_v1;
	int *spring_v2;
	TYPE *spring_len;
	int *spring_update_offset;
	TYPE *spring_stiff;

	//matrix full
	int MF_nnz;
	int *MF_rowInd;
	int *MF_colInd;
	TYPE *MF_Val;

	int *MF_U_rowInd;
	int *MF_U_colInd;
	TYPE *MF_U_Val;
	int MF_U_nnz;

	int *MF_GS_U_Ptr;
	int *MF_GS_U_rowInd;
	int *MF_GS_U_colInd;

	int *MF_D_rowInd;
	int *MF_D_colInd;
	TYPE *MF_D_Val;
	int MF_D_nnz;
	int *MF_D_offset;

	int *MF_L_rowInd;
	int *MF_L_colInd;
	TYPE *MF_L_Val;
	int MF_L_nnz;

	int *MF_GS_L_Ptr;
	int *MF_GS_L_rowInd;
	int *MF_GS_L_colInd;

	int MF_PD_nnz;
	int *MF_PD_rowPtr;
	int *MF_PD_rowInd;
	int *MF_PD_colInd;
	TYPE *MF_PD_Val;
	double *MF_PD_Val_d;

	int MF_PD_L_nnz;
	int *MF_PD_L_rowPtr;
	int *MF_PD_L_colInd;
	double *MF_PD_L_Val;

	int MF_PD_U_nnz;
	int *MF_PD_U_rowPtr;
	int *MF_PD_U_colInd;
	double *MF_PD_U_Val;

	int *MF_PD_Plu;
	int *MF_PD_Qlu;
	int *MF_PD_P;
	int *MF_PD_Q;

	int *MF_PD_reorder;
	int *MF_PD_reordered_rowPtr;
	int *MF_PD_reordered_colInd;
	TYPE *MF_PD_reordered_Val;
	void *MF_PD_buffer;

	csrluInfoHost_t MF_PD_info;

	TYPE *PD_X;
	TYPE *PD_B;

	int**   UtAUs_rowInd;
	int** UtAUs_rowPtr;
	int** UtAUs_colInd;
	TYPE** UtAUs_Val;
	int* UtAUs_nnz;
	TYPE** UtAUs_Diag;
	int **UtAUs_update_offset;
	int **UtAUs_mirror_offset;
	int **UtAUs_half_offset;
	int *UtAUs_half_num;
	TYPE **UtAUs_precomputed_4x4;
	int **UtAUs_D_offset;
	int **UtAUs_L_offset;

	int*  UtAUs_U_nnz;
	int** UtAUs_U_rowInd;
	int** UtAUs_U_colInd;
	TYPE** UtAUs_U_Val;
	
	int** UtAUs_GS_U_Ptr;
	int** UtAUs_GS_U_rowInd;
	int** UtAUs_GS_U_colInd;

	int*  UtAUs_D_nnz;
	int** UtAUs_D_rowInd;
	int** UtAUs_D_colInd;
	TYPE** UtAUs_D_Val;

	int*  UtAUs_L_nnz;
	int** UtAUs_L_rowInd;
	int** UtAUs_L_colInd;
	TYPE** UtAUs_L_Val;

	int** UtAUs_GS_L_Ptr;
	int** UtAUs_GS_L_rowInd;
	int** UtAUs_GS_L_colInd;
	

	TYPE *precomputed_Diag_addition;
	int **handle_child_num;
	int **handle_child;

	//reduction
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
	int *stored_as_dense;
	int *stored_as_LDU;
	int *handles_num;
	std::vector<iterationSetting> settings;
	int *iterations_num;

	//simulation setting
	int pd_iters;
	int use_Hessian;
	int use_WHM;
	int enable_benchmark;
	int enable_PD;

	//scene settings
	object *objects;
	object *dev_objects;
	int objects_num;

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
	TYPE*	dev_inertia_X;
	TYPE*	dev_rest_X;
	TYPE*	dev_E;
	TYPE*	dev_V;
	TYPE*	dev_prev_prev_X;
	TYPE*	dev_prev_X;		// previous X	(for Chebyshev)
	int*	dev_fixed;
	TYPE*   dev_fixed_X;
	int*	dev_more_fixed;
	int* dev_collision_fixed;
	int** dev_update_flag;
	TYPE*	dev_init_B;		// Initialized momentum condition in B
	TYPE*	dev_target_X;

	//for conjugated gradient
	TYPE **dev_temp_X;
	TYPE **dev_deltaX;
	TYPE **dev_B;
	TYPE **dev_R;
	TYPE **dev_P;
	TYPE **dev_AP;

	TYPE *dev_Energy;

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
	int **dev_UtAUs_update_offset;
	int **dev_UtAUs_mirror_offset;
	int **dev_UtAUs_half_offset;
	TYPE **dev_UtAUs_precomputed_4x4;
	int **dev_UtAUs_D_offset;
	int **dev_UtAUs_L_offset;

	int **dev_UtAUs_U_rowInd;
	int **dev_UtAUs_U_rowPtr;
	int **dev_UtAUs_U_colInd;
	TYPE **dev_UtAUs_U_Val;

	int **dev_UtAUs_GS_U_rowInd;
	int **dev_UtAUs_GS_U_rowPtr;
	int **dev_UtAUs_GS_U_colInd;

	int **dev_UtAUs_D_rowInd;
	int **dev_UtAUs_D_rowPtr;
	int **dev_UtAUs_D_colInd;
	TYPE **dev_UtAUs_D_Val;

	int **dev_UtAUs_L_rowInd;
	int **dev_UtAUs_L_rowPtr;
	int **dev_UtAUs_L_colInd;
	TYPE **dev_UtAUs_L_Val;

	int **dev_UtAUs_GS_L_rowInd;
	int **dev_UtAUs_GS_L_rowPtr;
	int **dev_UtAUs_GS_L_colInd;

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

	TYPE* dev_MF_Val;
	int* dev_MF_rowPtr;
	int* dev_MF_rowInd;
	int* dev_MF_colInd;
	TYPE *dev_MF_temp_Val;

	int* dev_MF_GS_U_rowInd;
	int* dev_MF_GS_U_rowPtr;
	int* dev_MF_GS_U_colInd;
	TYPE* dev_MF_U_Val;

	int* dev_MF_U_rowInd;
	int* dev_MF_U_rowPtr;
	int* dev_MF_U_colInd;

	int* dev_MF_D_rowInd;
	int* dev_MF_D_rowPtr;
	int* dev_MF_D_colInd;
	TYPE* dev_MF_D_Val;
	int* dev_MF_D_offset;

	int* dev_MF_L_rowInd;
	int* dev_MF_L_rowPtr;
	int* dev_MF_L_colInd;

	int* dev_MF_GS_L_rowInd;
	int* dev_MF_GS_L_rowPtr;
	int* dev_MF_GS_L_colInd;
	TYPE* dev_MF_L_Val;

	int *dev_MF_PD_rowInd;
	int *dev_MF_PD_rowPtr;
	int *dev_MF_PD_colInd;
	TYPE *dev_MF_PD_Val;

	int *dev_MF_PD_P;
	int *dev_MF_PD_Q;

	int *dev_where_to_update;
	int *dev_handles_num;

	int** dev_vertex2index;
	int** dev_index2vertex;

	int* dev_colors;

	int*	dev_all_VV;
	TYPE*	dev_all_VL;
	int*	dev_all_vv_num;

	int *dev_spring_v1;
	int *dev_spring_v2;
	TYPE *dev_spring_len;
	int *dev_spring_update_offset;

	TYPE *dev_UtAUs_chol;
	TYPE *dev_chol_fixed;
	int chol_bufferSize;
	TYPE *dev_chol_buffer;
	int *dev_chol_info;

	TYPE*	error;
	TYPE*	dev_error;

	cublasHandle_t cublasHandle;
	cusparseHandle_t cusparseHandle;
	cusolverSpHandle_t cusolverSpHandle;
	cusolverDnHandle_t cusolverDnHandle;
	cusparseMatDescr_t descr;
	cusparseMatDescr_t descrU;
	cusparseMatDescr_t descrL;
	cusparseSolveAnalysisInfo_t infoU;
	cusparseSolveAnalysisInfo_t infoL;

	cudaStream_t stream1, stream2, stream3;

	CUDA_PROJECTIVE_MESH()
	{
		fixed = new int[max_number];
		fixed_X = new TYPE[max_number * 3];
		target_X = new TYPE[max_number * 3];

		V = new TYPE[max_number * 3];

		memset(V, 0, sizeof(TYPE)*max_number * 3);
		memset(fixed, 0, sizeof(int)*max_number);

		all_VV = new int[max_number * 4];
		all_VL = new TYPE[max_number * 4];
		all_vv_num = new int[max_number];

		cost_ptr = 0;
		fps = 0;
		rho = 0.9992;
		control_mag = 400;
		damping = 1.0;

		spring_k = 3000000;	//5000000

		dev_X = 0;
		dev_V = 0;
		dev_init_B = 0;
		dev_prev_prev_X = 0;
		dev_prev_X = 0;


		dev_fixed = 0;
		dev_more_fixed = 0;

		dev_all_VV = 0;
		dev_all_VL = 0;
		dev_all_vv_num = 0;
	}

	~CUDA_PROJECTIVE_MESH()
	{
		//if (old_X)			delete[] old_X;
		//if (V)				delete[] V;
		//if (fixed)			delete[] fixed;
		//if (MD)				delete[] MD;
		//if (MF_rowInd)	delete[] MF_rowInd;
		//if (MF_colInd) delete[] MF_colInd;
		//if (MF_Val) delete[] MF_Val;
		//if (TQ)				delete[] TQ;
		//if (Tet_Temp)		delete[] Tet_Temp;
		//if (VTT)				delete[] VTT;
		//if (vtt_num)			delete[] vtt_num;
		//if (error)			delete[] error;

		////GPU Data
		//if (dev_X)			cudaFree(dev_X);
		//if (dev_E)			cudaFree(dev_E);
		//if (dev_V)			cudaFree(dev_V);
		//if (dev_next_X)		cudaFree(dev_next_X);
		//if (dev_prev_X)		cudaFree(dev_prev_X);
		//if (dev_fixed)		cudaFree(dev_fixed);
		//if (dev_more_fixed)	cudaFree(dev_more_fixed);
		//if (dev_init_B)		cudaFree(dev_init_B);

		//if (dev_Dm)			cudaFree(dev_Dm);
		//if (dev_inv_Dm)		cudaFree(dev_inv_Dm);
		//if (dev_Vol)			cudaFree(dev_Vol);
		//if (dev_Tet)			cudaFree(dev_Tet);
		//if (dev_TQ)			cudaFree(dev_TQ);
		//if (dev_Tet_Temp)	cudaFree(dev_Tet_Temp);
		//if (dev_VC)			cudaFree(dev_VC);
		//if (dev_MD)			cudaFree(dev_MD);
		//if (dev_MF)			cudaFree(dev_MF);
		//if (dev_VTT)			cudaFree(dev_VTT);
		//if (dev_vtt_num)		cudaFree(dev_vtt_num);

		//if (dev_error)		cudaFree(dev_error);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	//  Initialize functions
	///////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(TYPE t)
	{
		printf("\n");
		this->Build_Connectivity();
		printf(".");
		//Initialize_MD();
		InitializeContext();
		printf(".");
		//Coloring();
		//Initialize_MF(t);
		Build_VV();
		printf(".");

		dims = (int*)malloc(sizeof(int)*(layer + 1));
		dims[layer] = 3 * number;
		for (int l = 0; l < layer; l++)
			dims[l] = 12 * handles_num[l];

		memset(fixed_X, 0, sizeof(TYPE) * 3 * number);
		for (int v = 0; v < number; v++)
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
		compute_U_Matrix();
#ifdef MATRIX_OUTPUT
		char str[256];
		for (int l = 0; l < layer; l++)
		{
			sprintf(str, "benchmark\\U%d.txt", l);
			outputCsr(str, dims[l + 1], dims[l], Us_rowPtr[l], Us_colInd[l], Us_Val[l], Us_nnz[l]);
		}
#endif
		printf(".");
		compute_A_Matrix(t);
		printf(".");
		//outputBoo<TYPE>("benchmark\\UtAU.txt", handles_num[0], handles_num[0], UtAUs_rowInd[0], UtAUs_colInd[0], UtAUs_Val[0], UtAUs_nnz[0], 12);
		CPUCoo2GPUCsr();
		printf(".");
		//prepare_Diag_Part();
		printf(".");
		printf("\n");
	}

	void Build_VV()
	{
		std::pair<int, int> *all_E = new std::pair<int, int>[e_number * 4];
		int  all_e_number = 0;
		for (int e = 0; e < e_number; e++)
		{
			//Add original edges
			all_E[all_e_number] = std::make_pair(E[e * 2 + 0], E[e * 2 + 1]);
			all_e_number++;
			all_E[all_e_number] = std::make_pair(E[e * 2 + 1], E[e * 2 + 0]);
			all_e_number++;

			int t0 = ET[e * 2 + 0];
			int t1 = ET[e * 2 + 1];
			if (t0 == -1 || t1 == -1)	continue;
			int v2 = T[t0 * 3 + 0] + T[t0 * 3 + 1] + T[t0 * 3 + 2] - E[e * 2 + 0] - E[e * 2 + 1];
			int v3 = T[t1 * 3 + 0] + T[t1 * 3 + 1] + T[t1 * 3 + 2] - E[e * 2 + 0] - E[e * 2 + 1];
			all_E[all_e_number] = std::make_pair(v2, v3);
			all_e_number++;
			all_E[all_e_number] = std::make_pair(v3, v2);
			all_e_number++;
		}
		sort(all_E, all_E + all_e_number);

		int i1 = 0, i2 = 0;
		int pre_v = -1, pre_to = -1;
		for (int i = 0; i < all_e_number; i++)
		{
			if (all_E[i].first == pre_v && all_E[i].second == pre_to) continue;
			if (all_E[i].first != pre_v)
			{
				all_vv_num[i1] = i2;
				i1++;
			}
			all_VV[i2] = all_E[i].second;
			all_VL[i2] = dist(all_E[i].first, all_E[i].second);

			i2++;

			pre_v = all_E[i].first;
			pre_to = all_E[i].second;
		}
		all_vv_num[number] = i2;
		delete[] all_E;

		printf("N: %d, M:%d\n", number, all_e_number);
	}

	void build_constraints(std::map<std::pair<int, int>, int>& block_offset)
	{
		// ultimately rubbish code
		spring_v1 = new int[e_number * 4];
		spring_v2 = new int[e_number * 4];
		spring_len = new TYPE[e_number * 4];
		spring_update_offset = new int[e_number * 16];

		spring_num = 0;
		for (int e = 0; e < e_number; e++)
		{
			int v0 = E[e * 2 + 0];
			int v1 = E[e * 2 + 1];

			spring_v1[spring_num] = v0;
			spring_v2[spring_num] = v1;
			spring_len[spring_num] = dist(v0, v1);

#ifndef NO_ORDER
			v0 = vertex2index[layer][v0];
			v1 = vertex2index[layer][v1];
#endif

			spring_update_offset[spring_num * 4 + 0] = block_offset[std::make_pair(v0, v0)];
			spring_update_offset[spring_num * 4 + 1] = block_offset[std::make_pair(v0, v1)];
			spring_update_offset[spring_num * 4 + 2] = block_offset[std::make_pair(v1, v0)];
			spring_update_offset[spring_num * 4 + 3] = block_offset[std::make_pair(v1, v1)];
			spring_num++;

			//spring_v1[spring_num] = v1;
			//spring_v2[spring_num] = v0;
			//spring_len[spring_num] = dist(v1, v0);
			//spring_update_offset[spring_num * 4 + 0] = block_offset[std::make_pair(v1, v1)];
			//spring_update_offset[spring_num * 4 + 1] = block_offset[std::make_pair(v1, v0)];
			//spring_update_offset[spring_num * 4 + 2] = block_offset[std::make_pair(v0, v1)];
			//spring_update_offset[spring_num * 4 + 3] = block_offset[std::make_pair(v0, v0)];
			//spring_num++;

			int t0 = ET[e * 2 + 0];
			int t1 = ET[e * 2 + 1];
			if (t0 == -1 || t1 == -1)	continue;
			int v2 = T[t0 * 3 + 0] + T[t0 * 3 + 1] + T[t0 * 3 + 2] - E[e * 2 + 0] - E[e * 2 + 1];
			int v3 = T[t1 * 3 + 0] + T[t1 * 3 + 1] + T[t1 * 3 + 2] - E[e * 2 + 0] - E[e * 2 + 1];

			spring_v1[spring_num] = v2;
			spring_v2[spring_num] = v3;
			spring_len[spring_num] = dist(v2, v3);

#ifndef NO_ORDER
			v2 = vertex2index[layer][v2];
			v3 = vertex2index[layer][v3];
#endif

			spring_update_offset[spring_num * 4 + 0] = block_offset[std::make_pair(v2, v2)];
			spring_update_offset[spring_num * 4 + 1] = block_offset[std::make_pair(v2, v3)];
			spring_update_offset[spring_num * 4 + 2] = block_offset[std::make_pair(v3, v2)];
			spring_update_offset[spring_num * 4 + 3] = block_offset[std::make_pair(v3, v3)];
			spring_num++;

			//spring_v1[spring_num] = v3;
			//spring_v2[spring_num] = v2;
			//spring_len[spring_num] = dist(v3, v2);
			//spring_update_offset[spring_num * 4 + 0] = block_offset[std::make_pair(v3, v3)];
			//spring_update_offset[spring_num * 4 + 1] = block_offset[std::make_pair(v3, v2)];
			//spring_update_offset[spring_num * 4 + 2] = block_offset[std::make_pair(v2, v3)];
			//spring_update_offset[spring_num * 4 + 3] = block_offset[std::make_pair(v2, v2)];
			//spring_num++;
		}

		cudaError_t cudaErr;
		cudaErr = cudaMalloc(&dev_spring_v1, sizeof(int)*spring_num);
		cudaErr = cudaMalloc(&dev_spring_v2, sizeof(int)*spring_num);
		cudaErr = cudaMalloc(&dev_spring_len, sizeof(TYPE)*spring_num);
		cudaErr = cudaMalloc(&dev_spring_update_offset, sizeof(int)*spring_num * 4);

		cudaErr = cudaMemcpy(dev_spring_v1, spring_v1, sizeof(int)*spring_num, cudaMemcpyHostToDevice);
		cudaErr = cudaMemcpy(dev_spring_v2, spring_v2, sizeof(int)*spring_num, cudaMemcpyHostToDevice);
		cudaErr = cudaMemcpy(dev_spring_len, spring_len, sizeof(TYPE)*spring_num, cudaMemcpyHostToDevice);
		cudaErr = cudaMemcpy(dev_spring_update_offset, spring_update_offset, sizeof(int)*spring_num * 4, cudaMemcpyHostToDevice);
	}


	void InitializeContext()
	{
		cublasCreate(&cublasHandle);
		cusparseCreate(&cusparseHandle);
		cusolverSpCreate(&cusolverSpHandle);
		cusolverDnCreate(&cusolverDnHandle);

		cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

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
		err = cudaMalloc((void**)&dev_Energy, sizeof(TYPE));
		err = cudaMalloc((void**)&dev_X, sizeof(TYPE) * 3 * number);
		err = cudaMalloc((void**)&dev_old_X, sizeof(TYPE) * 3 * number);
		err = cudaMalloc((void**)&dev_inertia_X, sizeof(TYPE) * 3 * number);
		err = cudaMalloc(&dev_rest_X, sizeof(float) * 3 * number);
		err = cudaMalloc((void**)&dev_E, sizeof(int) * 3 * number);
		err = cudaMalloc((void**)&dev_V, sizeof(TYPE) * 3 * number);
		err = cudaMalloc((void**)&dev_prev_prev_X, sizeof(TYPE) * 3 * number);
		err = cudaMalloc((void**)&dev_prev_X, sizeof(TYPE) * 3 * number);
		err = cudaMalloc((void**)&dev_fixed, sizeof(int)*  number);
		err = cudaMalloc((void**)&dev_more_fixed, sizeof(int)*  number);
		err = cudaMalloc((void**)&dev_collision_fixed, sizeof(int)*number);
		err = cudaMalloc((void**)&dev_fixed_X, sizeof(TYPE) * 3 * number);
		err = cudaMalloc((void**)&dev_init_B, sizeof(TYPE) * 3 * number);
		err = cudaMalloc(&dev_target_X, sizeof(TYPE) * 3 * number);


		dev_temp_X = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_deltaX = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_B = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_R = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_P = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));
		dev_AP = (TYPE**)malloc(sizeof(TYPE*)*(layer + 1));

		for (int l = 0; l <= layer; l++)
		{
			err = cudaMalloc((void**)&dev_temp_X[l], sizeof(TYPE) * dims[l]);
			err = cudaMalloc((void**)&dev_deltaX[l], sizeof(TYPE) * dims[l]);
			err = cudaMalloc(&dev_B[l], sizeof(TYPE)*dims[l]);
			err = cudaMalloc((void**)&dev_R[l], sizeof(TYPE) * dims[l]);
			err = cudaMalloc((void**)&dev_P[l], sizeof(TYPE) * dims[l]);
			err = cudaMalloc((void**)&dev_AP[l], sizeof(TYPE) * dims[l]);
		}

		//err = cudaMalloc((void**)&dev_Dm, sizeof(TYPE)*tet_number * 9);
		//err = cudaMalloc((void**)&dev_inv_Dm, sizeof(TYPE)*tet_number * 9);
		//err = cudaMalloc((void**)&dev_Vol, sizeof(TYPE)*tet_number);
		//err = cudaMalloc((void**)&dev_Tet, sizeof(TYPE)*tet_number * 4);
		//err = cudaMalloc((void**)&dev_TQ, sizeof(TYPE)*tet_number * 4);
		//err = cudaMalloc((void**)&dev_Tet_Temp, sizeof(TYPE)*tet_number * 12);

		err = cudaMalloc(&dev_all_VV, sizeof(int)*all_vv_num[number]);
		err = cudaMalloc(&dev_all_VL, sizeof(TYPE) * all_vv_num[number]);
		err = cudaMalloc(&dev_all_vv_num, sizeof(int) *(number + 1));

		err = cudaMalloc((void**)&dev_VC, sizeof(TYPE)*number);
		err = cudaMalloc((void**)&dev_M, sizeof(TYPE)*number);
		// warning
		err = cudaMalloc((void**)&dev_MD, sizeof(TYPE)*number);
		// end 

		err = cudaMemcpy(dev_all_VV, all_VV, sizeof(int)*all_vv_num[number], cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_all_VL, all_VL, sizeof(TYPE)*all_vv_num[number], cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_all_vv_num, all_vv_num, sizeof(int)*(number + 1), cudaMemcpyHostToDevice);

		err = cudaMalloc((void**)&dev_colors, sizeof(int)*number);

		err = cudaMalloc((void**)&dev_error, sizeof(TYPE) * 3 * number);

		//Copy data into CUDA memory
		err = cudaMemcpy(dev_X, X, sizeof(TYPE) * 3 * number, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_rest_X, X, sizeof(TYPE) * 3 * number, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_V, V, sizeof(TYPE) * 3 * number, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_prev_X, X, sizeof(TYPE) * 3 * number, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_prev_prev_X, X, sizeof(TYPE) * 3 * number, cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_fixed, fixed, sizeof(int)*number, cudaMemcpyHostToDevice);
		err = cudaMemset(dev_more_fixed, 0, sizeof(int)*number);
		err = cudaMemcpy(dev_fixed_X, fixed_X, sizeof(TYPE) * 3 * number, cudaMemcpyHostToDevice);

		//err = cudaMemcpy(dev_Dm, Dm, sizeof(int)*tet_number * 9, cudaMemcpyHostToDevice);
		//err = cudaMemcpy(dev_inv_Dm, inv_Dm, sizeof(int)*tet_number * 9, cudaMemcpyHostToDevice);
		//err = cudaMemcpy(dev_Vol, Vol, sizeof(int)*tet_number, cudaMemcpyHostToDevice);
		//err = cudaMemcpy(dev_Tet, Tet, sizeof(int)*tet_number * 4, cudaMemcpyHostToDevice);

		err = cudaMemcpy(dev_M, M, sizeof(TYPE)*number, cudaMemcpyHostToDevice);
		//err = cudaMemcpy(dev_MD, MD, sizeof(TYPE)*number, cudaMemcpyHostToDevice);


		//err = cudaMemcpy(dev_VTT, VTT, sizeof(int)*tet_number * 4, cudaMemcpyHostToDevice);
		//err = cudaMemcpy(dev_TQ, TQ, sizeof(TYPE)*tet_number * 4, cudaMemcpyHostToDevice);
		//err = cudaMemcpy(dev_vtt_num, vtt_num, sizeof(int)*(number + 1), cudaMemcpyHostToDevice);
	}

	void build_Graph()
	{
		std::pair<int, int>* _E = new std::pair<int, int>[e_number * 4];
		int edges_num = 0;
		for (int e = 0; e < e_number; e++)
		{
			_E[edges_num++] = std::make_pair(E[e * 2 + 0], E[e * 2 + 1]);
			_E[edges_num++] = std::make_pair(E[e * 2 + 1], E[e * 2 + 0]);

			int t0 = ET[e * 2 + 0];
			int t1 = ET[e * 2 + 1];
			if (t0 == -1 || t1 == -1)	continue;
			int v2 = T[t0 * 3 + 0] + T[t0 * 3 + 1] + T[t0 * 3 + 2] - E[e * 2 + 0] - E[e * 2 + 1];
			int v3 = T[t1 * 3 + 0] + T[t1 * 3 + 1] + T[t1 * 3 + 2] - E[e * 2 + 0] - E[e * 2 + 1];
			_E[edges_num++] = std::make_pair(v2, v3);
			_E[edges_num++] = std::make_pair(v3, v2);
		}
		sort(_E, _E + edges_num);

		e_num = new int[number + 1];
		e_to = new int[edges_num];
		e_dist = new TYPE[edges_num];
		int t1 = 0, t2 = 0;
		int pre_i = -1, pre_to = -1;
		for (int i = 0; i < edges_num; i++)
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
		std::vector<int> handles;
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
			for (int j = 0; j < handles_num[layer - 1]; j++)
				if (acc_nd[j][i] < mini_dist)
				{
					mini_dist = acc_nd[j][i];
					arg_min = j;
				}
			handle[layer][i] = arg_min;
		}
	}

	void Coloring(int* t_e_num, int* t_e_to, int* t_coloring, int t_number, int t_init_color_num)
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
				if (stuck == 16)
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

		vertex2index = (int**)malloc(sizeof(int*)*(layer + 1));
		index2vertex = (int**)malloc(sizeof(int*)*(layer + 1));
		color_vertices_num = (int**)malloc(sizeof(int*)*(layer + 1));
		colors_num = (int*)malloc(sizeof(int)*(layer + 1));
		dev_vertex2index = (int**)malloc(sizeof(int*)*(layer + 1));
		dev_index2vertex = (int**)malloc(sizeof(int*)*(layer + 1));

		int *coloring = new int[number];
		std::pair<int, int> *temp = new std::pair<int, int>[e_num[number] * 2];
		int *temp_e_num = new int[number + 1], *temp_e_to = new int[e_num[number]];

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
			// coloring
			vertex2index[l] = new int[handles_num[l]];
			index2vertex[l] = new int[handles_num[l]];
			color_vertices_num[l] = new int[handles_num[l] + 1];
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

			//printf("%d:\n", l);
			//for (int i = 0; i < colors_num[l]; i++)
			//	printf("%d ", color_vertices_num[l][i]);
			//printf("\n");

			cudaErr = cudaMalloc(&dev_vertex2index[l], sizeof(int)*handles_num[l]);
			cudaErr = cudaMalloc(&dev_index2vertex[l], sizeof(int)*handles_num[l]);
			cudaErr = cudaMemcpy(dev_vertex2index[l], vertex2index[l], sizeof(int)*handles_num[l], cudaMemcpyHostToDevice);
			cudaErr = cudaMemcpy(dev_index2vertex[l], index2vertex[l], sizeof(int)*handles_num[l], cudaMemcpyHostToDevice);
		}
	}

	void outputA(TYPE t)
	{
		FILE *f = fopen("benchmark\\A.txt", "w");
		fprintf(f, "%d\n%d\n%d\n", 3*number, 3*number, 9*MF_nnz);
		for (int n = 0; n < MF_nnz; n++)
		{
			
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
		sprintf(file_name, "benchmark\\U%d.txt", layer - 1);
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
		Us_nnz = (int*)malloc(sizeof(int)*layer);
		for (int l = 0; l < layer - 1; l++)
		{
			Us_rowPtr[l] = new int[dims[l + 1] + 1];
			Us_colInd[l] = new int[dims[l + 1]];
			Us_Val[l] = new TYPE[dims[l + 1]];
			Us_nnz[l] = dims[l + 1];
			int i1 = 0;
			for (int r = 0; r < handles_num[l + 1]; r++)
			{
#ifdef NO_ORDER
				int c = handle[l + 1][r];
#else
				int c = vertex2index[l][handle[l + 1][index2vertex[l + 1][r]]];
#endif
				for (int t = 0; t < 12; t++)
				{
					Us_rowPtr[l][12 * r + t] = i1;
					Us_colInd[l][i1] = 12 * c + t;
					Us_Val[l][i1] = 1.0;
					i1++;
				}
			}
			Us_rowPtr[l][dims[l + 1]] = i1;
		}

		Us_nnz[layer - 1] = 4 * dims[layer];
		Us_rowPtr[layer - 1] = new int[dims[layer] + 1];
		Us_colInd[layer - 1] = new int[Us_nnz[layer - 1]];
		Us_Val[layer - 1] = new TYPE[Us_nnz[layer - 1]];
		
		int i1 = 0;
		for (int r = 0; r < number; r++)
		{
#ifdef NO_ORDER
			int v = r;
			int c = handle[layer][r];
#else
			int v = index2vertex[layer][r];
			int c = vertex2index[layer - 1][handle[layer][v]];
#endif
			for (int d = 0; d < 3; d++)
			{
				Us_rowPtr[layer - 1][r * 3 + d] = i1;
				for (int t = 0; t < 4; t++)
				{
					Us_colInd[layer - 1][i1] = 12 * c + 4 * d + t;
					Us_Val[layer - 1][i1] = (t == 3) ? 1 : X[3 * v + t];
					i1++;
				}
			}
		}
		Us_rowPtr[layer - 1][3 * number] = i1;

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
		std::map<std::pair<int, int>, TYPE*> cooMap;
		for (int i = 0; i < number; i++)
			for (int j = all_vv_num[i]; j < all_vv_num[i + 1]; j++)
			{
				int v[2];
				v[0] = i;
				v[1] = all_VV[j];

#ifndef NO_ORDER
				v[0] = vertex2index[layer][v[0]];
				v[1] = vertex2index[layer][v[1]];
#endif
				for (int p = 0; p < 2; p++)
					for (int q = 0; q < 2; q++)
					{
						std::pair<int, int> key = std::make_pair(v[p], v[q]);
						if (cooMap.find(key) == cooMap.end())
						{
							cooMap[key] = new TYPE[9];
							memset(cooMap[key], 0, sizeof(TYPE) * 9);
						}
						TYPE *ref = cooMap[key];
						if (p == q)
						{
							ref[0] += spring_k/2;
							ref[4] += spring_k/2;
							ref[8] += spring_k/2;
						}
						else
						{
							ref[0] -= spring_k/2;
							ref[4] -= spring_k/2;
							ref[8] -= spring_k/2;
						}

					}
			}
		for (int i = 0; i < number; i++)
		{
#ifdef NO_ORDER
			int v = i;
#else
			int v = index2vertex[layer][i];
#endif
			TYPE diag = M[v] * inv_t*inv_t;
			if (fixed[v])
				diag += control_mag;
			TYPE *ref = cooMap[std::make_pair(i, i)];
			ref[0] += diag;
			ref[4] += diag;
			ref[8] += diag;
		}
		{
			MF_nnz = cooMap.size();
			MF_Val = new TYPE[MF_nnz * 9];
			MF_colInd = new int[MF_nnz];
			MF_rowInd = new int[MF_nnz];
			MF_D_offset = new int[number];

			MF_PD_nnz = MF_nnz * 3;
			MF_PD_Val = new TYPE[MF_PD_nnz];
			MF_PD_rowInd = new int[MF_PD_nnz];
			MF_PD_colInd = new int[MF_PD_nnz];


			MF_L_nnz = MF_U_nnz = (MF_nnz - number) >> 1;
			MF_D_nnz = number;

			int off_l = 0, off_d = MF_L_nnz, off_u = MF_L_nnz + number;

			MF_GS_L_Ptr = new int[colors_num[layer] + 1];
			MF_GS_L_rowInd = new int[MF_L_nnz];
			MF_GS_L_colInd = new int[MF_L_nnz];
			MF_L_Val = MF_Val + off_l * 9;

			MF_L_rowInd = MF_rowInd + off_l;
			MF_L_colInd = MF_colInd + off_l;

			MF_D_rowInd = MF_rowInd + off_d;
			MF_D_colInd = MF_colInd + off_d;;
			MF_D_Val = MF_Val + off_d * 9;

			MF_U_rowInd = MF_rowInd + off_u;
			MF_U_colInd = MF_colInd + off_u;

			MF_GS_U_Ptr = new int[colors_num[layer] + 1];
			MF_GS_U_rowInd = new int[MF_U_nnz];
			MF_GS_U_colInd = new int[MF_U_nnz];
			MF_U_Val = MF_Val + off_u * 9;

			int i_l = 0, i_d = 0, i_u = 0;
			int p_base_l = -1, p_base_u = -1;
			int now_base_l = 0, next_base_l = color_vertices_num[layer][1];
			int r_base_u = -1, c_base_u = 0;
			int i_f = 0;
			int i_pd = 0;

			std::map<std::pair<int, int>, int> block_offset;
			for (auto iter : cooMap)
			{
				int r = iter.first.first, c = iter.first.second;
				TYPE *v = iter.second;

				//PD
				for(int i=0;i<3;i++)
					for(int j=0;j<3;j++)
						if (i == j)
						{
							MF_PD_rowInd[i_pd] = r * 3 + i;
							MF_PD_colInd[i_pd] = c * 3 + j;
							MF_PD_Val[i_pd] = v[i * 3 + j];
							i_pd++;
						}

				if (stored_as_LDU[layer])
				{
					if (r == c)
					{
						MF_D_rowInd[i_d] = r;
						MF_D_colInd[i_d] = c;
						for (int i = 0; i < 9; i++)
							MF_D_Val[i_d * 9 + i] = v[i];
						block_offset[iter.first] = off_d + i_d;
						i_d++;
					}

					if (r < c)
					{
						while (r >= c_base_u)
						{
							p_base_u++;
							MF_GS_U_Ptr[p_base_u] = i_u;
							r_base_u = color_vertices_num[layer][p_base_u];
							c_base_u = color_vertices_num[layer][p_base_u + 1];
						}
						MF_U_rowInd[i_u] = r;
						MF_U_colInd[i_u] = c;
						MF_GS_U_rowInd[i_u] = r - r_base_u;
						MF_GS_U_colInd[i_u] = c - c_base_u;
						for (int i = 0; i < 9; i++)
							MF_U_Val[i_u * 9 + i] = v[i];
						block_offset[iter.first] = off_u + i_u;
						i_u++;
					}

					if (r > c)
					{
						while (r >= next_base_l)
						{
							p_base_l++;
							MF_GS_L_Ptr[p_base_l] = i_l;
							now_base_l = color_vertices_num[layer][p_base_l + 1];
							next_base_l = color_vertices_num[layer][p_base_l + 2];
						}
						MF_L_rowInd[i_l] = r;
						MF_L_colInd[i_l] = c;
						MF_GS_L_rowInd[i_l] = r - now_base_l;
						MF_GS_L_colInd[i_l] = c;
						for (int i = 0; i < 9; i++)
							MF_L_Val[i_l * 9 + i] = v[i];
						block_offset[iter.first] = off_l + i_l;
						i_l++;
					}
				}
				else
				{
					MF_rowInd[i_f] = r;
					MF_colInd[i_f] = c;
					for (int i = 0; i < 9; i++)
						MF_Val[i_f * 9 + i] = v[i];
					block_offset[iter.first] = i_f;
					i_f++;
				}
			}
			sortCoo(MF_PD_rowInd, MF_PD_colInd, MF_PD_Val, MF_PD_nnz);
			if (stored_as_LDU[layer])
			{
				for (; p_base_u < colors_num[layer] - 1;)
					MF_GS_U_Ptr[++p_base_u] = i_u;
				for (; p_base_l < colors_num[layer] - 1;)
					MF_GS_L_Ptr[++p_base_l] = i_l;
			}
			for (int i = 0; i < number; i++)
				MF_D_offset[i] = block_offset[std::make_pair(i, i)];
			build_constraints(block_offset);
		}


		//finest layer end
		if (!layer)
			return;

		UtAUs_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_colInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		UtAUs_nnz = (int*)malloc(sizeof(int)*layer);
		
		UtAUs_L_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_L_colInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_L_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		UtAUs_L_nnz = (int*)malloc(sizeof(int)*layer);

		UtAUs_D_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_D_colInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_D_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		UtAUs_D_nnz = (int*)malloc(sizeof(int)*layer);

		UtAUs_U_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_U_colInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_U_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		UtAUs_U_nnz = (int*)malloc(sizeof(int)*layer);

		UtAUs_GS_U_Ptr = (int**)malloc(sizeof(int*)*layer);
		UtAUs_GS_U_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_GS_U_colInd = (int**)malloc(sizeof(int*)*layer);
		
		UtAUs_GS_L_Ptr = (int**)malloc(sizeof(int*)*layer);
		UtAUs_GS_L_rowInd = (int**)malloc(sizeof(int*)*layer);
		UtAUs_GS_L_colInd = (int**)malloc(sizeof(int*)*layer);

		UtAUs_update_offset = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_update_offset = (int**)malloc(sizeof(int*)*layer);

		UtAUs_mirror_offset = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_mirror_offset = (int**)malloc(sizeof(int*)*layer);
		
		UtAUs_half_num = (int*)malloc(sizeof(int)*layer);
		UtAUs_half_offset = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_half_offset = (int**)malloc(sizeof(int*)*layer);

		UtAUs_precomputed_4x4 = (TYPE**)malloc(sizeof(TYPE*)*layer);
		dev_UtAUs_precomputed_4x4 = (TYPE**)malloc(sizeof(TYPE*)*layer);

		UtAUs_D_offset = (int**)malloc(sizeof(int*)*layer);

		UtAUs_L_offset = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_L_offset = (int**)malloc(sizeof(int*)*layer);

		//initial coo44Map from cooMap
		std::map<std::pair<int, int>, TYPE*> coo44Map;
		for (auto iter : cooMap)
		{
			int r = iter.first.first, c = iter.first.second;
			TYPE *v = iter.second;
#ifndef NO_ORDER
			int i = index2vertex[layer][r], j = index2vertex[layer][c];
			int rd = vertex2index[layer - 1][handle[layer][i]];
			int cd = vertex2index[layer - 1][handle[layer][j]];
#else
			int i = r, j = c;
			int rd = handle[layer][i];
			int cd = handle[layer][j];
#endif
			std::pair<int, int> key = std::make_pair(rd, cd);
			if (coo44Map.find(key) == coo44Map.end())
			{
				coo44Map[key] = new TYPE[144];
				memset(coo44Map[key], 0, sizeof(TYPE) * 144);
			}
			for (int vi = 0; vi < 3; vi++)
				for (int di = 0; di < 4; di++)
					for (int vj = 0; vj < 3; vj++)
						for (int dj = 0; dj < 4; dj++)
						{
							TYPE v1 = (di == 3) ? 1 : X[3 * i + di];
							TYPE v2 = (dj == 3) ? 1 : X[3 * j + dj];
							coo44Map[key][(vi * 4 + di) * 12 + (vj * 4 + dj)] += v[3 * vi + vj] * v1*v2;
						}
		}

		//the rest case
		for (int l = layer - 1; l >= 0; l--)
		{
			UtAUs_nnz[l] = coo44Map.size();
			UtAUs_L_nnz[l] = (UtAUs_nnz[l] - handles_num[l]) >> 1;
			UtAUs_D_nnz[l] = handles_num[l];
			UtAUs_U_nnz[l] = (UtAUs_nnz[l] - handles_num[l]) >> 1;

			int off_l = 0, off_d = UtAUs_L_nnz[l], off_u = UtAUs_L_nnz[l] + handles_num[l];

			UtAUs_rowInd[l] = new int[UtAUs_nnz[l]];
			UtAUs_colInd[l] = new int[UtAUs_nnz[l]];
			UtAUs_Val[l] = new TYPE[UtAUs_nnz[l] * 144];
		
			UtAUs_L_rowInd[l] = UtAUs_rowInd[l] + off_l;
			UtAUs_L_colInd[l] = UtAUs_colInd[l] + off_l;
			UtAUs_L_Val[l] = UtAUs_Val[l] + 144 * off_l;

			UtAUs_D_rowInd[l] = UtAUs_rowInd[l] + off_d;
			UtAUs_D_colInd[l] = UtAUs_colInd[l] + off_d;
			UtAUs_D_Val[l] = UtAUs_Val[l] + 144 * off_d;

			UtAUs_U_rowInd[l] = UtAUs_rowInd[l] + off_u;
			UtAUs_U_colInd[l] = UtAUs_colInd[l] + off_u;
			UtAUs_U_Val[l] = UtAUs_Val[l] + 144 * off_u;

			UtAUs_GS_U_Ptr[l] = new int[colors_num[l] + 1];
			UtAUs_GS_U_rowInd[l] = new int[UtAUs_U_nnz[l]];
			UtAUs_GS_U_colInd[l] = new int[UtAUs_U_nnz[l]];

			UtAUs_GS_L_Ptr[l] = new int[colors_num[l] + 1];
			UtAUs_GS_L_rowInd[l] = new int[UtAUs_L_nnz[l]];
			UtAUs_GS_L_colInd[l] = new int[UtAUs_L_nnz[l]];

			UtAUs_D_offset[l] = new int[handles_num[l]];

			int i_l = 0, i_d = 0, i_u = 0;
			int p_base_l = -1,p_base_u=-1;
			int now_base_l = 0, next_base_l = color_vertices_num[l][1];
			int r_base_u = -1, c_base_u = 0;
			int i_f = 0;

			std::map<std::pair<int, int>, int> block_offset;

			for (auto iter : coo44Map)
			{
				int r = iter.first.first, c = iter.first.second;
				TYPE *v = iter.second;

				if (stored_as_LDU[l])
				{
					if (r == c)
					{
						UtAUs_D_rowInd[l][i_d] = r;
						UtAUs_D_colInd[l][i_d] = c;
						for (int i = 0; i < 144; i++)
							UtAUs_D_Val[l][i_d * 144 + i] = v[i];
						block_offset[iter.first] = off_d + i_d;
						i_d++;
					}

					if (r < c)
					{
						while (r >= c_base_u)
						{
							p_base_u++;
							UtAUs_GS_U_Ptr[l][p_base_u] = i_u;
							r_base_u = color_vertices_num[l][p_base_u];
							c_base_u = color_vertices_num[l][p_base_u + 1];
						}
						UtAUs_U_rowInd[l][i_u] = r;
						UtAUs_U_colInd[l][i_u] = c;
						UtAUs_GS_U_rowInd[l][i_u] = r - r_base_u;
						UtAUs_GS_U_colInd[l][i_u] = c - c_base_u;
						for (int i = 0; i < 144; i++)
							UtAUs_U_Val[l][i_u * 144 + i] = v[i];
						block_offset[iter.first] = off_u + i_u;
						i_u++;
					}

					if (r > c)
					{
						while (r >= next_base_l)
						{
							p_base_l++;
							UtAUs_GS_L_Ptr[l][p_base_l] = i_l;
							now_base_l = color_vertices_num[l][p_base_l + 1];
							next_base_l = color_vertices_num[l][p_base_l + 2];
						}
						UtAUs_L_rowInd[l][i_l] = r;
						UtAUs_L_colInd[l][i_l] = c;
						UtAUs_GS_L_rowInd[l][i_l] = r - now_base_l;
						UtAUs_GS_L_colInd[l][i_l] = c;
						for (int i = 0; i < 144; i++)
							UtAUs_L_Val[l][i_l * 144 + i] = v[i];
						block_offset[iter.first] = off_l + i_l;
						i_l++;
					}
				}
				else
				{
					UtAUs_rowInd[l][i_f] = r;
					UtAUs_colInd[l][i_f] = c;
					for (int i = 0; i < 144; i++)
						UtAUs_Val[l][i_f * 144 + i] = v[i];
					if (stored_as_dense[l])
						block_offset[iter.first] = r * handles_num[l] + c;
					else
						block_offset[iter.first] = i_f;
					i_f++;
				}
			}
			if (stored_as_LDU[l])
			{
				for (; p_base_u < colors_num[l] - 1;)
					UtAUs_GS_U_Ptr[l][++p_base_u] = i_u;
				for (; p_base_l < colors_num[l] - 1;)
					UtAUs_GS_L_Ptr[l][++p_base_l] = i_l;
			}

			for (int i = 0; i < handles_num[l]; i++)
				UtAUs_D_offset[l][i] = block_offset[std::make_pair(i, i)];

			//sortCoo(UtAUs_rowInd[l], UtAUs_colInd[l], UtAUs_Val[l], UtAUs_nnz[l]);
			if (l == layer - 1)
			{
				UtAUs_update_offset[l] = new int[MF_nnz];
				UtAUs_half_num[l] = 0;
				UtAUs_half_offset[l] = new int[MF_nnz];
				UtAUs_precomputed_4x4[l] = new TYPE[MF_nnz * 16];
				for (int n = 0; n < MF_nnz; n++)
				{
					int r = MF_rowInd[n], c = MF_colInd[n];
					int i = index2vertex[layer][r], j = index2vertex[layer][c];
					int rd = vertex2index[layer - 1][handle[layer][i]];
					int cd = vertex2index[layer - 1][handle[layer][j]];
					UtAUs_update_offset[l][n] = block_offset[std::make_pair(rd, cd)];
					if (rd >= cd)
						UtAUs_half_offset[l][UtAUs_half_num[l]++] = n;
					for (int di = 0; di < 4; di++)
						for (int dj = 0; dj < 4; dj++)
						{
							TYPE v1 = (di == 3) ? 1 : X[3 * i + di];
							TYPE v2 = (dj == 3) ? 1 : X[3 * j + dj];
							UtAUs_precomputed_4x4[l][16 * n + di * 4 + dj] = v1 * v2;
						}
				}
				
				cudaMalloc(&dev_UtAUs_update_offset[l], sizeof(int)*MF_nnz);
				cudaMemcpy(dev_UtAUs_update_offset[l], UtAUs_update_offset[l], sizeof(int)*MF_nnz, cudaMemcpyHostToDevice);
				cudaMalloc(&dev_UtAUs_half_offset[l], sizeof(int)*UtAUs_half_num[l]);
				cudaMemcpy(dev_UtAUs_half_offset[l], UtAUs_half_offset[l], sizeof(int)*UtAUs_half_num[l], cudaMemcpyHostToDevice);
				cudaMalloc(&dev_UtAUs_precomputed_4x4[l], sizeof(TYPE)*MF_nnz * 16);
				cudaMemcpy(dev_UtAUs_precomputed_4x4[l], UtAUs_precomputed_4x4[l], sizeof(TYPE)*MF_nnz * 16, cudaMemcpyHostToDevice);
			}
			else
			{
				UtAUs_update_offset[l] = new int[UtAUs_nnz[l+1]];
				UtAUs_half_num[l] = 0;
				UtAUs_half_offset[l] = new int[UtAUs_nnz[l + 1]];
				for (int n = 0; n < UtAUs_nnz[l + 1]; n++)
				{
					int r = UtAUs_rowInd[l+1][n], c = UtAUs_colInd[l+1][n];
					int i = index2vertex[l+1][r], j = index2vertex[l+1][c];
					int rd = vertex2index[l][handle[l+1][i]];
					int cd = vertex2index[l][handle[l+1][j]];
					UtAUs_update_offset[l][n] = block_offset[std::make_pair(rd, cd)];
					if (rd >= cd)
						UtAUs_half_offset[l][UtAUs_half_num[l]++] = n;
				}

				cudaMalloc(&dev_UtAUs_update_offset[l], sizeof(int)*UtAUs_nnz[l + 1]);
				cudaMemcpy(dev_UtAUs_update_offset[l], UtAUs_update_offset[l], sizeof(int)*UtAUs_nnz[l + 1], cudaMemcpyHostToDevice);
				cudaMalloc(&dev_UtAUs_half_offset[l], sizeof(int)*UtAUs_half_num[l]);
				cudaMemcpy(dev_UtAUs_half_offset[l], UtAUs_half_offset[l], sizeof(int)*UtAUs_half_num[l], cudaMemcpyHostToDevice);
			}

			UtAUs_mirror_offset[l] = new int[UtAUs_L_nnz[l]];
			UtAUs_L_offset[l] = new int[UtAUs_L_nnz[l]];
			int i_s = 0;
			for (int n = 0; n < UtAUs_nnz[l]; n++)
			{
				int r = UtAUs_rowInd[l][n], c = UtAUs_colInd[l][n];
				if (r > c)
				{
					UtAUs_L_offset[l][i_s] = block_offset[std::make_pair(r, c)];
					UtAUs_mirror_offset[l][i_s] = block_offset[std::make_pair(c, r)];
					i_s++;
				}
			}
			cudaMalloc(&dev_UtAUs_mirror_offset[l], sizeof(int)*UtAUs_L_nnz[l]);
			cudaMemcpy(dev_UtAUs_mirror_offset[l], UtAUs_mirror_offset[l], sizeof(int)*UtAUs_L_nnz[l], cudaMemcpyHostToDevice);
			cudaMalloc(&dev_UtAUs_L_offset[l], sizeof(int)*UtAUs_L_nnz[l]);
			cudaMemcpy(dev_UtAUs_L_offset[l], UtAUs_L_offset[l], sizeof(int)*UtAUs_L_nnz[l], cudaMemcpyHostToDevice);


			if (l)
			{
				std::map<std::pair<int, int>, TYPE*> _coo44Map;
				for (auto iter : coo44Map)
				{
					int r = iter.first.first, c = iter.first.second;
					int i = index2vertex[l][r], j = index2vertex[l][c];
					int rd = vertex2index[l - 1][handle[l][i]];
					int cd = vertex2index[l - 1][handle[l][j]];
					TYPE *v = iter.second;
					std::pair<int, int> key = std::make_pair(rd, cd);
					if (_coo44Map.find(key) == _coo44Map.end())
					{
						_coo44Map[key] = new TYPE[144];
						memset(_coo44Map[key], 0, sizeof(TYPE) * 144);
					}
					for (int i = 0; i < 144; i++)
						_coo44Map[key][i] += v[i];
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
		err = cudaMalloc(&dev_MF_Val, sizeof(TYPE)*MF_nnz * 9);
		err = cudaMalloc(&dev_MF_rowInd, sizeof(int)*MF_nnz);
		err = cudaMalloc(&dev_MF_colInd, sizeof(int)*MF_nnz);
		err = cudaMalloc(&dev_MF_temp_Val, sizeof(TYPE)*MF_nnz * 9 * 16);

		dev_MF_L_Val = dev_MF_Val;
		dev_MF_D_Val = dev_MF_Val + MF_L_nnz * 9;
		dev_MF_U_Val = dev_MF_Val + MF_L_nnz * 9 + number * 9;

		dev_MF_L_rowInd = dev_MF_rowInd;
		dev_MF_D_rowInd = dev_MF_rowInd+MF_L_nnz;
		dev_MF_U_rowInd = dev_MF_rowInd+MF_L_nnz+number;

		dev_MF_L_colInd = dev_MF_colInd;
		dev_MF_D_colInd = dev_MF_colInd + MF_L_nnz;
		dev_MF_U_colInd = dev_MF_colInd + MF_L_nnz + number;

		if (stored_as_LDU[layer])
		{
			//MF_L
			err = cudaMalloc((void**)&dev_MF_GS_L_rowInd, sizeof(int)*MF_L_nnz);
			err = cudaMalloc((void**)&dev_MF_GS_L_rowPtr, sizeof(int)*(number + colors_num[layer] + 1));
			err = cudaMalloc((void**)&dev_MF_GS_L_colInd, sizeof(int)*MF_L_nnz);

			//err = cudaMalloc((void**)&dev_MF_L_rowInd, sizeof(int)*MF_L_nnz);
			err = cudaMalloc((void**)&dev_MF_L_rowPtr, sizeof(int)*(number + 1));
			//err = cudaMalloc((void**)&dev_MF_L_colInd, sizeof(int)*MF_L_nnz);

			err = cudaMemcpy(dev_MF_GS_L_rowInd, MF_GS_L_rowInd, sizeof(int)*MF_L_nnz, cudaMemcpyHostToDevice);
			for (int i = 0; i < colors_num[layer] - 1; i++)
			{
				cuspErr = cusparseXcoo2csr(cusparseHandle, dev_MF_GS_L_rowInd + MF_GS_L_Ptr[i], MF_GS_L_Ptr[i + 1] - MF_GS_L_Ptr[i], \
					color_vertices_num[layer][i + 2] - color_vertices_num[layer][i + 1], \
					dev_MF_GS_L_rowPtr + color_vertices_num[layer][i + 1] - color_vertices_num[layer][1] + i, CUSPARSE_INDEX_BASE_ZERO);
			}
			err = cudaMemcpy(dev_MF_GS_L_colInd, MF_GS_L_colInd, sizeof(int)*MF_L_nnz, cudaMemcpyHostToDevice);

			err = cudaMemcpy(dev_MF_L_rowInd, MF_L_rowInd, sizeof(int)*MF_L_nnz, cudaMemcpyHostToDevice);
			cuspErr = cusparseXcoo2csr(cusparseHandle, dev_MF_L_rowInd, MF_L_nnz, number, dev_MF_L_rowPtr, CUSPARSE_INDEX_BASE_ZERO);
			err = cudaMemcpy(dev_MF_L_colInd, MF_L_colInd, sizeof(int)*MF_L_nnz, cudaMemcpyHostToDevice);
			err = cudaMemcpy(dev_MF_L_Val, MF_L_Val, sizeof(TYPE)*MF_L_nnz * 9, cudaMemcpyHostToDevice);

			//MF_D
			//err = cudaMalloc((void**)&dev_MF_D_rowInd, sizeof(int)*MF_D_nnz);
			err = cudaMalloc((void**)&dev_MF_D_rowPtr, sizeof(int)*(number + 1));
			//err = cudaMalloc((void**)&dev_MF_D_colInd, sizeof(int)*MF_D_nnz);

			err = cudaMemcpy(dev_MF_D_rowInd, MF_D_rowInd, sizeof(int)*MF_D_nnz, cudaMemcpyHostToDevice);
			cuspErr = cusparseXcoo2csr(cusparseHandle, dev_MF_D_rowInd, number, number, dev_MF_D_rowPtr, CUSPARSE_INDEX_BASE_ZERO);
			err = cudaMemcpy(dev_MF_D_colInd, MF_D_colInd, sizeof(int)*MF_D_nnz, cudaMemcpyHostToDevice);
			err = cudaMemcpy(dev_MF_D_Val, MF_D_Val, sizeof(TYPE)*MF_D_nnz * 9, cudaMemcpyHostToDevice);

			//MF_U
			err = cudaMalloc((void**)&dev_MF_GS_U_rowInd, sizeof(int)*MF_U_nnz);
			err = cudaMalloc((void**)&dev_MF_GS_U_rowPtr, sizeof(int)*(number + colors_num[layer] + 1));
			err = cudaMalloc((void**)&dev_MF_GS_U_colInd, sizeof(int)*MF_U_nnz);

			//err = cudaMalloc((void**)&dev_MF_U_rowInd, sizeof(int)*MF_U_nnz);
			err = cudaMalloc((void**)&dev_MF_U_rowPtr, sizeof(int)*(number + 1));
			//err = cudaMalloc((void**)&dev_MF_U_colInd, sizeof(int)*MF_U_nnz);

			err = cudaMemcpy(dev_MF_GS_U_rowInd, MF_GS_U_rowInd, sizeof(int)*MF_U_nnz, cudaMemcpyHostToDevice);
			for (int i = 0; i < colors_num[layer] - 1; i++)
			{
				cuspErr = cusparseXcoo2csr(cusparseHandle, dev_MF_GS_U_rowInd + MF_GS_U_Ptr[i], MF_GS_U_Ptr[i + 1] - MF_GS_U_Ptr[i], \
					color_vertices_num[layer][i + 1] - color_vertices_num[layer][i], dev_MF_GS_U_rowPtr + color_vertices_num[layer][i] + i, CUSPARSE_INDEX_BASE_ZERO);
			}

			err = cudaMemcpy(dev_MF_GS_U_colInd, MF_GS_U_colInd, sizeof(int)*MF_U_nnz, cudaMemcpyHostToDevice);

			err = cudaMemcpy(dev_MF_U_rowInd, MF_U_rowInd, sizeof(int)*MF_U_nnz, cudaMemcpyHostToDevice);
			cuspErr = cusparseXcoo2csr(cusparseHandle, dev_MF_U_rowInd, MF_U_nnz, number, dev_MF_U_rowPtr, CUSPARSE_INDEX_BASE_ZERO);
			err = cudaMemcpy(dev_MF_U_colInd, MF_U_colInd, sizeof(int)*MF_U_nnz, cudaMemcpyHostToDevice);
			err = cudaMemcpy(dev_MF_U_Val, MF_U_Val, sizeof(TYPE)*MF_U_nnz * 9, cudaMemcpyHostToDevice);
		}
		else
		{
			err = cudaMalloc(&dev_MF_rowPtr, sizeof(int)*(number + 1));
			err = cudaMemcpy(dev_MF_rowInd, MF_rowInd, sizeof(int)*MF_nnz, cudaMemcpyHostToDevice);
			cuspErr = cusparseXcoo2csr(cusparseHandle, dev_MF_rowInd, MF_nnz, number, dev_MF_rowPtr, CUSPARSE_INDEX_BASE_ZERO);
			err = cudaMemcpy(dev_MF_colInd, MF_colInd, sizeof(int)*MF_nnz, cudaMemcpyHostToDevice);
			err = cudaMemcpy(dev_MF_Val, MF_Val, sizeof(TYPE)*MF_nnz * 9, cudaMemcpyHostToDevice);
		}
		err = cudaMalloc(&dev_MF_D_offset, sizeof(int)*number);
		err = cudaMemcpy(dev_MF_D_offset, MF_D_offset, sizeof(int)*number, cudaMemcpyHostToDevice);

		if (enable_PD)
		{
			//PD
			err = cudaMalloc(&dev_MF_PD_Val, sizeof(TYPE)*MF_PD_nnz);
			err = cudaMalloc(&dev_MF_PD_rowInd, sizeof(int)*MF_PD_nnz);
			err = cudaMalloc(&dev_MF_PD_rowPtr, sizeof(int)*(3 * number + 1));
			err = cudaMalloc(&dev_MF_PD_colInd, sizeof(int)*MF_PD_nnz);

			err = cudaMemcpy(dev_MF_PD_rowInd, MF_PD_rowInd, sizeof(int)*MF_PD_nnz, cudaMemcpyHostToDevice);
			cuspErr = cusparseXcoo2csr(cusparseHandle, dev_MF_PD_rowInd, MF_PD_nnz, 3 * number, dev_MF_PD_rowPtr, CUSPARSE_INDEX_BASE_ZERO);
			MF_PD_rowPtr = new int[3 * number + 1];
			err = cudaMemcpy(MF_PD_rowPtr, dev_MF_PD_rowPtr, sizeof(int)*(3 * number + 1), cudaMemcpyDeviceToHost);
			err = cudaMemcpy(dev_MF_PD_colInd, MF_PD_colInd, sizeof(int)*MF_PD_nnz, cudaMemcpyHostToDevice);
			err = cudaMemcpy(dev_MF_PD_Val, MF_PD_Val, sizeof(TYPE)*MF_PD_nnz, cudaMemcpyHostToDevice);

			//Host: factor
			MF_PD_reorder = new int[3 * number];
			MF_PD_reordered_colInd = new int[MF_PD_nnz];
			MF_PD_reordered_rowPtr = new int[3 * number + 1];
			MF_PD_reordered_Val = new TYPE[MF_PD_nnz];
			cusolverSpXcsrsymamdHost(cusolverSpHandle, 3 * number, MF_PD_nnz, descr, MF_PD_rowPtr, MF_PD_colInd, MF_PD_reorder);
			memcpy(MF_PD_reordered_rowPtr, MF_PD_rowPtr, sizeof(int)*(3 * number + 1));
			memcpy(MF_PD_reordered_colInd, MF_PD_colInd, sizeof(int)*MF_PD_nnz);
			size_t buffer_size;
			cusolverSpXcsrperm_bufferSizeHost(cusolverSpHandle, 3 * number, 3 * number, MF_PD_nnz, descr, \
				MF_PD_reordered_rowPtr, MF_PD_reordered_colInd, MF_PD_reorder, MF_PD_reorder, &buffer_size);
			MF_PD_buffer = malloc(buffer_size);
			int *mapping = new int[MF_PD_nnz];
			for (int i = 0; i < MF_PD_nnz; i++)
				mapping[i] = i;
			cusolverSpXcsrpermHost(cusolverSpHandle, 3 * number, 3 * number, MF_PD_nnz, descr, \
				MF_PD_reordered_rowPtr, MF_PD_reordered_colInd, MF_PD_reorder, MF_PD_reorder, mapping, MF_PD_buffer);
			for (int i = 0; i < MF_PD_nnz; i++)
				MF_PD_reordered_Val[i] = MF_PD_Val[mapping[i]];
			delete[] mapping;
			cusolverSpCreateCsrluInfoHost(&MF_PD_info);
			cusolverSpXcsrluAnalysisHost(cusolverSpHandle, 3 * number, MF_PD_nnz, descr, \
				MF_PD_reordered_rowPtr, MF_PD_reordered_colInd, MF_PD_info);
			size_t internal_size;
			cusolverSpScsrluBufferInfoHost(cusolverSpHandle, 3 * number, MF_PD_nnz, descr, \
				MF_PD_reordered_Val, MF_PD_reordered_rowPtr, MF_PD_reordered_colInd, MF_PD_info, &internal_size, &buffer_size);
			delete[] MF_PD_buffer;
			MF_PD_buffer = malloc(buffer_size);
			cusolverSpScsrluFactorHost(cusolverSpHandle, 3 * number, MF_PD_nnz, descr, \
				MF_PD_reordered_Val, MF_PD_reordered_rowPtr, MF_PD_reordered_colInd, MF_PD_info, 1.0, MF_PD_buffer);

			PD_B = new TYPE[3 * number];
			PD_X = new TYPE[3 * number];
		}

		// UtAUs
		dev_UtAUs_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);
		dev_UtAUs_Val_Dense = (TYPE**)malloc(sizeof(TYPE*)*layer);
		dev_UtAUs_Diag = (TYPE**)malloc(sizeof(TYPE*)*layer);

		dev_UtAUs_L_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_L_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_L_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_L_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);

		dev_UtAUs_D_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_D_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_D_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_D_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);

		dev_UtAUs_U_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_U_colInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_U_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_U_Val = (TYPE**)malloc(sizeof(TYPE*)*layer);

		dev_UtAUs_GS_U_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_GS_U_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_GS_U_colInd = (int**)malloc(sizeof(int*)*layer);
		
		dev_UtAUs_GS_L_rowInd = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_GS_L_rowPtr = (int**)malloc(sizeof(int*)*layer);
		dev_UtAUs_GS_L_colInd = (int**)malloc(sizeof(int*)*layer);

		dev_UtAUs_D_offset = (int**)malloc(sizeof(int*)*layer);
		

		for (int l = layer - 1; l >= 0; l--)
		{
			if (stored_as_dense[l])
			{
				TYPE *temp = new TYPE[dims[l] * dims[l]];
				memset(temp, 0, sizeof(TYPE)*dims[l] * dims[l]);
				for (int i = 0; i < UtAUs_nnz[l]; i++)
					for (int dx = 0; dx < 12; dx++)
						for (int dy = 0; dy < 12; dy++)
							temp[dims[l] * (12 * UtAUs_colInd[l][i] + dy) + 12 * UtAUs_rowInd[l][i] + dx] = UtAUs_Val[l][144 * i + 12 * dx + dy];
				err = cudaMalloc((void**)&dev_UtAUs_Val[l], sizeof(float)*dims[l] * dims[l]);
				err = cudaMemcpy(dev_UtAUs_Val[l], temp, sizeof(float)*dims[l] * dims[l], cudaMemcpyHostToDevice);
				
				chol_bufferSize = 0;
				checkCudaErrors(cusolverDnSpotrf_bufferSize(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, dims[l], dev_UtAUs_Val[l], dims[l], &chol_bufferSize));
				cudaMalloc(&dev_chol_info, sizeof(int));
				cudaMalloc(&dev_UtAUs_chol, sizeof(TYPE)*dims[l] * dims[l]);
				cudaMalloc(&dev_chol_fixed, sizeof(TYPE)*dims[l]);
				cudaMalloc(&dev_chol_buffer, sizeof(TYPE)*chol_bufferSize);
				cudaMemcpy(dev_UtAUs_chol, dev_UtAUs_Val[l], sizeof(float)*dims[l] * dims[l], cudaMemcpyDeviceToDevice);

				float fix = 1e-5;
				float *h_fix = new float[dims[l]];
				for (int i = 0; i < dims[l]; i++)
					h_fix[i] = fix;
				cudaMemcpy(dev_chol_fixed, h_fix, sizeof(TYPE)*dims[l], cudaMemcpyHostToDevice);
				
				cublasSaxpy(cublasHandle, dims[l], &one, dev_chol_fixed, 1, dev_UtAUs_chol, dims[l] + 1);
				cudaMemset(dev_chol_info, 0, sizeof(int));
				checkCudaErrors(cusolverDnSpotrf(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, dims[l], dev_UtAUs_chol, dims[l], dev_chol_buffer, chol_bufferSize, dev_chol_info));
				int check;
				cudaMemcpy(&check, dev_chol_info, sizeof(int), cudaMemcpyDeviceToHost);
				printf("\ncheck=%d\n", check);
			}
			else
			{
				err = cudaMalloc((void**)&dev_UtAUs_rowInd[l], sizeof(int)*UtAUs_nnz[l]);
				err = cudaMalloc((void**)&dev_UtAUs_colInd[l], sizeof(int)*UtAUs_nnz[l]);
				err = cudaMalloc((void**)&dev_UtAUs_Val[l], sizeof(TYPE)*UtAUs_nnz[l] * 144);

				dev_UtAUs_L_rowInd[l] = dev_UtAUs_rowInd[l];
				dev_UtAUs_L_colInd[l] = dev_UtAUs_colInd[l];
				dev_UtAUs_L_Val[l] = dev_UtAUs_Val[l];

				dev_UtAUs_D_rowInd[l] = dev_UtAUs_rowInd[l] + UtAUs_L_nnz[l];
				dev_UtAUs_D_colInd[l] = dev_UtAUs_colInd[l] + UtAUs_L_nnz[l];
				dev_UtAUs_D_Val[l] = dev_UtAUs_Val[l] + UtAUs_L_nnz[l] * 144;

				dev_UtAUs_U_rowInd[l] = dev_UtAUs_rowInd[l] + UtAUs_L_nnz[l] + UtAUs_D_nnz[l];
				dev_UtAUs_U_colInd[l] = dev_UtAUs_colInd[l] + UtAUs_L_nnz[l] + UtAUs_D_nnz[l];
				dev_UtAUs_U_Val[l] = dev_UtAUs_Val[l] + UtAUs_L_nnz[l] * 144 + UtAUs_D_nnz[l] * 144;

				
				if (stored_as_LDU[l])
				{
					//L
					err = cudaMalloc(&dev_UtAUs_GS_L_rowInd[l], sizeof(int)*UtAUs_L_nnz[l]);
					err = cudaMalloc(&dev_UtAUs_GS_L_rowPtr[l], sizeof(int)*(handles_num[l] + colors_num[l] + 1));
					err = cudaMalloc(&dev_UtAUs_GS_L_colInd[l], sizeof(int)*UtAUs_L_nnz[l]);

					err = cudaMalloc(&dev_UtAUs_L_rowPtr[l], sizeof(int)*(handles_num[l] + 1));

					err = cudaMemcpy(dev_UtAUs_GS_L_rowInd[l], UtAUs_GS_L_rowInd[l], sizeof(int)*UtAUs_L_nnz[l], cudaMemcpyHostToDevice);

					for (int i = 0; i < colors_num[l] - 1; i++)
					{
						cuspErr = cusparseXcoo2csr(cusparseHandle, dev_UtAUs_GS_L_rowInd[l] + UtAUs_GS_L_Ptr[l][i], UtAUs_GS_L_Ptr[l][i + 1] - UtAUs_GS_L_Ptr[l][i], \
							color_vertices_num[l][i + 2] - color_vertices_num[l][i + 1], \
							dev_UtAUs_GS_L_rowPtr[l] + color_vertices_num[l][i + 1] - color_vertices_num[l][1] + i, CUSPARSE_INDEX_BASE_ZERO);

					}
					err = cudaMemcpy(dev_UtAUs_GS_L_colInd[l], UtAUs_GS_L_colInd[l], sizeof(int)*UtAUs_L_nnz[l], cudaMemcpyHostToDevice);


					err = cudaMemcpy(dev_UtAUs_L_rowInd[l], UtAUs_L_rowInd[l], sizeof(int)*UtAUs_L_nnz[l], cudaMemcpyHostToDevice);
					cusparseXcoo2csr(cusparseHandle, dev_UtAUs_L_rowInd[l], UtAUs_L_nnz[l], handles_num[l], dev_UtAUs_L_rowPtr[l], CUSPARSE_INDEX_BASE_ZERO);
					err = cudaMemcpy(dev_UtAUs_L_colInd[l], UtAUs_L_colInd[l], sizeof(int)*UtAUs_L_nnz[l], cudaMemcpyHostToDevice);
					err = cudaMemcpy(dev_UtAUs_L_Val[l], UtAUs_L_Val[l], sizeof(TYPE)*UtAUs_L_nnz[l] * 144, cudaMemcpyHostToDevice);

					//D
					err = cudaMalloc((void**)&dev_UtAUs_D_rowPtr[l], sizeof(int)*(handles_num[l] + 1));

					err = cudaMemcpy(dev_UtAUs_D_rowInd[l], UtAUs_D_rowInd[l], sizeof(int)*UtAUs_D_nnz[l], cudaMemcpyHostToDevice);
					cuspErr = cusparseXcoo2csr(cusparseHandle, dev_UtAUs_D_rowInd[l], handles_num[l], handles_num[l], dev_UtAUs_D_rowPtr[l], CUSPARSE_INDEX_BASE_ZERO);
					err = cudaMemcpy(dev_UtAUs_D_colInd[l], UtAUs_D_colInd[l], sizeof(int)*UtAUs_D_nnz[l], cudaMemcpyHostToDevice);
					err = cudaMemcpy(dev_UtAUs_D_Val[l], UtAUs_D_Val[l], sizeof(TYPE)*UtAUs_D_nnz[l] * 144, cudaMemcpyHostToDevice);

					//U
					err = cudaMalloc((void**)&dev_UtAUs_GS_U_rowInd[l], sizeof(int)*UtAUs_U_nnz[l]);
					err = cudaMalloc((void**)&dev_UtAUs_GS_U_rowPtr[l], sizeof(int)*(handles_num[l] + colors_num[l] + 1));
					err = cudaMalloc((void**)&dev_UtAUs_GS_U_colInd[l], sizeof(int)*UtAUs_U_nnz[l]);

					err = cudaMalloc((void**)&dev_UtAUs_U_rowPtr[l], sizeof(int)*(handles_num[l] + 1));

					err = cudaMemcpy(dev_UtAUs_GS_U_rowInd[l], UtAUs_GS_U_rowInd[l], sizeof(int)*UtAUs_U_nnz[l], cudaMemcpyHostToDevice);
					for (int i = 0; i < colors_num[l] - 1; i++)
					{
						cuspErr = cusparseXcoo2csr(cusparseHandle, dev_UtAUs_GS_U_rowInd[l] + UtAUs_GS_U_Ptr[l][i], UtAUs_GS_U_Ptr[l][i + 1] - UtAUs_GS_U_Ptr[l][i], \
							color_vertices_num[l][i + 1] - color_vertices_num[l][i], dev_UtAUs_GS_U_rowPtr[l] + color_vertices_num[l][i] + i, CUSPARSE_INDEX_BASE_ZERO);
					}

					err = cudaMemcpy(dev_UtAUs_GS_U_colInd[l], UtAUs_GS_U_colInd[l], sizeof(int)*UtAUs_U_nnz[l], cudaMemcpyHostToDevice);

					err = cudaMemcpy(dev_UtAUs_U_rowInd[l], UtAUs_U_rowInd[l], sizeof(int)*UtAUs_U_nnz[l], cudaMemcpyHostToDevice);
					cuspErr = cusparseXcoo2csr(cusparseHandle, dev_UtAUs_U_rowInd[l], UtAUs_U_nnz[l], handles_num[l], dev_UtAUs_U_rowPtr[l], CUSPARSE_INDEX_BASE_ZERO);
					err = cudaMemcpy(dev_UtAUs_U_colInd[l], UtAUs_U_colInd[l], sizeof(int)*UtAUs_U_nnz[l], cudaMemcpyHostToDevice);
					err = cudaMemcpy(dev_UtAUs_U_Val[l], UtAUs_U_Val[l], sizeof(TYPE)*UtAUs_U_nnz[l] * 144, cudaMemcpyHostToDevice);
				}
				else
				{
					err = cudaMalloc(&dev_UtAUs_rowPtr[l], sizeof(int)*(handles_num[l] + 1));
					err = cudaMemcpy(dev_UtAUs_rowInd[l], UtAUs_rowInd[l], sizeof(int)*UtAUs_nnz[l], cudaMemcpyHostToDevice);
					cuspErr = cusparseXcoo2csr(cusparseHandle, dev_UtAUs_rowInd[l], UtAUs_nnz[l], handles_num[l], dev_UtAUs_rowPtr[l], CUSPARSE_INDEX_BASE_ZERO);
					err = cudaMemcpy(dev_UtAUs_colInd[l], UtAUs_colInd[l], sizeof(int)*UtAUs_nnz[l], cudaMemcpyHostToDevice);
					err = cudaMemcpy(dev_UtAUs_Val[l], UtAUs_Val[l], sizeof(TYPE)*UtAUs_nnz[l] * 144, cudaMemcpyHostToDevice);
				}

				cudaMalloc(&dev_UtAUs_D_offset[l], sizeof(int)*handles_num[l]);
				cudaMemcpy(dev_UtAUs_D_offset[l], UtAUs_D_offset[l], sizeof(int)*handles_num[l], cudaMemcpyHostToDevice);
			}
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

		for (int i = 0; i < number; i++)
			for (int dj = 0; dj < 4; dj++)
				for (int di = 0; di < 4; di++)
				{
					TYPE v1 = di == 3 ? 1 : X[3 * i + di];
					TYPE v2 = dj == 3 ? 1 : X[3 * i + dj];
					precomputed_Diag_addition[i * 16 + 4 * dj + di] = v1 * v2;
				}

		cudaErr = cudaMalloc(&dev_precomputed_Diag_addition, sizeof(float)*number * 16);
		cudaErr = cudaMemcpy(dev_precomputed_Diag_addition, precomputed_Diag_addition, sizeof(float)*number * 16, cudaMemcpyHostToDevice);

		cudaErr = cudaMalloc(&dev_MF_Diag_addition, sizeof(float)*dims[layer]);

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
		Control_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_X, dev_more_fixed, number, select_v);
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
			MF_Diag_Update_Kernel << <((dims[l] + threadsPerBlock - 1) / threadsPerBlock), threadsPerBlock >> > (dev_temp_X[l], dev_MF_Diag_addition, P, dims[l]);
		else
			UtAUs_Diag_Update_Kernel << <((handles_num[l] + threadsPerBlock - 1) / threadsPerBlock), threadsPerBlock >> > (dev_temp_X[l], dev_UtAUs_Diag_addition[l], P, handles_num[l]);
		cudaErr = cudaGetLastError();
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
	void calculateAP(int l, float *AP, const float *alpha, const float *P)
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
		{
			if (stored_as_LDU[l])
			{
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, number, number, MF_L_nnz, alpha, descr, \
					dev_MF_L_Val, dev_MF_L_rowPtr, dev_MF_L_colInd, 3, P, &one, AP);
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, number, number, MF_D_nnz, alpha, descr, \
					dev_MF_D_Val, dev_MF_D_rowPtr, dev_MF_D_colInd, 3, P, &one, AP);
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, number, number, MF_U_nnz, alpha, descr, \
					dev_MF_U_Val, dev_MF_U_rowPtr, dev_MF_U_colInd, 3, P, &one, AP);
			}
			else
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, number, number, MF_nnz, alpha, descr, \
					dev_MF_Val, dev_MF_rowPtr, dev_MF_colInd, 3, P, &one, AP);
			
		}
		else
		{
			if (stored_as_dense[l])
				cublasStatus = cublasSgemv(cublasHandle, CUBLAS_OP_N, dims[l], dims[l], alpha, dev_UtAUs_Val[l], dims[l], P, 1, &one, AP, 1);
			else
			{
				if (stored_as_LDU[l])
				{
					cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, handles_num[l], handles_num[l], UtAUs_L_nnz[l], \
						alpha, descr, dev_UtAUs_L_Val[l], dev_UtAUs_L_rowPtr[l], dev_UtAUs_L_colInd[l], 12, P, &one, AP);
					cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, handles_num[l], handles_num[l], UtAUs_D_nnz[l], \
						alpha, descr, dev_UtAUs_D_Val[l], dev_UtAUs_D_rowPtr[l], dev_UtAUs_D_colInd[l], 12, P, &one, AP);
					cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, handles_num[l], handles_num[l], UtAUs_U_nnz[l], \
						alpha, descr, dev_UtAUs_U_Val[l], dev_UtAUs_U_rowPtr[l], dev_UtAUs_U_colInd[l], 12, P, &one, AP);
				}
				else
					cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, handles_num[l], handles_num[l], UtAUs_nnz[l], \
						alpha, descr, dev_UtAUs_Val[l], dev_UtAUs_rowPtr[l], dev_UtAUs_colInd[l], 12, P, &one, AP);
			}
		}

		//Diag_Update(l, AP, alpha, P);
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

	void performCGIteration(int l, int max_iter, float tol)
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;
		float r0, r1;
		float alpha, beta, neg_alpha;
		float dot;
		int k;
		float *P = dev_P[l];
		float *R = dev_R[l];
		float *AP = dev_AP[l];
		float *deltaX = dev_deltaX[l];
		cublasStatus = cublasSdot(cublasHandle, dims[l], R, 1, R, 1, &r1);
		if (r1 < EPSILON) return;
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
			//printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
			k++;
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

			float *P = dev_P[l];
			float *R = dev_R[l];
			float *B = dev_B[l];
			float *deltaX = dev_deltaX[l];
			for (int k = 0; k < max_iter; k++)
			{
				cublasErr = cublasScopy(cublasHandle, dims[l], B, 1, dev_temp_X[l], 1);
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, number, number, MF_L_nnz, &minus_one, descr, \
					dev_MF_L_Val, dev_MF_L_rowPtr, dev_MF_L_colInd, 3, deltaX, &one, dev_temp_X[l]);

				cudaErr = cudaMemset(deltaX, 0, sizeof(float)*dims[l]);

				for (int c = colors_num[l] - 1; c >= 0; c--)
				{
					int base = color_vertices_num[l][c];
					Colored_GS_MF_Kernel << <(color_vertices_num[l][c + 1] - base + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > \
						(deltaX , dev_MF_D_Val, dev_temp_X[l], base, color_vertices_num[l][c + 1] - base);
					if (c)
					{
						cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, color_vertices_num[l][c] - color_vertices_num[l][c - 1], \
							color_vertices_num[l][colors_num[l]] - color_vertices_num[l][c], MF_GS_U_Ptr[c] - MF_GS_U_Ptr[c - 1], &minus_one, descr, \
							dev_MF_U_Val + MF_GS_U_Ptr[c - 1] * 9, dev_MF_GS_U_rowPtr + color_vertices_num[l][c - 1] + c - 1, dev_MF_GS_U_colInd + MF_GS_U_Ptr[c - 1], \
							3, deltaX + 3 * base, &one, dev_temp_X[l] + 3 * color_vertices_num[l][c - 1]);
					}
				}

				cublasErr = cublasScopy(cublasHandle, dims[l], B, 1, dev_temp_X[l], 1);
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, number, number, MF_U_nnz, &minus_one, descr, \
					dev_MF_U_Val, dev_MF_U_rowPtr, dev_MF_U_colInd, 3, deltaX, &one, dev_temp_X[l]);

				cudaErr = cudaMemset(deltaX, 0, sizeof(float)*dims[l]);

				for (int c = 0; c < colors_num[l]; c++)
				{
					int base= color_vertices_num[l][c];
					Colored_GS_MF_Kernel << <(color_vertices_num[l][c + 1] - base + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > \
						(deltaX, dev_MF_D_Val, dev_temp_X[l], base, color_vertices_num[l][c + 1] - base);
					if (c < colors_num[l] - 1)
					{
						cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, \
							color_vertices_num[l][c + 2] - color_vertices_num[l][c + 1], color_vertices_num[l][c + 1], MF_GS_L_Ptr[c + 1] - MF_GS_L_Ptr[c], \
							&minus_one, descr, dev_MF_L_Val + MF_GS_L_Ptr[c] * 9, dev_MF_GS_L_rowPtr + color_vertices_num[l][c + 1] - color_vertices_num[l][1] + c, \
							dev_MF_GS_L_colInd + MF_GS_L_Ptr[c], 3, deltaX, &one, dev_temp_X[l] + 3 * color_vertices_num[l][c + 1]);
					}
				}
			}
			cublasScopy(cublasHandle, dims[l], B, 1, R, 1);
			calculateAP(l, R, &minus_one, deltaX);
		}
		else
		{
			float *P = dev_P[l];
			float *R = dev_R[l];
			float *B = dev_B[l];
			float *deltaX = dev_deltaX[l];
			for (int k = 0; k < max_iter; k++)
			{
				cublasErr = cublasScopy(cublasHandle, dims[l], B, 1, dev_temp_X[l], 1);
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, handles_num[l], handles_num[l], \
					UtAUs_L_nnz[l], &minus_one, descr, dev_UtAUs_L_Val[l], dev_UtAUs_L_rowPtr[l], dev_UtAUs_L_colInd[l], 12, deltaX, &one, dev_temp_X[l]);

				cudaErr = cudaMemset(deltaX, 0, sizeof(float)*dims[l]);
				for (int c = colors_num[l] - 1; c >= 0; c--)
				{
					int base = color_vertices_num[l][c];

					//Colored_GS_UtAU_Kernel << <(color_vertices_num[l][c + 1] - base + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >\
							(deltaX, dev_UtAUs_D_Val[l], dev_temp_X[l], base, color_vertices_num[l][c + 1] - base);
					Mogai_Kernel << <color_vertices_num[l][c + 1] - base, 12 >> > \
						(deltaX, dev_UtAUs_D_Val[l], dev_temp_X[l], base);
					//Mogai_2_Kernel << <(color_vertices_num[l][c + 1] - base + 1) / 2, 24 >> > \
							(deltaX, dev_UtAUs_D_Val[l], dev_temp_X[l], base, color_vertices_num[l][c + 1] - base);

					if (c)
					{
						if (stored_as_dense[l])
							cublasErr = cublasSgemv(cublasHandle, CUBLAS_OP_N, 4 * (color_vertices_num[l][c] - color_vertices_num[l][c - 1]), \
								4 * (color_vertices_num[l][colors_num[l]] - color_vertices_num[l][c]), &minus_one, \
								dev_UtAUs_Val_Dense[l] + dims[l] * 4 * color_vertices_num[l][c] + 4 * color_vertices_num[l][c - 1], dims[l], \
								P + 4 * base, 1, &one, dev_temp_X[l] + 4 * color_vertices_num[l][c - 1], 1);
						else
							cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, color_vertices_num[l][c] - color_vertices_num[l][c - 1], \
								color_vertices_num[l][colors_num[l]] - color_vertices_num[l][c], UtAUs_GS_U_Ptr[l][c] - UtAUs_GS_U_Ptr[l][c - 1], &minus_one, descr, \
								dev_UtAUs_U_Val[l] + 144 * UtAUs_GS_U_Ptr[l][c - 1], dev_UtAUs_GS_U_rowPtr[l] + color_vertices_num[l][c - 1] + c - 1, dev_UtAUs_GS_U_colInd[l] + UtAUs_GS_U_Ptr[l][c - 1], \
								12, deltaX + 12 * base, &one, dev_temp_X[l] + 12 * color_vertices_num[l][c - 1]);
					}

				}

				cublasErr = cublasScopy(cublasHandle, dims[l], B, 1, dev_temp_X[l], 1);
				cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, handles_num[l], handles_num[l], \
					UtAUs_U_nnz[l], &minus_one, descr, dev_UtAUs_U_Val[l], dev_UtAUs_U_rowPtr[l], dev_UtAUs_U_colInd[l], 12, deltaX, &one, dev_temp_X[l]);

				cudaErr = cudaMemset(deltaX, 0, sizeof(float)*dims[l]);
				for (int c = 0; c < colors_num[l]; c++)
				{
					int base = color_vertices_num[l][c];
					//Colored_GS_UtAU_Kernel << <(color_vertices_num[l][c + 1] - base + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >\
							(deltaX, dev_UtAUs_D_Val[l], dev_temp_X[l], base, color_vertices_num[l][c + 1] - base);
					Mogai_Kernel << <color_vertices_num[l][c + 1] - base, 12 >> > \
						(deltaX, dev_UtAUs_D_Val[l], dev_temp_X[l], base);
					//Mogai_2_Kernel << <(color_vertices_num[l][c + 1] - base + 1) / 2, 24 >> > \
							(deltaX, dev_UtAUs_D_Val[l], dev_temp_X[l], base, color_vertices_num[l][c + 1] - base);
					if (c < colors_num[l] - 1)
					{
						if (stored_as_dense[l])
							cublasErr = cublasSgemv(cublasHandle, CUBLAS_OP_N, 4 * (color_vertices_num[l][c] - color_vertices_num[l][c - 1]), \
								4 * (color_vertices_num[l][colors_num[l]] - color_vertices_num[l][c]), &minus_one, \
								dev_UtAUs_Val_Dense[l] + dims[l] * 4 * color_vertices_num[l][c] + 4 * color_vertices_num[l][c - 1], dims[l], \
								P + 4 * base, 1, &one, dev_temp_X[l] + 4 * color_vertices_num[l][c - 1], 1);
						else
							cuspErr = cusparseSbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, color_vertices_num[l][c + 2] - color_vertices_num[l][c + 1], \
								color_vertices_num[l][c + 1], UtAUs_GS_L_Ptr[l][c + 1] - UtAUs_GS_L_Ptr[l][c], &minus_one, descr, \
								dev_UtAUs_L_Val[l] + 144 * UtAUs_GS_L_Ptr[l][c], dev_UtAUs_GS_L_rowPtr[l] + color_vertices_num[l][c + 1] - color_vertices_num[l][1] + c, \
								dev_UtAUs_GS_L_colInd[l] + UtAUs_GS_L_Ptr[l][c], 12, deltaX, &one, dev_temp_X[l] + 12 * color_vertices_num[l][c + 1]);
					}

				}

			}
			cublasScopy(cublasHandle, dims[l], B, 1, R, 1);
			calculateAP(l, R, &minus_one, deltaX);
		}
	}

	void performJacobiIteration(int l, int max_iter, float tol)
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasErr;
		float r1;
		if (l == layer)
		{
			float *P = dev_P[l];
			float *R = dev_R[l];
			float *deltaX = dev_deltaX[l];
			for (int k = 0; k < max_iter; k++)
			{
				cudaMemset(P, 0, sizeof(float)*dims[l]);

				Jacobi_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (P, dev_MF_Val, R, dev_MF_D_offset, number);

				cublasErr = cublasSaxpy(cublasHandle, dims[l], &one, P, 1, deltaX, 1);

				calculateAP(l, R, &minus_one, P);
			}
		}
		else
		{
			float *P = dev_P[l];
			float *R = dev_R[l];
			float *deltaX = dev_deltaX[l];
			for (int k = 0; k < max_iter; k++)
			{
				cudaMemset(P, 0, sizeof(float)*dims[l]);

				Jacobi_Mogai_Kernel << <handles_num[l] , 12 >> > (P, dev_UtAUs_Val[l], R, dev_UtAUs_D_offset[l], handles_num[l]);

				cublasErr = cublasSaxpy(cublasHandle, dims[l], &one, P, 1, deltaX, 1);

				calculateAP(l, R, &minus_one, P);
			}
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
		
		cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l - 1], dims[l], \
			Us_nnz[l - 1], &one, descr, dev_Uts_Val[l - 1], dev_Uts_rowPtr[l - 1], dev_Uts_colInd[l - 1], dev_R[l], &zero, dev_B[l - 1]);

		cublasScopy(cublasHandle, dims[l - 1], dev_B[l - 1], 1, dev_R[l - 1], 1);

		cudaErr = cudaMemset(dev_deltaX[l - 1], 0, sizeof(float) * dims[l - 1]);

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

		cuspErr = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dims[l + 1], dims[l], \
			Us_nnz[l], &one, descr, dev_Us_Val[l], dev_Us_rowPtr[l], dev_Us_colInd[l], dev_deltaX[l], &zero, dev_temp_X[l + 1]);

		cublasStatus = cublasSaxpy(cublasHandle, dims[l + 1], &one, dev_temp_X[l + 1], 1, dev_deltaX[l + 1], 1);

		calculateAP(l + 1, dev_R[l + 1], &minus_one, dev_temp_X[l + 1]);

		l++;
	}

	void solve(int l)
	{
		if (l == layer)
		{
			float *temp = new float[3 * number];
			cudaMemcpy(temp, dev_B[l], sizeof(float) * 3 * number, cudaMemcpyDeviceToHost);
			for (int i = 0; i < 3 * number; i++)
				PD_B[i] = temp[MF_PD_reorder[i]];
			cusolverSpScsrluSolveHost(cusolverSpHandle, 3 * number, PD_B, PD_X, MF_PD_info, MF_PD_buffer);
			for (int i = 0; i < 3 * number; i++)
				temp[MF_PD_reorder[i]] = PD_X[i];
			cudaMemcpy(dev_deltaX[l], temp, sizeof(float) * 3 * number, cudaMemcpyHostToDevice);
			calculateAP(l, dev_R[l], &minus_one, dev_deltaX[l]);
			//cudaMemcpy(dev_deltaX[l], dev_B[l], sizeof(float) * 3 * number, cudaMemcpyDeviceToDevice);
			//cusolverRfSolve(cusolverRfHandle, dev_MF_PD_P, dev_MF_PD_Q, 1, dev_temp_X[l], 3 * number, dev_deltaX, 3 * number);
		}
		if (l == 0 && layer)
		{
			cudaMemcpy(dev_deltaX[l], dev_B[l], sizeof(float)*dims[l], cudaMemcpyDeviceToDevice);
			checkCudaErrors(cusolverDnSpotrs(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, dims[l], 1, dev_UtAUs_chol, dims[l], dev_deltaX[l], dims[l], dev_chol_info));
			calculateAP(l, dev_R[l], &minus_one, dev_deltaX[l]);
		}
	}

	void Apply_Constraints(TYPE t)
	{}

	void Update(TYPE t)
	{}

	void computeGradient(TYPE t)
	{
		cudaMemset(dev_B[layer], 0, sizeof(float) * dims[layer]);

		Energy_Gradient_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_B[layer], dev_all_VV, dev_all_VL, dev_all_vv_num, spring_k, dev_fixed, dev_more_fixed, \
			dev_collision_fixed, dev_fixed_X, dev_X, dev_vertex2index[layer], control_mag, collision_mag, number);

		Inertia_Gradient_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_B[layer], dev_inertia_X, dev_X, dev_vertex2index[layer], dev_M, 1.0 / t, number);
	}

	void computeEnergy(float *E, const float *X, TYPE t)
	{
		cudaMemset(dev_Energy, 0, sizeof(float));
		Energy_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_Energy, dev_all_VV, dev_all_VL, dev_all_vv_num, spring_k, dev_fixed, dev_more_fixed, \
			dev_collision_fixed, dev_fixed_X, X, dev_vertex2index[layer], control_mag, collision_mag, number);

		Inertia_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_Energy, dev_inertia_X, X, dev_vertex2index[layer], dev_M, 1.0 / t, number);

		cudaMemcpy(E, dev_Energy, sizeof(float), cudaMemcpyDeviceToHost);
	}

	void computeHessian(TYPE t)
	{
		cudaMemset(dev_MF_Val, 0, sizeof(float)*MF_nnz * 9);

		Hessian_Kernel << <(spring_num + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > \
			(dev_MF_Val, dev_X, dev_spring_v1, dev_spring_v2, dev_spring_len, dev_spring_update_offset, spring_k, spring_num);

		Hessian_Diag_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > \
			(dev_MF_Val, dev_M, 1.0 / t, dev_fixed, dev_more_fixed, control_mag, dev_vertex2index[layer], dev_MF_D_offset, number);
		//cudaDeviceSynchronize();
		//}
		//printf("%f ", timer.Get_Time());
		for (int l = layer - 1; l >= 0; l--)
		{
			cudaMemset(dev_UtAUs_Val[l], 0, sizeof(float)*UtAUs_nnz[l] * 9);

			//timer.Start();
			//for (int tst = 0; tst < 50; tst++)
			//{
			int n = ((l + 1) == layer) ? MF_nnz : UtAUs_nnz[l + 1];
			float *temp = ((l + 1) == layer) ? dev_MF_Val : dev_UtAUs_Val[l + 1];
			//int n = UtAUs_L_nnz[l + 1]+ UtAUs_D_nnz[l + 1];
			//int n = UtAUs_half_num[l];
			if(l)
			UtAU_2_UtAU_sparse_Kernel << <(n + UtAUs_threadPerBlock - 1) / UtAUs_threadPerBlock, dim3(9, UtAUs_threadPerBlock) >> > \
				(dev_UtAUs_Val[l], temp, dev_UtAUs_update_offset[l], n);
			else
			{
			}
			//	cudaDeviceSynchronize();
			//}
			//printf("%f ", timer.Get_Time());
			//}

		}

	}

	void lineSearch(TYPE t)
	{
		float alpha = 0.003, beta = 0.1;
		cudaMemcpy(dev_temp_X[layer], dev_X, sizeof(float) * 3 * number, cudaMemcpyDeviceToDevice);
		float E0;
		computeEnergy(&E0, dev_temp_X[layer], t);
		for (int i = 0; i < 3; i++)
		{
#ifdef NO_ORDER
			cublasStatus = cublasSaxpy(cublasHandle, dims[layer], &one, dev_deltaX[layer], 1, dev_X, 1);
#else
			Update_DeltaX_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_X, dev_deltaX[layer], dev_index2vertex[layer], number);
#endif
			float E = 0, L = 0;
			computeEnergy(&E, dev_X, t);
			cublasSdot(cublasHandle, 3 * number, dev_deltaX[layer], 1, dev_B[layer], 1, &L);
			E += L * alpha*t;
			if (E < E0 + EPSILON) return;
			cudaMemcpy(dev_X, dev_temp_X[layer], sizeof(float) * 3 * number, cudaMemcpyDeviceToDevice);
			cublasSscal(cublasHandle, 3 * number, &beta, dev_deltaX[layer], 1);
			//printf("should not occur in PD\n");
		}
#ifdef NO_ORDER
		cublasStatus = cublasSaxpy(cublasHandle, dims[layer], &one, dev_deltaX[layer], 1, dev_X, 1);
#else
		Update_DeltaX_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_X, dev_deltaX[layer], dev_index2vertex[layer], number);
#endif
	}

	void Update(TYPE t, int iterations, TYPE dir[])
	{
		cudaError_t cudaErr;
		cusparseStatus_t cuspErr;
		cublasStatus_t cublasStatus;

		int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;
		//int tet_blocksPerGrid = (tet_number + tet_threadsPerBlock - 1) / tet_threadsPerBlock;

		TIMER fps_timer;
		TIMER timer;
		TIMER test_timer;

		float now_time = 0;

		float g_norm;
		float E;
		float e_norm;

		//whm
		float omega = 1.0;

		float finest_update_time = 0;
		float coarse_update_time = 0;
		float gradient_time = 0;
		float smoothing_time = 0;
		float solve_time = 0;
		float sample_time = 0;

		// Step 0: Set up Diag data
		Fixed_Update_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_X, dev_collision_fixed, dev_fixed, dev_more_fixed, dev_fixed_X, number, dir[0], dir[1], dir[2]);

		//if (layer)
		//{
		//	int sum = 0;
		//	for (int l = 0; l < layer; l++)
		//		sum += handles_num[l] * 16;
		//	cudaErr = cudaMemset(*dev_UtAUs_Diag_addition, 0, sizeof(float)*sum);
		//}
		//cudaErr = cudaMemset(dev_MF_Diag_addition, 0, sizeof(float)*dims[layer]);
		//Diag_Update_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_MF_Diag_addition, dev_more_fixed, dev_collision_fixed, dev_vertex2index[layer], control_mag, \
		//	collision_mag, dev_where_to_update, dev_precomputed_Diag_addition, *dev_UtAUs_Diag_addition, dev_handles_num, number, layer);

		for (int s = 0; s < sub_step; s++)
		{

			cublasStatus = cublasScopy(cublasHandle, dims[layer], dev_X, 1, dev_old_X, 1);

			// Step 1: Basic update
			Basic_Update_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_X, dev_V, damping, t, number);

			cublasStatus = cublasScopy(cublasHandle, dims[layer], dev_X, 1, dev_inertia_X, 1);



			// Our Step 3: MultiGrid Method
			timer.Start();

			if (enable_benchmark==1)
			{
				float min_E;
				computeEnergy(&min_E, dev_target_X, t);
				fprintf(benchmark, "%f\n", min_E);
			}


			for (int it = 0; pd_iters == -1 || it < pd_iters; it++)
			{
				//Tet_Constraint_Kernel << <tet_blocksPerGrid, tet_threadsPerBlock >> > (dev_X, dev_Tet, dev_inv_Dm, dev_Vol, dev_Tet_Temp, elasticity, tet_number, l);
				//cudaErr = cudaGetLastError();

				if (use_WHM)
				{
					cublasScopy(cublasHandle, 3 * number, dev_prev_X, 1, dev_prev_prev_X, 1);
					cublasScopy(cublasHandle, 3 * number, dev_X, 1, dev_prev_X, 1);
				}
				//timer.Start();
				//for (int tst = 0; tst < 50; tst++)
				//{
				if (use_Hessian)
				{
					test_timer.Start();
					for (int tst = 0; tst < 50; tst++)
					{
						cudaErr = cudaMemset(dev_MF_Val, 0, sizeof(float)*MF_nnz * 9);

						Hessian_Kernel << <(spring_num + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > \
							(dev_MF_Val, dev_X, dev_spring_v1, dev_spring_v2, dev_spring_len, dev_spring_update_offset, spring_k, spring_num);

						Hessian_Diag_Kernel << <(number + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > \
							(dev_MF_Val, dev_M, 1.0 / t, dev_fixed, dev_more_fixed, control_mag, dev_vertex2index[layer], dev_MF_D_offset, number);

						cudaDeviceSynchronize();
					}

					finest_update_time += test_timer.Get_Time() * 20;

						//cudaDeviceSynchronize();
						//}
						//printf("%f ", timer.Get_Time());
					test_timer.Start();
					for(int tst=0;tst<50;tst++)
					{
						for (int l = layer - 1; l >= 0; l--)
						{
							if (stored_as_dense[l])
								cudaMemset(dev_UtAUs_Val[l], 0, sizeof(float)*dims[l] * dims[l]);
							else
								cudaErr = cudaMemset(dev_UtAUs_Val[l], 0, sizeof(float)*UtAUs_nnz[l] * 144);
							if (l == layer - 1)
							{
								int n = UtAUs_half_num[l];
								//int n = MF_nnz;
								MF_Mapping_Kernel << <(n * 9 + Reduction_threadsPerBlock - 1) / Reduction_threadsPerBlock, Reduction_threadsPerBlock * 16 >> > \
									(dev_MF_temp_Val, dev_MF_Val, dev_UtAUs_precomputed_4x4[l], dev_UtAUs_half_offset[l], n * 9);

								if (stored_as_dense[l])
								{
									MF_2_UtAU_dense_reduction_Kernel << <(n * 9 + Reduction_threadsPerBlock - 1) / Reduction_threadsPerBlock, Reduction_threadsPerBlock * 16 >> > \
										(dev_UtAUs_Val[l], dev_MF_temp_Val, dev_UtAUs_update_offset[l], dev_UtAUs_half_offset[l], handles_num[l], n * 9);
									UtAUs_dense_Mirror_Kernel << <(UtAUs_L_nnz[l] + UtAUs_threadPerBlock - 1) / UtAUs_threadPerBlock, UtAUs_threadPerBlock * 144 >> > \
										(dev_UtAUs_Val[l], dev_UtAUs_L_offset[l], dev_UtAUs_mirror_offset[l], handles_num[l], UtAUs_L_nnz[l]);

									cudaMemcpy(dev_UtAUs_chol, dev_UtAUs_Val[l], sizeof(float)*dims[l] * dims[l], cudaMemcpyDeviceToDevice);
									//cublasSaxpy(cublasHandle, dims[l], &one, dev_chol_fixed, 1, dev_UtAUs_chol, dims[l] + 1);
									checkCudaErrors(cusolverDnSpotrf(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, dims[l], dev_UtAUs_chol, dims[l], dev_chol_buffer, chol_bufferSize, dev_chol_info));


								}
								else
								{
									MF_2_UtAU_sparse_reduction_Kernel << <(n * 9 + Reduction_threadsPerBlock - 1) / Reduction_threadsPerBlock, Reduction_threadsPerBlock * 16 >> > \
										(dev_UtAUs_Val[l], dev_MF_temp_Val, dev_UtAUs_update_offset[l], dev_UtAUs_half_offset[l], n * 9);

									UtAUs_sparse_Mirror_Kernel << <(UtAUs_L_nnz[l] + UtAUs_threadPerBlock - 1) / UtAUs_threadPerBlock, UtAUs_threadPerBlock * 144 >> > \
										(dev_UtAUs_Val[l], dev_UtAUs_L_offset[l], dev_UtAUs_mirror_offset[l], UtAUs_L_nnz[l]);

									UtAUs_Diag_Rank_Fix_Kernel << <(handles_num[l] + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > \
										(dev_UtAUs_Val[l], dev_UtAUs_D_offset[l], handles_num[l]);
								}
								//}
								//printf("%f ", timer.Get_Time());
							}
							else
							{
								//timer.Start();
								//for (int tst = 0; tst < 50; tst++)
								//{
								int n = UtAUs_nnz[l + 1];
								//int n = UtAUs_L_nnz[l + 1]+ UtAUs_D_nnz[l + 1];
								//int n = UtAUs_half_num[l];
								if (l || !stored_as_dense[l])
									UtAU_2_UtAU_sparse_Kernel << <(n + UtAUs_threadPerBlock - 1) / UtAUs_threadPerBlock, dim3(144, UtAUs_threadPerBlock) >> > \
									(dev_UtAUs_Val[l], dev_UtAUs_Val[l + 1], dev_UtAUs_update_offset[l], dev_UtAUs_half_offset[l], n);
								else
								{
									UtAU_2_UtAU_dense_Kernel << <(n + UtAUs_threadPerBlock - 1) / UtAUs_threadPerBlock, dim3(144, UtAUs_threadPerBlock) >> > \
										(dev_UtAUs_Val[l], dev_UtAUs_Val[l + 1], dev_UtAUs_update_offset[l], handles_num[l], n);
									cudaMemcpy(dev_UtAUs_chol, dev_UtAUs_Val[l], sizeof(float)*dims[l] * dims[l], cudaMemcpyDeviceToDevice);
									cublasSaxpy(cublasHandle, dims[l], &one, dev_chol_fixed, 1, dev_UtAUs_chol, dims[l] + 1);
									checkCudaErrors(cusolverDnSpotrf(cusolverDnHandle, CUBLAS_FILL_MODE_LOWER, dims[l], dev_UtAUs_chol, dims[l], dev_chol_buffer, chol_bufferSize, dev_chol_info));

								}
								//	cudaDeviceSynchronize();
								//}
								//printf("%f ", timer.Get_Time());
								//}
							}
						}

						cudaDeviceSynchronize();

					}
					coarse_update_time += test_timer.Get_Time() * 20;
				}

#ifdef MATRIX_OUTPUT
				char str[256];
				sprintf(str, "benchmark\\A.txt");
				cudaMemcpy(MF_Val, dev_MF_Val, sizeof(float)*MF_nnz * 9, cudaMemcpyDeviceToHost);
				outputBoo(str, number, number, MF_rowInd, MF_colInd, MF_Val, MF_nnz, 3);
				for (int l = 0; l < layer; l++)
				{
					sprintf(str, "benchmark\\UtAU%d.txt", l);
					cudaMemcpy(UtAUs_Val[l], dev_UtAUs_Val[l], sizeof(float)*UtAUs_nnz[l] * 144, cudaMemcpyDeviceToHost);
					outputBoo(str, handles_num[l], handles_num[l], UtAUs_rowInd[l], UtAUs_colInd[l], UtAUs_Val[l], UtAUs_nnz[l], 12);
				}

#endif
				test_timer.Start();
				for (int tst = 0; tst < 50; tst++)
				{

					cudaErr = cudaMemset(dev_B[layer], 0, sizeof(float) * dims[layer]);

					Energy_Gradient_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_B[layer], dev_all_VV, dev_all_VL, dev_all_vv_num, spring_k, dev_fixed, dev_more_fixed, \
						dev_collision_fixed, dev_fixed_X, dev_X, dev_vertex2index[layer], control_mag, collision_mag, number);

					Inertia_Gradient_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_B[layer], dev_inertia_X, dev_X, dev_vertex2index[layer], dev_M, 1.0 / t, number);

					cublasScopy(cublasHandle, dims[layer], dev_B[layer], 1, dev_R[layer], 1);

					cudaErr = cudaMemset(dev_deltaX[layer], 0, sizeof(float) * dims[layer]);

					cudaDeviceSynchronize();
				}
				gradient_time += test_timer.Get_Time() * 20;
				if (enable_benchmark==1)
				{
					cublasStatus = cublasSdot(cublasHandle, 3 * number, dev_R[layer], 1, dev_R[layer], 1, &g_norm);
					computeEnergy(&E, dev_X, t);
					cudaMemcpy(dev_temp_X[layer], dev_X, sizeof(float) * 3 * number, cudaMemcpyDeviceToDevice);
					cublasSaxpy(cublasHandle, 3 * number, &minus_one, dev_target_X, 1, dev_temp_X[layer], 1);
					cublasStatus = cublasSdot(cublasHandle, 3 * number, dev_temp_X[layer], 1, dev_temp_X[layer], 1, &e_norm);
					fprintf(benchmark, "%f %f %f %f\n", timer.Get_Time(), g_norm, E, e_norm);
				}
				//if (timer.Get_Time() > 10) break;
				
				//timer.Start();
				int now_Layer = layer;
#ifdef SETTINGF

				int idx = 0;

				for (iterationSetting s : settings)
				{
					test_timer.Start();
					for (int tst = 0; tst < 50; tst++)
					{
						switch (s.first)
						{
						case Jacobi:
						{
							cublasScopy(cublasHandle, dims[now_Layer], dev_B[now_Layer], 1, dev_R[now_Layer], 1);
							cudaErr = cudaMemset(dev_deltaX[now_Layer], 0, sizeof(float) * dims[now_Layer]);
							performJacobiIteration(now_Layer, s.second, tol);
							break;
						}
						case GaussSeidel:
						{
							cublasScopy(cublasHandle, dims[now_Layer], dev_B[now_Layer], 1, dev_R[now_Layer], 1);
							cudaErr = cudaMemset(dev_deltaX[now_Layer], 0, sizeof(float) * dims[now_Layer]);
							performGaussSeidelIteration(now_Layer, s.second, tol);
							break;
						}
						case CG:
						{
							cublasScopy(cublasHandle, dims[now_Layer], dev_B[now_Layer], 1, dev_R[now_Layer], 1);
							cudaErr = cudaMemset(dev_deltaX[now_Layer], 0, sizeof(float) * dims[now_Layer]);
							performCGIteration(now_Layer, s.second, tol);
							break;
						}
						case Direct:
						{
							cublasScopy(cublasHandle, dims[now_Layer], dev_B[now_Layer], 1, dev_R[now_Layer], 1);
							solve(now_Layer);
							break;
						}
						case DownSample:
						{
							downSample(now_Layer);
							if(tst<49) now_Layer++;
							break;
						}
						case UpSample:
						{
							upSample(now_Layer);
							if(tst<49) now_Layer--;
							break;
						}
						default:
							break;
						}
						cudaDeviceSynchronize();
					}

					switch (s.first)
					{
					case Jacobi:
					case GaussSeidel:
					case CG:
						smoothing_time += test_timer.Get_Time() * 20;
						break;
					case Direct:
						solve_time += test_timer.Get_Time() * 20;
						break;
					case DownSample:
					case UpSample:
						sample_time += test_timer.Get_Time() * 20;
					default:
						break;
					}

				}
				printf("%f %f %f %f %f %f\n", finest_update_time, coarse_update_time, gradient_time, smoothing_time, solve_time, sample_time);
					
#endif
#ifdef SETTING1
				//performCGIteration(now_Layer, 25, tol);
				//performJacobiIteration(now_Layer, 1, tol);
				performGaussSeidelIteration(now_Layer, 16, tol);
				//performGaussSeidelIteration(now_Layer, 64, tol);
				//performJacobiIteration(now_Layer, 1, tol)
#endif

#ifdef SETTING2
				//performCGIteration(now_Layer, 25, tol);

				//performCGIteration(now_Layer, 5, tol);
				//downSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);
				//upSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);

				performGaussSeidelIteration(now_Layer, 5, tol);
				downSample(now_Layer);
				performCGIteration(now_Layer, 30, tol);
				upSample(now_Layer);
				performGaussSeidelIteration(now_Layer, 5, tol);

#endif

#ifdef SETTING3
				//performCGIteration(now_Layer, 20, tol);
				//downSample(now_Layer);
				//downSample(now_Layer);
				//performCGIteration(now_Layer, 20, tol);
				//upSample(now_Layer);
				//upSample(now_Layer);

				performGaussSeidelIteration(now_Layer, 3, tol);
				downSample(now_Layer);
				performGaussSeidelIteration(now_Layer, 3, tol);
				downSample(now_Layer);
				performGaussSeidelIteration(now_Layer, 3, tol);
				//performCGIteration(now_Layer, 20, tol);
				upSample(now_Layer);
				performGaussSeidelIteration(now_Layer, 3, tol);
				upSample(now_Layer);
				performGaussSeidelIteration(now_Layer, 3, tol);

#endif

#ifdef SETTING4
				//performCGIteration(now_Layer, 1, tol);

				//performCGIteration(now_Layer, 5, tol);
				//downSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);
				//downSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);
				//downSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);
				//upSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);
				//upSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);
				//upSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);

				//performGaussSeidelIteration(now_Layer, 1, tol);
				//downSample(now_Layer);
				//performGaussSeidelIteration(now_Layer, 1, tol);
				//downSample(now_Layer);
				//performGaussSeidelIteration(now_Layer, 1, tol);
				//downSample(now_Layer);
				//performGaussSeidelIteration(now_Layer, 1, tol);
				//downSample(now_Layer);
				//performCGIteration(now_Layer, 5, tol);
				////performGaussSeidelIteration(now_Layer, 7, tol);
				//upSample(now_Layer);
				//performGaussSeidelIteration(now_Layer, 1, tol);
				//upSample(now_Layer);
				//performGaussSeidelIteration(now_Layer, 1, tol);
				//upSample(now_Layer);
				//performGaussSeidelIteration(now_Layer, 1, tol);
				//upSample(now_Layer);
				//performGaussSeidelIteration(now_Layer, 1, tol);
				performGaussSeidelIteration(now_Layer, 1, tol);
#endif
#ifdef NO_ORDER
				cublasStatus = cublasSaxpy(cublasHandle, dims[layer], &one, dev_deltaX[layer], 1, dev_X, 1);
#else
				Update_DeltaX_Kernel << <blocksPerGrid, threadsPerBlock >> > (dev_X, dev_deltaX[layer], dev_index2vertex[layer], number);
#endif
				//cudaMemcpy(dev_temp_X[layer], dev_X, sizeof(float)*dims[layer], cudaMemcpyDeviceToDevice);
				//cublasSaxpy(cublasHandle, dims[layer], &minus_one, dev_target_X, 1, dev_temp_X[layer], 1);
				//float metric;
				//cublasSdot(cublasHandle, dims[layer], dev_temp_X[layer], 1, dev_temp_X[layer], 1, &metric);
				//float *temp_vec = new float[dims[layer]];
				//cudaMemcpy(temp_vec, dev_temp_X[layer], sizeof(float)*dims[layer], cudaMemcpyDeviceToHost);
				//cudaDeviceSynchronize();
				//metric = 0;
				//for (int ii = 0; ii < dims[layer]; ii++)
				//	if (metric < abs(temp_vec[ii])) metric = abs(temp_vec[ii]);
				////printf("metric:%.10f\n", sqrt(metric) / number);
				//printf("metric:%.10f\n", metric);
				//delete[] temp_vec;

				//lineSearch(t);
				if (use_WHM)
				{
					if (it <= 10) omega = 1;
					else if (it == 11) omega = 2 / (2 - rho * rho);
					else omega = 4 / (4 - rho * rho*omega);
					float p_1 = 1.0, p_2 = 1.0 - p_1;
					float p_3 = omega, p_4 = 1.0 - omega;
					cublasSscal(cublasHandle, 3 * number, &p_1, dev_X, 1);
					cublasSaxpy(cublasHandle, 3 * number, &p_2, dev_prev_X, 1, dev_X, 1);
					cublasSscal(cublasHandle, 3 * number, &p_3, dev_X, 1);
					cublasSaxpy(cublasHandle, 3 * number, &p_4, dev_prev_prev_X, 1, dev_X, 1);
				}

				cudaDeviceSynchronize();
				now_time += timer.Get_Time();
				printf("\n");
			}
			//
			// Step 3.5: compare dev_temp_X and dev_X
			//cublasSaxpy(cublasHandle, 3 * number, &minus_one, dev_X, 1, dev_temp_X, 1);
			//float rrr;
			//cublasSdot(cublasHandle, 3 * number, dev_temp_X, 1, dev_temp_X, 1, &rrr);
			//printf("error:%f\n", rrr);

			// Step 4: Finalizing update
			cublasStatus = cublasSaxpy(cublasHandle, 3 * number, &minus_one, dev_X, 1, dev_old_X, 1);
			float neg_inv_t = -1.0 / t;
			cublasStatus = cublasSscal(cublasHandle, 3 * number, &neg_inv_t, dev_old_X, 1);
			cublasStatus = cublasScopy(cublasHandle, 3 * number, dev_old_X, 1, dev_V, 1);

			if (enable_benchmark==2)
			{
				cublasStatus = cublasSdot(cublasHandle, 3 * number, dev_R[layer], 1, dev_R[layer], 1, &g_norm);
				fprintf(benchmark, "%f\n", g_norm);
			}

			if(enable_benchmark==1)
				fprintf(benchmark, "\n");

		}
		//Output to main memory for rendering
		cudaMemcpy(X, dev_X, sizeof(TYPE) * 3 * number, cudaMemcpyDeviceToHost);
		cudaMemcpy(V, dev_V, sizeof(TYPE) * 3 * number, cudaMemcpyDeviceToHost);


		cost[cost_ptr] = fps_timer.Get_Time();

		cost_ptr = (cost_ptr + 1) % 8;
		fps = 8 / (cost[0] + cost[1] + cost[2] + cost[3] + cost[4] + cost[5] + cost[6] + cost[7]);

	}

	///////////////////////////////////////////////////////////////////////////////////////////
	//  IO functions
	///////////////////////////////////////////////////////////////////////////////////////////
	void Write(std::fstream &output)
	{
		Write_Binaries(output, X, number * 3);
		Write_Binaries(output, V, number * 3);
	}

	bool Write_File(const char *file_name)
	{
		std::fstream output;
		output.open(file_name, std::ios::out | std::ios::binary);
		if (!output.is_open()) { printf("Error, file not open.\n"); return false; }
		Write(output);
		output.close();
		return true;
	}

	void Read(std::fstream &input)
	{
		Read_Binaries(input, X, number * 3);
		Read_Binaries(input, V, number * 3);
	}

	bool Read_File(const char *file_name)
	{
		std::fstream input;
		input.open(file_name, std::ios::in | std::ios::binary);
		if (!input.is_open()) { printf("Error, file not open.\n");	return false; }
		Read(input);
		input.close();
		return true;
	}
};


#endif