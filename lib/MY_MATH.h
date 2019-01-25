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
//  MY_MATH library.
//  Contents:
//		Min and Max functions, float2integer functions, random functions
//		Vector and matrix functions
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __MY_MATH_H__
#define __MY_MATH_H__

#define MY_PI		3.14159265358979323846f
#define PI_180		0.0174532925f
#define INV_PI		0.31830988618379067154f
#define INV_TWOPI	0.15915494309189533577f
#define MY_INFINITE 999999999
#define INV_255		0.00392156862745098039f
#define EPSILON		1e-7f

#include <string.h>
#include <stdio.h>
#include <math.h>
#define FORCEINLINE inline


#define ADD(a, b, c)	{c[0]=a[0]+b[0]; c[1]=a[1]+b[1]; c[2]=a[2]+b[2];}
#define SUB(a, b, c)	{c[0]=a[0]-b[0]; c[1]=a[1]-b[1]; c[2]=a[2]-b[2];}
#define DOT(a, b)		(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
#define CROSS(a, b, r)	{r[0]=a[1]*b[2]-a[2]*b[1]; r[1]=a[2]*b[0]-a[0]*b[2]; r[2]=a[0]*b[1]-a[1]*b[0];}
#define	MIN(a,b)		((a)<(b)?(a):(b))
#define	MAX(a,b)		((a)>(b)?(a):(b))
#define SQR(a)			((a)*(a))
#define CLAMP(a, l, h)  (((a)>(h))?(h):(((a)<(l))?(l):(a)))
#define SIGN(a)			((a)<0?-1:1)
#define SWAP(X, Y)      {temp={X}; X=(Y); Y=(temp);}

#define Distance2(x, y) ((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]))


///////////////////////////////////////////////////////////////////////////////////////////
//  Min and Max Functions.
///////////////////////////////////////////////////////////////////////////////////////////
template<class T> FORCEINLINE 
T Min(const T a, const T b)
{
	return ((a)<(b)?(a):(b));
}

template<class T> FORCEINLINE 
T Min(const T a, const T b, const T c)
{
	T r=a;
	if(b<r) r=b;
	if(c<r) r=c; 
	return r;
}

template<class T> FORCEINLINE 
T Min(const T a, const T b, const T c, const T d)
{
	T r=a;
	if(b<r) r=b;
	if(c<r) r=c; 
	if(d<r) r=d;
	return r;
}

template<class T> FORCEINLINE 
T Max(const T a, const T b)
{
	return ((a)>(b)?(a):(b));
}

template<class T> FORCEINLINE 
T Max(const T a, const T b, const T c)	
{
	T r=a;
	if(b>r) r=b;
	if(c>r) r=c; 
	return r;
}

template<class T> FORCEINLINE 
T Max(const T a, const T b, const T c, const T d)
{
	T r=a;
	if(b>r) r=b;
	if(c>r) r=c; 
	if(d>r) r=d; 
	return r;
}

template<class T> FORCEINLINE 
T Max(const T a, const T b, const T c, const T d, const T e)
{
	T r=a;
	if(b>r) r=b;
	if(c>r) r=c; 
	if(d>r) r=d;
	if(e>r)	r=e;
	return r;
}

template<class T> FORCEINLINE 
T Max_By_Abs(const T a, const T b)
{
	return fabsf(a)>fabsf(b)?a:b;
}

template<class T> FORCEINLINE
T Min_By_Abs(const T a, const T b)
{
	return fabsf(a)<fabsf(b)?a:b;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Math and Misc Functions.
///////////////////////////////////////////////////////////////////////////////////////////
template<class T> FORCEINLINE 
void Swap(T &a, T &b)
{
	T c=a; a=b; b=c;
}

template<class T> FORCEINLINE 
T Cotangent(const T x0[], const T x1[], const T x2[])
{
	T x10[3], x20[3];
	x10[0]=x1[0]-x0[0];
	x10[1]=x1[1]-x0[1];
	x10[2]=x1[2]-x0[2];
	x20[0]=x2[0]-x0[0];
	x20[1]=x2[1]-x0[1];
	x20[2]=x2[2]-x0[2];

	T dot=DOT(x10, x20);
	return dot/sqrt(DOT(x10, x10)*DOT(x20, x20) - dot*dot);
}

///////////////////////////////////////////////////////////////////////////////////////////
//  2D matrix functions.
///////////////////////////////////////////////////////////////////////////////////////////
template <class T> FORCEINLINE
void Matrix_Inverse_2(T *A, T *R)
{
	T inv_det=1/(A[0]*A[3]-A[1]*A[2]);
	R[0]= A[3]*inv_det;
	R[1]=-A[1]*inv_det;
	R[2]=-A[2]*inv_det;
	R[3]= A[0]*inv_det;
}

template <class T> FORCEINLINE
void Matrix_Transpose_2(T *A, T *R)
{
	memcpy(R, A, sizeof(T)*4);
	Swap(R[1], R[2]);
}

template <class T> FORCEINLINE
void Matrix_Product_2(T *A, T *B, T *R)
{
	T temp_R[4];
	temp_R[0]=A[0]*B[0]+A[1]*B[2];
	temp_R[1]=A[0]*B[1]+A[1]*B[3];
	temp_R[2]=A[2]*B[0]+A[3]*B[2];
	temp_R[3]=A[2]*B[1]+A[3]*B[3];
	R[0]=temp_R[0];
	R[1]=temp_R[1];
	R[2]=temp_R[2];
	R[3]=temp_R[3];
}

template <class T> FORCEINLINE
void Matrix_Vector_Product_2(T *A, T *x, T *r)	//r=A*x
{
	r[0]=A[0]*x[0]+A[1]*x[1];
	r[1]=A[2]*x[0]+A[3]*x[1];	
}

template <class T> FORCEINLINE
void Matrix_T_Product_2(T *A, T *B, T *R)
{
	for(int i=0; i<2; i++)
	for(int j=0; j<2; j++)
		R[i*2+j]=A[i]*B[j]+A[i+2]*B[j+2];	
}

template <class T> FORCEINLINE
void ED_2(T *G, T *w, T *V) // G-> V'w V (eigenvalue decomposition)
{
	T a=1;
	T b=-(G[0]+G[3]);
	T c=G[0]*G[3]-G[1]*G[2];
	T delta=(b*b-4*c);
	if(delta<0)	{delta=0;}
	else		delta=sqrtf(delta);

	w[0]=(-b+delta)*0.5f;
	w[1]=(-b-delta)*0.5f;
	T inv_length;
	a=G[0]-w[0];
	b=G[1];

	if(fabsf(a)<1e-6f && fabsf(b)<1e-6f)
	{
		V[0]=1;
		V[1]=0;
		V[2]=0;
		V[3]=1;
	}
	else
	{
		inv_length=1/sqrtf(a*a+b*b);
		V[0]= b*inv_length;
		V[1]=-a*inv_length;
		V[2]=-V[1];
		V[3]= V[0];
	}
	if(V[0]<0)
	{
		V[0]=-V[0];
		V[1]=-V[1];
		V[2]=-V[2];
		V[3]=-V[3];
	}	
	if(w[0]<0)	w[0]=0;
	if(w[1]<0)	w[1]=0;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  3D vector functions.
///////////////////////////////////////////////////////////////////////////////////////////
template <class T> FORCEINLINE 
T Magnitude(T *x)
{
	return sqrtf(DOT(x, x));
}

template <class T> FORCEINLINE 
T Distance(T x0[], T x1[])
{
	T x[3]={x0[0]-x1[0], x0[1]-x1[1], x0[2]-x1[2]};
	return sqrt(DOT(x,x));
}

template <class T> FORCEINLINE 
T Distance_Squared(T x0[], T x1[])
{
	T x[3]={x0[0]-x1[0], x0[1]-x1[1], x0[2]-x1[2]};
	return DOT(x,x);
}

template <class T> FORCEINLINE 
T Dot(T *v0, T *v1)
{
	return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
}

template <class T> FORCEINLINE 
void Cross(T* a, T* b, T* r)
{
	r[0]=a[1]*b[2]-a[2]*b[1]; 
	r[1]=a[2]*b[0]-a[0]*b[2]; 
	r[2]=a[0]*b[1]-a[1]*b[0];
}

template <class T> FORCEINLINE 
T Normalize(T *x)
{
	T m=Magnitude(x);
	if(m<1e-14f)	return m;//{printf("ERROR: vector cannot be normalized.\n"); return m;}
	T inv_m=1/m;
	x[0]*=inv_m;
	x[1]*=inv_m;
	x[2]*=inv_m;
	return m;
}

template <class T> FORCEINLINE 
T Area_Squared(T* V0, T* V1, T* V2)
{
	T E10[3], E20[3], N[3];
	E10[0]=V1[0]-V0[0];
	E10[1]=V1[1]-V0[1];
	E10[2]=V1[2]-V0[2];
	E20[0]=V2[0]-V0[0];
	E20[1]=V2[1]-V0[1];
	E20[2]=V2[2]-V0[2];
	CROSS(E10, E20, N);
	return DOT(N, N);
}


template <class T> FORCEINLINE
T Normal(T *p0, T *p1, T *p2, T *normal, bool normalized=true)
{
	T e0[3], e1[3];
	for(int i=0; i<3; i++)
	{
		e0[i]=p1[i]-p0[i];
		e1[i]=p2[i]-p0[i];
	}
	CROSS(e0, e1, normal);
	if(normalized)	return Normalize(normal);
	return	0;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  3D matrix functions.
///////////////////////////////////////////////////////////////////////////////////////////
template <class T> FORCEINLINE
T Determinant_3(T x[])
{
	return x[0]*(x[4]*x[8]-x[7]*x[5])+x[3]*(x[7]*x[2]-x[1]*x[8])+x[6]*(x[1]*x[5]-x[4]*x[2]);
}


template <class T> FORCEINLINE
void Matrix_Add_3(T *A, T a, T *B, T b, T *R)	//R=aA+bB
{
	for(int i=0; i<9; i++)
		R[i]=A[i]*a+B[i]*b;
}

template <class T> FORCEINLINE
void Matrix_Add_3(T *A, T *B, T *R)				//R=A+B
{
	for(int i=0; i<9; i++)
		R[i]=A[i]+B[i];
}

template <class T> FORCEINLINE
void Matrix_Substract_3(T *A, T *B, T *R)		//R=A-B
{
	for(int i=0; i<9; i++)
		R[i]=A[i]-B[i];
}

template <class T> FORCEINLINE
T Matrix_Inverse_3(T *A, T *R)				//R=inv(A)
{
	R[0]=A[4]*A[8]-A[7]*A[5];
	R[1]=A[7]*A[2]-A[1]*A[8];
	R[2]=A[1]*A[5]-A[4]*A[2];
	R[3]=A[5]*A[6]-A[3]*A[8];
	R[4]=A[0]*A[8]-A[2]*A[6];
	R[5]=A[2]*A[3]-A[0]*A[5];
	R[6]=A[3]*A[7]-A[4]*A[6];
	R[7]=A[1]*A[6]-A[0]*A[7];
	R[8]=A[0]*A[4]-A[1]*A[3];
	T det=A[0]*R[0]+A[3]*R[1]+A[6]*R[2];
	T inv_det=1/det;
	for(int i=0; i<9; i++)	R[i]*=inv_det;
	return det;
}

template <class T> FORCEINLINE					//R=A'
void Matrix_Transpose_3(T *A, T *R)
{
	if(R!=A) memcpy(R, A, sizeof(T)*9);
	Swap(R[1], R[3]);
	Swap(R[2], R[6]);
	Swap(R[5], R[7]);
}

template <class T> FORCEINLINE
void Matrix_Factorization_3(T *A, T *R)			//R=chol(A), Chelosky factorization
{
	R[0]=sqrtf(A[0]);
	R[1]=A[1]/R[0];
	R[2]=A[2]/R[0];
	R[3]=0;
	R[4]=sqrtf(A[4]-R[1]*R[1]);
	R[5]=(A[5]-R[1]*R[2])/R[4];
	R[6]=0;
	R[7]=0;
	R[8]=sqrtf(A[8]-R[2]*R[2]-R[5]*R[5]);
}

template <class T> FORCEINLINE
void Matrix_Product_3(T *A, T *B, T *R)		//R=A*B
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

template <class T> FORCEINLINE
void Matrix_T_Product_3(T *A, T *B, T *R)	//R=A'*B
{

	R[0]=A[0]*B[0]+A[3]*B[3]+A[6]*B[6];
	R[1]=A[0]*B[1]+A[3]*B[4]+A[6]*B[7];
	R[2]=A[0]*B[2]+A[3]*B[5]+A[6]*B[8];
	R[3]=A[1]*B[0]+A[4]*B[3]+A[7]*B[6];
	R[4]=A[1]*B[1]+A[4]*B[4]+A[7]*B[7];
	R[5]=A[1]*B[2]+A[4]*B[5]+A[7]*B[8];
	R[6]=A[2]*B[0]+A[5]*B[3]+A[8]*B[6];
	R[7]=A[2]*B[1]+A[5]*B[4]+A[8]*B[7];
	R[8]=A[2]*B[2]+A[5]*B[5]+A[8]*B[8];
}

template <class T> FORCEINLINE
void Matrix_Product_T_3(T *A, T *B, T *R)	//R=A*B'
{
    R[0]=A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
    R[1]=A[0]*B[3]+A[1]*B[4]+A[2]*B[5];
    R[2]=A[0]*B[6]+A[1]*B[7]+A[2]*B[8];
    R[3]=A[3]*B[0]+A[4]*B[1]+A[5]*B[2];
    R[4]=A[3]*B[3]+A[4]*B[4]+A[5]*B[5];
    R[5]=A[3]*B[6]+A[4]*B[7]+A[5]*B[8];
    R[6]=A[6]*B[0]+A[7]*B[1]+A[8]*B[2];
    R[7]=A[6]*B[3]+A[7]*B[4]+A[8]*B[5];
    R[8]=A[6]*B[6]+A[7]*B[7]+A[8]*B[8];
}

template <class T> FORCEINLINE
void Matrix_Vector_Product_3(T *A, T *x, T *r)	//r=A*x
{
	r[0]=A[0]*x[0]+A[1]*x[1]+A[2]*x[2];
	r[1]=A[3]*x[0]+A[4]*x[1]+A[5]*x[2];
	r[2]=A[6]*x[0]+A[7]*x[1]+A[8]*x[2];
}

template <class T> FORCEINLINE
void Matrix_T_Vector_Product_3(T *A, T *x, T *r)//r=A'*x
{
	r[0]=A[0]*x[0]+A[3]*x[1]+A[6]*x[2];
	r[1]=A[1]*x[0]+A[4]*x[1]+A[7]*x[2];
	r[2]=A[2]*x[0]+A[5]*x[1]+A[8]*x[2];
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Arbitrary Dimensional Functions.
///////////////////////////////////////////////////////////////////////////////////////////
template <class T> 
void Matrix_Product_4(T *a, T *b, T *r)
{
	//r=a*b
	T temp[16];
	memset(temp, 0, sizeof(T)*16);
	for(int i=0; i<4; i++)	for(int j=0; j<4; j++)	for(int k=0; k<4; k++)
		temp[i*4+j]+=a[i*4+k]*b[k*4+j];
	
	memcpy(r, temp, sizeof(T)*16);
}

template <class T> 
void Matrix_Vector3_Product_4(T *A, T *x, T *r)
{
	r[0]=A[0]*x[0]+A[1]*x[1]+A[ 2]*x[2]+A[ 3];
	r[1]=A[4]*x[0]+A[5]*x[1]+A[ 6]*x[2]+A[ 7];
	r[2]=A[8]*x[0]+A[9]*x[1]+A[10]*x[2]+A[11];
}

template <class T> 
void Matrix_Vector_Product_4(T *A, T *x, T *r)
{
	r[0]=A[ 0]*x[0]+A[ 1]*x[1]+A[ 2]*x[2]+A[ 3]*x[3];
	r[1]=A[ 4]*x[0]+A[ 5]*x[1]+A[ 6]*x[2]+A[ 7]*x[3];
	r[2]=A[ 8]*x[0]+A[ 9]*x[1]+A[10]*x[2]+A[11]*x[3];
	r[3]=A[12]*x[0]+A[13]*x[1]+A[14]*x[2]+A[15]*x[3];
}

template <class T> FORCEINLINE					//R=A'
void Matrix_Transpose_4(T *A, T *R)
{
	if(R!=A) memcpy(R, A, sizeof(T)*16);
	Swap(R[1], R[4]);
	Swap(R[2], R[8]);
	Swap(R[3], R[12]);
	Swap(R[6], R[9]);
	Swap(R[7], R[13]);
	Swap(R[11], R[14]);
}

template <class T> FORCEINLINE T 
Norm(T *x, int number=3)	//infinite norm
{
	T ret=0;
	for(int i=0; i<number; i++)	
		if(ret<fabsf(x[i]))	ret=fabsf(x[i]);
	return ret;
}

template <class T> FORCEINLINE T 
Dot(T *x, T *y, int number)
{
	T ret=0;
	for(int i=0; i<number; i++)	ret+=x[i]*y[i];
	return ret;
}

template <class T> FORCEINLINE
void Matrix_Transpose(T *A, T *R, int nx, int ny)				//R=A'
{
	for(int i=0; i<nx; i++)
	for(int j=0; j<ny; j++)
		R[j*nx+i]=A[i*ny+j];
}


template <class T> FORCEINLINE
void Matrix_Product(T *A, T *B, T *R, int nx, int ny, int nz)	//R=A*B
{
	memset(R, 0, sizeof(T)*nx*nz);
	for(int i=0; i<nx; i++)
	for(int j=0; j<nz; j++)
	for(int k=0; k<ny; k++)
        R[i*nz+j]+=A[i*ny+k]*B[k*nz+j];
}

template <class T> FORCEINLINE
void Matrix_Product_T(T *A, T *B, T *R, int nx, int ny, int nz)	//R=A*B'
{
    memset(R, 0, sizeof(T)*nx*nz);
    for(int i=0; i<nx; i++)
    for(int j=0; j<nz; j++)
    for(int k=0; k<ny; k++)
        R[i*nz+j]+=A[i*ny+k]*B[j*ny+k];
}

template <class T> FORCEINLINE
T Matrix_Product_T(T *A, T *B, int nx, int ny, int nz, int i, int j)	//R=A*B'
{
    T ret=0;
    for(int k=0; k<ny; k++)
        ret+=A[i*ny+k]*B[j*ny+k];
    return ret;
}

template <class T> FORCEINLINE
void Matrix_Self_Product(T *A, T *R, int nx, int ny)			//R=A'*A
{
	memset(R, 0, sizeof(T)*ny*ny);
	for(int i=0; i<ny; i++)
	for(int j=i; j<ny; j++)
	{
		for(int k=0; k<nx; k++)
			R[i*ny+j]+=A[k*ny+i]*A[k*ny+j];
		if(i!=j)	R[j*ny+i]=R[i*ny+j];		
	}
}

template <class T>
void Gaussian_Elimination(T *a, int n, T *b)
{
	int* indxc=new int[n];
	int* indxr=new int[n];
	int* ipiv =new int[n];
	int i,icol,irow,j,k,l,ll;
	T big,dum,pivinv,temp;

	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) 
	{ 
		big=0.0;	
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) 
				{
					if (ipiv[k] ==0) 
					{
						if (fabs(a[j*n+k]) >= big) 
						{
							big=fabs(a[j*n+k]);
							irow=j;
							icol=k;
						}
					}
				}
		++(ipiv[icol]);

		if (irow != icol) 
		{
			for (l=0;l<n;l++) {temp=a[irow*n+l]; a[irow*n+l]=a[icol*n+l]; a[icol*n+l]=temp;}
			temp=b[irow]; b[irow]=b[icol]; b[icol]=temp;
		}
		indxr[i]=irow; 
		indxc[i]=icol; 
		if (a[icol*n+icol] == 0.0) printf("Error: Singular Matrix in Gaussian_Elimination.");
		pivinv=1.0/a[icol*n+icol];
		a[icol*n+icol]=1.0;
		for (l=0;l<n;l++) a[icol*n+l] *= pivinv;
		b[icol] *= pivinv;

		for (ll=0;ll<n;ll++) 
			if (ll != icol) 
			{
				dum=a[ll*n+icol];
				a[ll*n+icol]=0.0;
				for (l=0;l<n;l++) a[ll*n+l] -= a[icol*n+l]*dum;
				b[ll] -= b[icol]*dum;
			}
	}

	for (l=n-1;l>1;l--) 
	{
		if (indxr[l] != indxc[l])
		for (k=0;k<n;k++)
		{
			temp=a[k*n+indxr[l]];
			a[k*n+indxr[l]]=a[k*n+indxc[l]];
			a[k*n+indxc[l]]=temp;
		}
	} 
	delete []ipiv;
	delete []indxr;
	delete []indxc;
}


///////////////////////////////////////////////////////////////////////////////////////////
//  SVD function <from numerical recipes in C++>
//		Given a matrix a[1..m][1..n], this routine computes its singular value
//		decomposition, A = U.W.VT.  The matrix U replaces a on output.  The diagonal
//		matrix of singular values W is output as a vector w[1..n].  The matrix V (not
//		the transpose VT) is output as v[1..n][1..n].
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE> FORCEINLINE
TYPE pythag(TYPE a, TYPE b)
{
	TYPE at = fabs(a), bt = fabs(b), ct, result;
	if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}	

template <class TYPE>
void SVD(TYPE u[], int m, int n, TYPE w[], TYPE v[])
{
	bool flag;
	int i,its,j,jj,k,l,nm;
	TYPE anorm,c,f,g,h,s,scale,x,y,z;
	TYPE *rv1=new TYPE[n];
	g = scale = anorm = 0.0; //Householder reduction to bidiagonal form.
	for(i=0;i<n;i++) 
	{
		l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i < m) 
		{
			for (k=i;k<m;k++) scale += fabs(u[k*n+i]);
			if (scale != 0.0) 
			{
				for (k=i;k<m;k++) 
				{
					u[k*n+i] /= scale;
					s += u[k*n+i]*u[k*n+i];
				}
				f=u[i*n+i];
				g = -sqrt(s)*SIGN(f);
				h=f*g-s;
				u[i*n+i]=f-g;
				for (j=l;j<n;j++) 
				{
					for (s=0.0,k=i;k<m;k++) s += u[k*n+i]*u[k*n+j];
					f=s/h;
					for (k=i;k<m;k++) u[k*n+j] += f*u[k*n+i];
				}
				for (k=i;k<m;k++) u[k*n+i] *= scale;
			}
		}
		w[i]=scale *g;
		g=s=scale=0.0;
		if(i+1 <= m && i+1 != n) 
		{
			for(k=l;k<n;k++) scale += fabs(u[i*n+k]);
			if(scale != 0.0) 
			{
				for (k=l;k<n;k++) 
				{
					u[i*n+k] /= scale;
					s += u[i*n+k]*u[i*n+k];
				}
				f=u[i*n+l];
				g = -sqrt(s)*SIGN(f);
				h=f*g-s;
				u[i*n+l]=f-g;
				for (k=l;k<n;k++) rv1[k]=u[i*n+k]/h;
				for (j=l;j<m;j++) 
				{
					for (s=0.0,k=l;k<n;k++) s += u[j*n+k]*u[i*n+k];
					for (k=l;k<n;k++) u[j*n+k] += s*rv1[k];
				}
				for (k=l;k<n;k++) u[i*n+k] *= scale;
			}
		}
		anorm=MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
	}
	for(i=n-1;i>=0;i--) 
	{ //Accumulation of right-hand transformations.
		if (i < n-1) 
		{
			if (g != 0.0) 
			{
				for (j=l;j<n;j++) //Double division to avoid possible under
					v[j*n+i]=(u[i*n+j]/u[i*n+l])/g;
				for (j=l;j<n;j++) 
				{
					for (s=0.0,k=l;k<n;k++) s += u[i*n+k]*v[k*n+j];
					for (k=l;k<n;k++) v[k*n+j] += s*v[k*n+i];
				}
			}
			for (j=l;j<n;j++) v[i*n+j]=v[j*n+i]=0.0;
		}
		v[i*n+i]=1.0;
		g=rv1[i];
		l=i;
	}
	for(i=MIN(m,n)-1;i>=0;i--) 
	{	//Accumulation of left-hand transformations.
		l=i+1;
		g=w[i];
		for (j=l;j<n;j++) u[i*n+j]=0.0;
		if (g != 0.0) 
		{
			g=1.0/g;
			for (j=l;j<n;j++) 
			{
				for (s=0.0,k=l;k<m;k++) s += u[k*n+i]*u[k*n+j];
				f=(s/u[i*n+i])*g;
				for (k=i;k<m;k++) u[k*n+j] += f*u[k*n+i];
			}
			for (j=i;j<m;j++) u[j*n+i] *= g;
		} 
		else for (j=i;j<m;j++) u[j*n+i]=0.0;
		++u[i*n+i];
	}
	
	for(k=n-1;k>=0;k--) 
	{ //Diagonalization of the bidiagonal form: Loop over
		for (its=0;its<30;its++) 
		{ //singular values, and over allowed iterations.
			flag=true;
			for (l=k;l>=0;l--) 
			{ //Test for splitting.
				nm=l-1;
				if ((TYPE)(fabs(rv1[l])+anorm) == anorm) 
				{
					flag=false;
					break;
				}
				if ((TYPE)(fabs(w[nm])+anorm) == anorm) break;
			}
			if(flag) 
			{
				c=0.0; //Cancellation of rv1[l], if l > 0.
				s=1.0;
				for (i=l;i<k+1;i++) 
				{
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if ((TYPE)(fabs(f)+anorm) == anorm) break;
					g=w[i];
					h=pythag(f,g);
					w[i]=h;
					h=1.0/h;
					c=g*h;
					s = -f*h;
					for (j=0;j<m;j++) 
					{
						y=u[j*n+nm];
						z=u[j*n+i];
						u[j*n+nm]=y*c+z*s;
						u[j*n+i]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if (l == k) 
			{	// Convergence.
				if (z < 0.0) 
				{ //Singular value is made nonnegative.
					w[k] = -z;
					for (j=0;j<n;j++) v[j*n+k] = -v[j*n+k];
				}
				break;
			}
			if (its == 29) {printf("Error: no convergence in 30 svdcmp iterations");getchar();}
			x=w[l]; //Shift from bottom 2-by-2 minor.
			nm=k-1;
			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=pythag(f,1.0);
			f=((x-z)*(x+z)+h*((y/(f+fabs(g)*SIGN(f)))-h))/x;
			c=s=1.0; //Next QR transformation:
			for (j=l;j<=nm;j++) 
			{
				i=j+1;
				g=rv1[i];
				y=w[i];
				h=s*g;
				g=c*g;
				z=pythag(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g=g*c-x*s;
				h=y*s;
				y *= c;
				for (jj=0;jj<n;jj++) 
				{
					x=v[jj*n+j];
					z=v[jj*n+i];
					v[jj*n+j]=x*c+z*s;
					v[jj*n+i]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j]=z; //Rotation can be arbitrary if z D 0.
				if (z) 
				{
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=0;jj<m;jj++) 
				{
					y=u[jj*n+j];
					z=u[jj*n+i];
					u[jj*n+j]=y*c+z*s;
					u[jj*n+i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
	delete []rv1;
}



template <class TYPE>
void SVD3(TYPE u[3][3], TYPE w[3], TYPE v[3][3])
{
	
	TYPE	anorm,c,f,g,h,s,scale;
	TYPE	x,y,z;
	TYPE	rv1[3];
	g = scale = anorm = 0.0; //Householder reduction to bidiagonal form.

	for(int i=0; i<3; i++) 
	{
		int l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if(i<3) 
		{
			for(int k=i; k<3; k++) scale += fabsf(u[k][i]);
			if(scale!=0) 
			{
				for(int k=i; k<3; k++) 
				{
					u[k][i]/=scale;
					s+=u[k][i]*u[k][i];
				}
				f=u[i][i];
				g=-sqrtf(s)*SIGN(f);
				h=f*g-s;
				u[i][i]=f-g;
				for(int j=l; j<3; j++) 
				{
					s=0;
					for(int k=i;k<3;k++)	s+=u[k][i]*u[k][j];
					f=s/h;
					for(int k=i; k<3; k++)	u[k][j]+=f*u[k][i];
				}
				for(int k=i; k<3; k++)		u[k][i]*=scale;
			}
		}
		w[i]=scale*g;

		g=s=scale=0.0;
		if(i<=2 && i!=2) 
		{
			for(int k=l; k<3; k++)	scale+=fabsf(u[i][k]);
			if(scale!=0) 
			{
				for(int k=l; k<3; k++) 
				{
					u[i][k]/=scale;
					s+=u[i][k]*u[i][k];
				}
				f=u[i][l];
				g=-sqrtf(s)*SIGN(f);
				h=f*g-s;
				u[i][l]=f-g;
				for(int k=l; k<3; k++) rv1[k]=u[i][k]/h;
				for(int j=l; j<3; j++) 
				{
					s=0;
					for(int k=l; k<3; k++)	s+=u[j][k]*u[i][k];
					for(int k=l; k<3; k++)	u[j][k]+=s*rv1[k];
				}
				for(int k=l; k<3; k++) u[i][k]*=scale;
			}
		}
		anorm=MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
	}
	
	for(int i=2, l; i>=0; i--) //Accumulation of right-hand transformations.
	{ 
		if(i<2) 
		{
			if(g!=0) 
			{
				for(int j=l; j<3; j++) //Double division to avoid possible under
					v[j][i]=(u[i][j]/u[i][l])/g;
				for(int j=l; j<3; j++) 
				{
					s=0;
					for(int k=l; k<3; k++)	s+=u[i][k]*v[k][j];
					for(int k=l; k<3; k++)	v[k][j]+=s*v[k][i];
				}
			}
			for(int j=l; j<3; j++)	v[i][j]=v[j][i]=0.0;
		}
		v[i][i]=1.0;
		g=rv1[i];
		l=i;
	}

	for(int i=2; i>=0; i--) //Accumulation of left-hand transformations.
	{	
		int l=i+1;
		g=w[i];
		for(int j=l; j<3; j++) u[i][j]=0;
		if(g!=0)
		{
			g=1/g;
			for(int j=l; j<3; j++) 
			{
				s=0;
				for(int k=l; k<3; k++)	s+=u[k][i]*u[k][j];
				f=(s/u[i][i])*g;
				for(int k=i; k<3; k++)	u[k][j]+=f*u[k][i];
			}
			for(int j=i; j<3; j++)		u[j][i]*=g;
		} 
		else for(int j=i; j<3; j++)		u[j][i]=0.0;
		u[i][i]++;
	}

	for(int k=2; k>=0; k--)				//Diagonalization of the bidiagonal form: Loop over
	{ 
		for(int its=0; its<30; its++)	//singular values, and over allowed iterations.
		{ 
			bool flag=true;
			int  l;
			int	 nm;
			for(l=k; l>=0; l--)			//Test for splitting.
			{ 
				nm=l-1;
				if((TYPE)(fabs(rv1[l])+anorm)==anorm) 
				{
					flag=false;
					break;
				}
				if((TYPE)(fabs(w[nm])+anorm)==anorm)	break;
			}
			if(flag)
			{
				c=0.0; //Cancellation of rv1[l], if l > 0.
				s=1.0;
				for(int i=l; i<k+1; i++) 
				{
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if((TYPE)(fabs(f)+anorm) == anorm) break;
					g=w[i];
					h=pythag(f,g);
					w[i]=h;
					h=1.0/h;
					c= g*h;
					s=-f*h;
					for(int j=0; j<3; j++) 
					{
						y=u[j][nm];
						z=u[j][i ];
						u[j][nm]=y*c+z*s;
						u[j][i ]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if(l==k)		// Convergence.
			{	
				if(z<0.0)	// Singular value is made nonnegative.
				{ 
					w[k]=-z;
					for(int j=0; j<3; j++) v[j][k]=-v[j][k];
				}
				break;
			}
			if(its==29) {printf("Error: no convergence in 30 svdcmp iterations");getchar();}
			x=w[l]; //Shift from bottom 2-by-2 minor.
			nm=k-1;
			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=pythag(f, (TYPE)1.0);
			f=((x-z)*(x+z)+h*((y/(f+fabs(g)*SIGN(f)))-h))/x;
			c=s=1.0; //Next QR transformation:
			
			for(int j=l; j<=nm; j++) 
			{
				int i=j+1;
				g=rv1[i];
				y=w[i];
				h=s*g;
				g=c*g;
				z=pythag(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g=g*c-x*s;
				h=y*s;
				y*=c;
				for(int jj=0; jj<3; jj++) 
				{
					x=v[jj][j];
					z=v[jj][i];
					v[jj][j]=x*c+z*s;
					v[jj][i]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j]=z; //Rotation can be arbitrary if z D 0.
				if(z) 
				{
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for(int jj=0; jj<3; jj++) 
				{
					y=u[jj][j];
					z=u[jj][i];
					u[jj][j]=y*c+z*s;
					u[jj][i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
}


///////////////////////////////////////////////////////////////////////////////////////////
//  SVD3x3 Fast SVD of a 3x3 matrix, based on Sifakis's paper
///////////////////////////////////////////////////////////////////////////////////////////
#define GAMMA   5.82842712475
#define C_STAR  0.92387953251
#define S_STAR  0.38268343236

template <class TYPE>
void Quaternion_to_Rotation(TYPE q[4], TYPE R[3][3])
{
    TYPE q00=q[0]*q[0];
    TYPE q11=q[1]*q[1];
    TYPE q22=q[2]*q[2];
    TYPE q33=q[3]*q[3];
    TYPE q01=q[0]*q[1];
    TYPE q02=q[0]*q[2];
    TYPE q03=q[0]*q[3];
    TYPE q12=q[1]*q[2];
    TYPE q13=q[1]*q[3];
    TYPE q23=q[2]*q[3];
    R[0][0]=q33+q00-q11-q22;
    R[1][1]=q33-q00+q11-q22;
    R[2][2]=q33-q00-q11+q22;
    R[0][1]=2*(q01-q23);
    R[1][0]=2*(q01+q23);
    R[0][2]=2*(q02+q13);
    R[2][0]=2*(q02-q13);
    R[1][2]=2*(q12-q03);
    R[2][1]=2*(q12+q03);
}


template <class TYPE>
void SVD3x3(TYPE A[3][3], TYPE U[3][3], TYPE S[3], TYPE q[4], TYPE V[3][3], int iterations=8)
{
    //Part 1: Compute q for V
    TYPE B[3][3];
    Quaternion_to_Rotation(q, V);
    Matrix_Product_3(&A[0][0], &V[0][0], &U[0][0]);
    Matrix_T_Product_3(&U[0][0], &U[0][0], &B[0][0]);
    
    TYPE c_h, s_h, omega;
    TYPE c, s, n;
    bool b;
    TYPE t0, t1, t2, t3, t4, t5;
    TYPE cc, cs, ss, cn, sn;
    TYPE temp_q[4];
    for(int l=0; l<iterations; l++)
    {
        //if(Max(fabs(B[0][1]), fabs(B[0][2]), fabs(B[1][2]))<1e-10f) break;
        
        int i, j, k;
        i=fabs(B[0][1])>fabs(B[0][2])?0:2;
        i=(fabs(B[0][1])>fabs(B[0][2])?fabs(B[0][1]):fabs(B[0][2]))>fabs(B[1][2])?i:1;
        j=(i+1)%3;
        k=3-i-j;
        
        c_h=2*(B[i][i]-B[j][j]);
        s_h=B[i][j];
        b=GAMMA*s_h*s_h<c_h*c_h;
        
        //printf("first c_h: %f; s_h: %f (%f)\n", c_h, s_h, fabs(c_h)+fabs(s_h));
        omega=1.0/(fabs(c_h)+fabs(s_h));
        
        c_h=b?omega*c_h:C_STAR;
        s_h=b?omega*s_h:S_STAR;
        //printf("c_h: %f; s_h: %f (%f)\n", c_h, s_h, omega);
        
        t0=c_h*c_h;
        t1=s_h*s_h;
        n=t0+t1;
        c=t0-t1;
        s=2*c_h*s_h;
        
        //Q[0][0]=c;
        //Q[1][1]=c;
        //Q[2][2]=n;
        //Q[0][1]=-s;
        //Q[1][0]=s;
        //printf("CSN: %f, %f, %f\n", c, s, n);
        //printf("N: %f\n", n);
        //printf("inv: %f\n", inv_length);
        cc=c*c;
        cs=c*s;
        ss=s*s;
        cn=c*n;
        sn=s*n;
        t0=cc*B[i][i]+2*cs*B[i][j]+ss*B[j][j];
        t1=ss*B[i][i]-2*cs*B[i][j]+cc*B[j][j];
        t2=cs*(B[j][j]-B[i][i])+(cc-ss)*B[i][j];
        t3= cn*B[i][k]+sn*B[j][k];
        t4=-sn*B[i][k]+cn*B[j][k];
        t5=B[k][k]*n*n;
        
        B[i][i]=t0;
        B[j][j]=t1;
        B[i][j]=B[j][i]=t2;
        B[i][k]=B[k][i]=t3;
        B[j][k]=B[k][j]=t4;
        B[k][k]=t5;
        
        //Update q
        temp_q[i]=c_h*q[i]+q[j]*s_h;
        temp_q[j]=c_h*q[j]-q[i]*s_h;
        temp_q[k]=c_h*q[k]+q[3]*s_h;
        temp_q[3]=c_h*q[3]-q[k]*s_h;
        memcpy(q, temp_q, sizeof(TYPE)*4);
    }
    
    //part 2: normalize q and obtain V
    TYPE inv_q_length2=1.0/(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    TYPE inv_q_length=sqrt(inv_q_length2);
    q[0]*=inv_q_length;
    q[1]*=inv_q_length;
    q[2]*=inv_q_length;
    q[3]*=inv_q_length;
    S[0]=sqrt(B[0][0])*inv_q_length2;
    S[1]=sqrt(B[1][1])*inv_q_length2;
    S[2]=sqrt(B[2][2])*inv_q_length2;
    
    Quaternion_to_Rotation(q, V);
    
    //Part 3: fix negative S
    int i;
    i=fabs(S[0])<fabs(S[1])?0:1;
    i=(fabs(S[0])<fabs(S[1])?fabs(S[0]):fabs(S[1]))<fabs(S[2])?i:2;
    if(A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])+A[1][0]*(A[2][1]*A[0][2]-A[0][1]*A[2][2])+A[2][0]*(A[0][1]*A[1][2]-A[1][1]*A[0][2])<0)
        S[i]=-S[i];
    
    //Part 4: obtain U
    TYPE rate;
    Matrix_Product_3(&A[0][0], &V[0][0], &U[0][0]);
    rate=1/S[0];
    U[0][0]*=rate;
    U[1][0]*=rate;
    U[2][0]*=rate;
    rate=1/S[1];
    U[0][1]*=rate;
    U[1][1]*=rate;
    U[2][1]*=rate;
    rate=1/S[2];
    U[0][2]*=rate;
    U[1][2]*=rate;
    U[2][2]*=rate;

}

#undef GAMMA
#undef C_STAR
#undef S_STAR


///////////////////////////////////////////////////////////////////////////////////////////
//  Color Spectrum Functions.
///////////////////////////////////////////////////////////////////////////////////////////
template <class T> FORCEINLINE
void Spectrum(T i, T *v, T min=-1, T max=1)
  {
	i=(i-min)/(max-min);
	if(i>=1)			{ v[0]=1;		v[1]=0;		v[2]=0;		}
	else if(i>=0.75)	{ v[0]=1;		v[1]=4-4*i;	v[2]=0;		}
	else if(i>=0.5)		{ v[0]=4*i-2;	v[1]=1;		v[2]=0;		}
	else if(i>=0.25)	{ v[0]=0;		v[1]=1;		v[2]=2-4*i;	}
	else if(i>=0)		{ v[0]=0;		v[1]=4*i;	v[2]=1;		}
	else				{ v[0]=0;		v[1]=0;		v[2]=1;		}
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Integer Functions.
///////////////////////////////////////////////////////////////////////////////////////////
#if (defined(__linux__) && defined(__i386__)) || defined(WIN32)
#define FAST_INT 1
#endif
#define _doublemagicroundeps		(.5-1.4e-11)	//almost .5f = .5f - 1e^(number of exp bit)

FORCEINLINE
int Round(double val) 
{
#ifdef FAST_INT
#define _doublemagic				double(6755399441055744.0)	//2^52 * 1.5,  uses limited precision to floor
	val		= val + _doublemagic;
	return ((long*)&val)[0];
#undef _doublemagic
#else
	return int (val+_doublemagicroundeps);
#endif
}

FORCEINLINE 
int Float_to_Int(double val) 
{
#ifdef FAST_INT
	return (val<0) ?  Round(val+_doublemagicroundeps) :
	Round(val-_doublemagicroundeps);
#else
	return (int)val;
#endif
}

FORCEINLINE 
int Floor(double val) 
{
#ifdef FAST_INT
	return Round(val - _doublemagicroundeps);
#else
	return (int)floorf(val);
#endif
}

FORCEINLINE 
int Ceiling(double val) 
{
#ifdef FAST_INT
	return Round(val + _doublemagicroundeps);
#else
	return (int)ceilf(val);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Pseudo Random Functions. RandomFloat and RandomUInt
///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//	are met:
//		1.	Redistributions of source code must retain the above copyright
//			notice, this list of conditions and the following disclaimer.
//		2.	Redistributions in binary form must reproduce the above copyright
//			notice, this list of conditions and the following disclaimer in the
//			documentation and/or other materials provided with the distribution.
//		3.	The names of its contributors may not be used to endorse or promote
//			products derived from this software without specific prior written
//			permission.
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */
static unsigned long mt[N];		/* the array for the state vector  */
static int mti=N+1;				/* mti==N+1 means mt[N] is not initialized */

FORCEINLINE 
void init_genrand(unsigned long seed) 
{
	mt[0]= seed & 0xffffffffUL;
	for (mti=1; mti<N; mti++) {
		mt[mti] =
		(1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
		/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
		/* In the previous versions, MSBs of the seed affect   */
		/* only MSBs of the array mt[].                        */
		/* 2002/01/09 modified by Makoto Matsumoto             */
		mt[mti] &= 0xffffffffUL;
		/* for >32 bit machines */
	}
}
FORCEINLINE 
unsigned long genrand_int32(void)
{
	unsigned long y;
	static unsigned long mag01[2]={0x0UL, MATRIX_A};
	/* mag01[x] = x * MATRIX_A  for x=0,1 */

	if (mti >= N) { /* generate N words at one time */
		int kk;

		if (mti == N+1)   /* if init_genrand() has not been called, */
			init_genrand(5489UL); /* default initial seed */

		for (kk=0;kk<N-M;kk++) {
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		for (;kk<N-1;kk++) {
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
		mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

		mti = 0;
	}
	y = mt[mti++];
	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);
	return y;
}

FORCEINLINE 
float RandomFloat()			/* generates a random number on [0,1)-real-interval */
{
	return genrand_int32()*((float)1.0/(float)4294967296.0);	/* divided by 2^32 */
}

FORCEINLINE 
float RandomFloat2()		/* generates a random number on [0,1]-real-interval */
{
	return genrand_int32()*((float)1.0/(float)4294967295.0);	/* divided by 2^32-1 */
}

FORCEINLINE 
unsigned long RandomUInt() 
{
	return genrand_int32();
}

#undef N
#undef M
#undef MATRIX_A 
#undef UPPER_MASK
#undef LOWER_MASK


#endif
