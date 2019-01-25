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
//  Class PROJECTIVE_TET_MESH
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef	__WHMIN_PROJECTIVE_TET_MESH_H__
#define __WHMIN_PROJECTIVE_TET_MESH_H__
#include <Eigen/Sparse>
#include "TET_MESH.h"
#include "TIMER.h"


///////////////////////////////////////////////////////////////////////////////////////////
//  Get the Rotation part using a fast anlytic method
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
void Get_Rotation(TYPE F[3][3], TYPE R[3][3])
{
    TYPE C[3][3];
    memset(&C[0][0], 0, sizeof(TYPE)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        C[i][j]+=F[k][i]*F[k][j];
    
    TYPE C2[3][3];
    memset(&C2[0][0], 0, sizeof(TYPE)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        C2[i][j]+=C[i][k]*C[j][k];
    
    TYPE det    =   F[0][0]*F[1][1]*F[2][2]+
                    F[0][1]*F[1][2]*F[2][0]+
                    F[1][0]*F[2][1]*F[0][2]-
                    F[0][2]*F[1][1]*F[2][0]-
                    F[0][1]*F[1][0]*F[2][2]-
                    F[0][0]*F[1][2]*F[2][1];
    
    TYPE I_c    =   C[0][0]+C[1][1]+C[2][2];
    TYPE I_c2   =   I_c*I_c;
    TYPE II_c   =   0.5*(I_c2-C2[0][0]-C2[1][1]-C2[2][2]);
    TYPE III_c  =   det*det;
    TYPE k      =   I_c2-3*II_c;
    
    TYPE inv_U[3][3];
    if(k<1e-10f)
    {
        TYPE inv_lambda=1/sqrt(I_c/3);
        memset(inv_U, 0, sizeof(TYPE)*9);
        inv_U[0][0]=inv_lambda;
        inv_U[1][1]=inv_lambda;
        inv_U[2][2]=inv_lambda;
    }
    else
    {
        TYPE l = I_c*(I_c*I_c-4.5*II_c)+13.5*III_c;
        TYPE k_root = sqrt(k);
        TYPE value=l/(k*k_root);
        if(value<-1.0) value=-1.0;
        if(value> 1.0) value= 1.0;
        TYPE phi = acos(value);
        TYPE lambda2=(I_c+2*k_root*cos(phi/3))/3.0;
        TYPE lambda=sqrt(lambda2);
        
        TYPE III_u = sqrt(III_c);
        if(det<0)   III_u=-III_u;
        TYPE I_u = lambda + sqrt(-lambda2 + I_c + 2*III_u/lambda);
        TYPE II_u=(I_u*I_u-I_c)*0.5;

        
        TYPE U[3][3];
        TYPE inv_rate, factor;
        
        inv_rate=1/(I_u*II_u-III_u);
        factor=I_u*III_u*inv_rate;
        
        memset(U, 0, sizeof(TYPE)*9);
        U[0][0]=factor;
        U[1][1]=factor;
        U[2][2]=factor;
        
        factor=(I_u*I_u-II_u)*inv_rate;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            U[i][j]+=factor*C[i][j]-inv_rate*C2[i][j];
        
        inv_rate=1/III_u;
        factor=II_u*inv_rate;
        memset(inv_U, 0, sizeof(TYPE)*9);
        inv_U[0][0]=factor;
        inv_U[1][1]=factor;
        inv_U[2][2]=factor;
        
        factor=-I_u*inv_rate;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            inv_U[i][j]+=factor*U[i][j]+inv_rate*C[i][j];
    }
    
    memset(&R[0][0], 0, sizeof(TYPE)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        R[i][j]+=F[i][k]*inv_U[k][j];
}



///////////////////////////////////////////////////////////////////////////////////////////
//  class PROJECTIVE_TET_MESH
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
class PROJECTIVE_TET_MESH: public TET_MESH<TYPE> 
{
public:
	TYPE*	old_X;
	TYPE*	Error;
	TYPE*	V;
	int*	fixed;

	TYPE	rho;
	TYPE	gravity;
	TYPE	elasticity;
	
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<TYPE> >	solver;
	Eigen::SparseMatrix<TYPE>								matrix;

	TYPE*	MD;			//matrix diagonal
	TYPE*	Tet_Temp;
	int*	VTT;		//The index list mapping to Tet_Temp
	int*	vtt_num;


	PROJECTIVE_TET_MESH()
	{
		old_X	= new TYPE	[max_number*3];
		Error	= new TYPE	[max_number*3];
		V		= new TYPE	[max_number*3];
		fixed	= new int	[max_number  ];

		MD		= new TYPE	[max_number  ];
		Tet_Temp= new TYPE	[max_number*24];

		VTT		= new int	[max_number*4];
		vtt_num	= new int	[max_number  ];

		rho			= 0.9992;
		gravity		= -9.8;
		elasticity	= 5000000;

		memset(		V, 0, sizeof(TYPE)*max_number*3);
		memset(	fixed, 0, sizeof(int )*max_number  );
	}
	
	~PROJECTIVE_TET_MESH()
	{
		if(old_X)		delete[] old_X;
		if(Error)		delete[] Error;
		if(V)			delete[] V;
		if(fixed)		delete[] fixed;
		if(MD)			delete[] MD;
		if(Tet_Temp)	delete[] Tet_Temp;
		if(VTT)			delete[] VTT;
		if(vtt_num)		delete[] vtt_num;
	}


///////////////////////////////////////////////////////////////////////////////////////////
//  Initialize functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(TYPE t)
	{
		TET_MESH<TYPE>::Initialize();
		Initialize_MD();
		Initialize_Eigen_Solve(t);
		Build_VTT();
	}

	void Initialize_Eigen_Solve(TYPE t)
	{
		std::vector<Eigen::Triplet<TYPE>> coefficients; 

		for(int i=0; i<number; i++)
		{
			TYPE c=(M[i]+fixed[i])/(t*t);
			//printf("input c %d: %f\n", i, c);
			coefficients.push_back(Eigen::Triplet<TYPE>(i, i, c));
		}

		for(int t=0; t<tet_number; t++)
		{
			int*	v=&Tet[t*4];
			TYPE*	idm=&inv_Dm[t*9];
			TYPE	half_matrix[3][4];
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
			
			TYPE	full_matrix[4][4];
			Matrix_Self_Product(&half_matrix[0][0], &full_matrix[0][0], 3, 4);

			for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)			
				coefficients.push_back(Eigen::Triplet<TYPE>(v[i], v[j], full_matrix[i][j]*Vol[t]*elasticity));			
		}
		
		matrix.resize(number, number);		
		matrix.setFromTriplets(coefficients.begin(), coefficients.end());
		solver.compute(matrix);		
	}
	
	void Initialize_MD()	//matrix diagonal
	{
		memset(MD, 0, sizeof(TYPE)*number);
		for(int t=0; t<tet_number; t++)
		{
			int*	v=&Tet[t*4];
			TYPE*	idm=&inv_Dm[t*9];

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

///////////////////////////////////////////////////////////////////////////////////////////
//  Basic update functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Update(TYPE t, int select_v, TYPE target[])
	{
		for(int i=0; i<number; i++) if(fixed[i]==0)
		{			
			//Apply damping
			V[i*3+0]*=0.999;
			V[i*3+1]*=0.999;
			V[i*3+2]*=0.999;
			//Apply Gravity
			V[i*3+1]+=gravity*t;
			//Position update
			X[i*3+0]+=V[i*3+0]*t;
			X[i*3+1]+=V[i*3+1]*t;
			X[i*3+2]+=V[i*3+2]*t;			
		}
	}	

///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint functions
///////////////////////////////////////////////////////////////////////////////////////////
    void Begin_Constraints()
	{
		memcpy(old_X, X, sizeof(TYPE)*number*3);
	}
	
	void End_Constraints(TYPE t)
	{
		TYPE inv_t=1/t;
		for(int v=0; v<number; v++)
		{
			V[v*3+0]+=(X[v*3+0]-old_X[v*3+0])*inv_t;
			V[v*3+1]+=(X[v*3+1]-old_X[v*3+1])*inv_t;
			V[v*3+2]+=(X[v*3+2]-old_X[v*3+2])*inv_t;
		}
	}

	void Direct_Constraints(TYPE* next_X, TYPE t)
	{
		TIMER timer;
		//Step 1: Set up Tet_Temp
		for(int t=0; t<tet_number; t++)
		{
			int v0=Tet[t*4+0];
			int v1=Tet[t*4+1];
			int v2=Tet[t*4+2];
			int v3=Tet[t*4+3];
			int p0=Tet[t*4+0]*3;
			int p1=Tet[t*4+1]*3;
			int p2=Tet[t*4+2]*3;
			int p3=Tet[t*4+3]*3;

			TYPE* idm=&inv_Dm[t*9];

			TYPE Ds[9];
			Ds[0]=X[p1+0]-X[p0+0];
			Ds[3]=X[p1+1]-X[p0+1];
			Ds[6]=X[p1+2]-X[p0+2];
			Ds[1]=X[p2+0]-X[p0+0];
			Ds[4]=X[p2+1]-X[p0+1];
			Ds[7]=X[p2+2]-X[p0+2];
			Ds[2]=X[p3+0]-X[p0+0];
			Ds[5]=X[p3+1]-X[p0+1];
			Ds[8]=X[p3+2]-X[p0+2];

			TYPE F[9], R[9], B[3], C[9];;
			Matrix_Product_3(Ds, idm, F);
			memcpy(R, F, sizeof(TYPE)*9);
			SVD3((TYPE (*)[3])R, B, (TYPE (*)[3])C);

			int small_id;
			if(fabsf(B[0])<fabsf(B[1]) && fabsf(B[0])<fabsf(B[2]))	small_id=0;
			else if(fabsf(B[1])<fabsf(B[2]))						small_id=1;
			else													small_id=2;			
			if(R[0]*(R[4]*R[8]-R[7]*R[5])+R[3]*(R[7]*R[2]-R[1]*R[8])+R[6]*(R[1]*R[5]-R[4]*R[2])<0)
			{				
				R[0+small_id]=-R[0+small_id];
				R[3+small_id]=-R[3+small_id];
				R[6+small_id]=-R[6+small_id];
			}
			if(C[0]*(C[4]*C[8]-C[7]*C[5])+C[3]*(C[7]*C[2]-C[1]*C[8])+C[6]*(C[1]*C[5]-C[4]*C[2])<0)
			{
				C[0+small_id]=-C[0+small_id];
				C[3+small_id]=-C[3+small_id];
				C[6+small_id]=-C[6+small_id];
			}
			
			TYPE C_transpose[9], new_R[9];
			Matrix_Transpose_3(C, C_transpose);
			Matrix_Product_3(R, C_transpose, new_R);

			TYPE half_matrix[3][4], result_matrix[3][4];

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

			//Matrix_Substract_3(new_R, F, new_R);
			Matrix_Product(new_R, &half_matrix[0][0], &result_matrix[0][0], 3, 3, 4);
			
			TYPE rate=Vol[t]*elasticity;
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

		//Step 2: update bx, by, bz
		Eigen::VectorXd bx(number);
		Eigen::VectorXd by(number);
		Eigen::VectorXd bz(number);
		for(int i=0; i<number; i++)		
		{
			TYPE c=(M[i]+fixed[i])/(t*t);
			bx(i)=c*old_X[i*3+0];
			by(i)=c*old_X[i*3+1];
			bz(i)=c*old_X[i*3+2];

			for(int index=vtt_num[i]; index<vtt_num[i+1]; index++)
			{
				bx(i)+=Tet_Temp[VTT[index]*3+0];
				by(i)+=Tet_Temp[VTT[index]*3+1];
				bz(i)+=Tet_Temp[VTT[index]*3+2];
			}
		}

		Eigen::VectorXd x, y, z;		
		x = solver.solve(bx);
		y = solver.solve(by);
		z = solver.solve(bz);

		for(int i=0; i<number; i++)
		{
			next_X[i*3+0]=x(i);
			next_X[i*3+1]=y(i);
			next_X[i*3+2]=z(i);
		}		
	}

	void Jacobi_Constraints(TYPE* next_X, TYPE t)
	{
		Get_Error(t);
		TYPE error_sum=0;
		for(int i=0; i<number; i++)		
		{
			TYPE c=(M[i]+fixed[i])/(t*t);
			next_X[i*3+0]=Error[i*3+0]/(c+MD[i])+X[i*3+0];
			next_X[i*3+1]=Error[i*3+1]/(c+MD[i])+X[i*3+1];
			next_X[i*3+2]=Error[i*3+2]/(c+MD[i])+X[i*3+2];
		}		
	}

	void Get_Error(TYPE t)
	{
		//Step 1: Set up Tet_Temp
		for(int t=0; t<tet_number; t++)
		{
			int v0=Tet[t*4+0];
			int v1=Tet[t*4+1];
			int v2=Tet[t*4+2];
			int v3=Tet[t*4+3];
			int p0=Tet[t*4+0]*3;
			int p1=Tet[t*4+1]*3;
			int p2=Tet[t*4+2]*3;
			int p3=Tet[t*4+3]*3;

			TYPE* idm=&inv_Dm[t*9];
			TYPE Ds[9];
			Ds[0]=X[p1+0]-X[p0+0];
			Ds[3]=X[p1+1]-X[p0+1];
			Ds[6]=X[p1+2]-X[p0+2];
			Ds[1]=X[p2+0]-X[p0+0];
			Ds[4]=X[p2+1]-X[p0+1];
			Ds[7]=X[p2+2]-X[p0+2];
			Ds[2]=X[p3+0]-X[p0+0];
			Ds[5]=X[p3+1]-X[p0+1];
			Ds[8]=X[p3+2]-X[p0+2];

			TYPE F[9], R[9], B[3], C[9], new_R[9];
			Matrix_Product_3(Ds, idm, F);
			Get_Rotation((TYPE (*)[3])F, (TYPE (*)[3])new_R);
			
			TYPE half_matrix[3][4], result_matrix[3][4];
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

			Matrix_Substract_3(new_R, F, new_R);
			Matrix_Product(new_R, &half_matrix[0][0], &result_matrix[0][0], 3, 3, 4);
			
			TYPE rate=Vol[t]*elasticity;
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

		//Step 2: Build the equations
		for(int i=0; i<number; i++)		
		{
			TYPE c=(M[i]+fixed[i])/(t*t);
			TYPE b[3];
			b[0]=c*old_X[i*3+0];
			b[1]=c*old_X[i*3+1];
			b[2]=c*old_X[i*3+2];			
			for(int index=vtt_num[i]; index<vtt_num[i+1]; index++)
			{
				b[0]+=Tet_Temp[VTT[index]*3+0];
				b[1]+=Tet_Temp[VTT[index]*3+1];
				b[2]+=Tet_Temp[VTT[index]*3+2];
			}

			Error[i*3+0]=b[0]-c*X[i*3+0];
			Error[i*3+1]=b[1]-c*X[i*3+1];
			Error[i*3+2]=b[2]-c*X[i*3+2];
		}
	}

	void Update(TYPE t, int iterations, int select_v, TYPE target[])
	{
		Begin_Constraints();	
		Update(t, select_v, target);
		Begin_Constraints();

		TYPE omega;
		TYPE* prev_X=new TYPE[number*3];
		TYPE* next_X=new TYPE[number*3];
		memcpy(prev_X, X, sizeof(TYPE)*number*3);
		
		TIMER timer;

		TYPE error=1, last_error;
		for(int l=0; l<iterations; l++)
		{
			//Direct_Constraints(next_X, t);
			Jacobi_Constraints(next_X, t);		

			if(l<=10)		omega=1;
			else if(l==11)	omega=2/(2-rho*rho);
			else			omega=4/(4-rho*rho*omega);
			//omega=1;

			for(int i=0; i<number*3; i++)
			{
				next_X[i]=(next_X[i]-X[i])*0.9+X[i];

				next_X[i]=omega*(next_X[i]-prev_X[i])+prev_X[i];
				prev_X[i]=X[i];
				X[i]=next_X[i];
			}
		}
		printf("time: %f\n", timer.Get_Time());
		
		delete[] prev_X;
		delete[] next_X;
		End_Constraints(t);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  IO functions for storing simulation states
///////////////////////////////////////////////////////////////////////////////////////////
	void Write(std::fstream &output)
	{
		Write_Binaries(output, X, number*3);
		Write_Binaries(output, old_X, number*3);
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
		Read_Binaries(input, old_X, number*3);
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