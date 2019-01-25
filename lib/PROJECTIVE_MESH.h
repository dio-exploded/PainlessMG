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
//  Class PROJECTIVE_MESH
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_PROJECTIVE_MESH_H__
#define __WHMIN_PROJECTIVE_MESH_H__
#include	"../lib/DYNAMIC_MESH.h"
#include	<Eigen/Sparse>


template <class TYPE>
class PROJECTIVE_MESH: public DYNAMIC_MESH<TYPE>
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
    using   DYNAMIC_MESH<TYPE>::V;
    
	// Position data
	TYPE*	prev_X;
	TYPE*	next_X;
	TYPE*	backup_X;
	TYPE*	Error;

	// Length data
	TYPE*	L;
	
	TYPE	under_relax;
	TYPE	rho;
	TYPE	air_damping;
	TYPE	lap_damping;
	int		lap_damping_loop;
	TYPE	gravity;

	// The all neighborhood data structure needed to form the matrix
	int*	all_VV;
	TYPE*	all_VL;
	TYPE*	all_VW;
	TYPE*	all_VC;
	int*	all_vv_num;

	TYPE*	fixed;

	TYPE	spring_k;
	TYPE	bending_k;

	// Eigen matrix
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<TYPE> >	solver;
	Eigen::SparseMatrix<TYPE>								sparse_matrix;
	

///////////////////////////////////////////////////////////////////////////////////////////
//  Constructor and Deconstructor
///////////////////////////////////////////////////////////////////////////////////////////
	PROJECTIVE_MESH()
	{	
		prev_X		= new TYPE	[max_number*3];
		next_X		= new TYPE	[max_number*3];
		backup_X	= new TYPE	[max_number*3];
		Error		= new TYPE	[max_number*3];

		L			= new TYPE	[max_number*3];

		all_VV		= new int	[max_number*12];
		all_VL		= new TYPE	[max_number*12];
		all_VW		= new TYPE	[max_number*12];
		all_VC		= new TYPE	[max_number];	
		all_vv_num	= new int	[max_number];
	
		fixed		= new TYPE	[max_number];
		
		under_relax			= 1;
		rho					= 0.9992;
		air_damping			= 0.9999;
		lap_damping			= 0.1;
		lap_damping_loop	= 4;
		gravity				= -9.8;

		spring_k			= 5000000;	//1000000
		bending_k			= 10000;	//10000

		memset(fixed, 0, sizeof(TYPE)*max_number);
	}
	
	~PROJECTIVE_MESH()
	{
		if(prev_X)		delete[] prev_X;
		if(next_X)		delete[] next_X;
		if(backup_X)	delete[] backup_X;
		if(Error)		delete[] Error;

		if(L)			delete[] L;

		if(fixed)		delete[] fixed;

		if(all_VV)		delete[] all_VV;
		if(all_VL)		delete[] all_VL;
		if(all_VW)		delete[] all_VW;
		if(all_VC)		delete[] all_VC;
		if(all_vv_num)	delete[] all_vv_num;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Initialization function
///////////////////////////////////////////////////////////////////////////////////////////
	void Initialize(TYPE t)
	{
		//Initialize connectivity
		Build_Connectivity();		
		//Initialize edge length
		for(int e=0; e<e_number; e++)
		{
			int i=E[e*2+0];
			int j=E[e*2+1];
			L[e]=Distance(&X[i*3], &X[j*3]);
		}
		//Initialize (full) neighborhoods and Eigen matrix
		Build_TN();		
		Build_All_Neighborhood(t);
		Initialize_Eigen_Solve(t);
	}

	void Build_All_Neighborhood(const float t)
	{
		//Step 1: create all edges, including original and bending edges
		int* all_E=new int[e_number*8];
		int  all_e_number=0;
		for(int e=0; e<e_number; e++)
		{
			//Add original edges
			all_E[all_e_number*2+0]=E[e*2+0];
			all_E[all_e_number*2+1]=E[e*2+1];
			all_e_number++;
			all_E[all_e_number*2+0]=E[e*2+1];
			all_E[all_e_number*2+1]=E[e*2+0];
			all_e_number++;
			//Add bending edges
			int t0=ET[e*2+0];
			int t1=ET[e*2+1];
			if(t0==-1 || t1==-1)	continue;
			int v2=T[t0*3+0]+T[t0*3+1]+T[t0*3+2]-E[e*2+0]-E[e*2+1];
			int v3=T[t1*3+0]+T[t1*3+1]+T[t1*3+2]-E[e*2+0]-E[e*2+1];			
			all_E[all_e_number*2+0]=v2;
			all_E[all_e_number*2+1]=v3;
			all_e_number++;
			all_E[all_e_number*2+0]=v3;
			all_E[all_e_number*2+1]=v2;
			all_e_number++;
		}
		Quick_Sort_BE(all_E, 0, all_e_number-1);

		//Step 2: Set all_vv_num and all_VV
		int e=0;
		int all_vv_ptr=0;
		for(int i=0; i<number; i++)
		{
			all_vv_num[i]=all_vv_ptr;
			for(; e<all_e_number; e++)
			{			
				if(all_E[e*2]!=i)						break;		// not in the right vertex
				if(e!=0 && all_E[e*2+1]==all_E[e*2-1])	continue;	// duplicate
				all_VV[all_vv_ptr++]=all_E[e*2+1];
			}
		}
		all_vv_num[number]=all_vv_ptr;
		delete[] all_E;
		
		//Step 3: Set all_VL, all_VC, and all_VW
		for(int i=0; i<all_vv_num[number]; i++)	
			all_VL[i]=-1;
		for(int i=0; i<all_vv_num[number]; i++)	
			all_VW[i]= 0;
		for(int i=0; i<number; i++)		
			all_VC[i]=(M[i]+fixed[i])/(t*t);
		
		for(int e=0; e<e_number; e++)
		{
			int v[4];
			v[0]=E[e*2+0];
			v[1]=E[e*2+1];
			
			//First, handle spring length
			TYPE l=Distance(&X[E[e*2+0]*3], &X[E[e*2+1]*3]);
			all_VL[Find_Neighbor(v[0], v[1])]=l;
			all_VL[Find_Neighbor(v[1], v[0])]=l;
			all_VC[v[0]]+=spring_k;
			all_VC[v[1]]+=spring_k;
			all_VW[Find_Neighbor(v[0], v[1])]-=spring_k;
			all_VW[Find_Neighbor(v[1], v[0])]-=spring_k;

			//Next, handle bending weights
			int t0=ET[e*2+0];
			int t1=ET[e*2+1];
			if(t0==-1 || t1==-1)	continue;
			v[2]=T[t0*3+0]+T[t0*3+1]+T[t0*3+2]-v[0]-v[1];
			v[3]=T[t1*3+0]+T[t1*3+1]+T[t1*3+2]-v[0]-v[1];
			TYPE c01=Cotangent(&X[v[0]*3], &X[v[1]*3], &X[v[2]*3]);
			TYPE c02=Cotangent(&X[v[0]*3], &X[v[1]*3], &X[v[3]*3]);
			TYPE c03=Cotangent(&X[v[1]*3], &X[v[0]*3], &X[v[2]*3]);
			TYPE c04=Cotangent(&X[v[1]*3], &X[v[0]*3], &X[v[3]*3]);			
			TYPE area0=sqrt(Area_Squared(&X[v[0]*3], &X[v[1]*3], &X[v[2]*3]));
			TYPE area1=sqrt(Area_Squared(&X[v[0]*3], &X[v[1]*3], &X[v[3]*3]));
			TYPE weight=1/(area0+area1);
			TYPE k[4];
			k[0]= c03+c04;
			k[1]= c01+c02;
			k[2]=-c01-c03;
			k[3]=-c02-c04;

			for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
			{
				if(i==j)	all_VC[v[i]]+=k[i]*k[j]*bending_k;
				else		all_VW[Find_Neighbor(v[i], v[j])]+=k[i]*k[j]*bending_k;
			}
		}
	}

	int Find_Neighbor(int i, int j)
	{
		for(int index=all_vv_num[i]; index<all_vv_num[i+1]; index++)
			if(all_VV[index]==j)	return index;
		printf("ERROR: failed to find the neighbor in all_VV.\n"); getchar();
		return -1;
	}

	void Quick_Sort_BE(int a[], int l, int r)
	{				
		if(l>=r)	return;
		int j=Quick_Sort_Partition_BE(a, l, r);
		Quick_Sort_BE(a, l, j-1);
		Quick_Sort_BE(a, j+1, r);		
	}
	
	int Quick_Sort_Partition_BE(int a[], int l, int r) 
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

	void Initialize_Eigen_Solve(TYPE t)
	{
		std::vector<Eigen::Triplet<TYPE>> coefficients; // list of non-zeros coefficients

		for(int i=0; i<number; i++)		
		{
			coefficients.push_back(Eigen::Triplet<TYPE>(i, i, all_VC[i]));			
			for(int index=all_vv_num[i]; index<all_vv_num[i+1]; index++)			
				coefficients.push_back(Eigen::Triplet<TYPE>(i, all_VV[index], all_VW[index]));
		}		

		sparse_matrix.resize(number, number);		
		sparse_matrix.setFromTriplets(coefficients.begin(), coefficients.end());
		solver.compute(sparse_matrix);		
	}
	
///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Get_Error(TYPE t)
	{
		for(int i=0; i<number; i++)		
		{
			TYPE c=(M[i]+fixed[i])/(t*t);
			Error[i*3+0]=c*old_X[i*3+0];
			Error[i*3+1]=c*old_X[i*3+1];
			Error[i*3+2]=c*old_X[i*3+2];
		}		
		for(int i=0; i<number; i++)
		{
			for(int index=all_vv_num[i]; index<all_vv_num[i+1]; index++)
			{
				int j=all_VV[index];

				// Remove the off-diagonal (Jacobi method)
				Error[i*3+0]-=all_VW[index]*X[j*3+0];
				Error[i*3+1]-=all_VW[index]*X[j*3+1];
				Error[i*3+2]-=all_VW[index]*X[j*3+2];
				
				// Add the other part of b
				if(all_VL[index]==-1)	continue;
				TYPE d[3];
				d[0]=X[i*3+0]-X[j*3+0];
				d[1]=X[i*3+1]-X[j*3+1];
				d[2]=X[i*3+2]-X[j*3+2];
				TYPE new_L=spring_k*all_VL[index]/sqrt(DOT(d, d));
				Error[i*3+0]+=d[0]*new_L;
				Error[i*3+1]+=d[1]*new_L;
				Error[i*3+2]+=d[2]*new_L;
			}
		}		
		 
		for(int i=0; i<number; i++)
		{
			Error[i*3+0]=Error[i*3+0]-all_VC[i]*X[i*3+0];
			Error[i*3+1]=Error[i*3+1]-all_VC[i]*X[i*3+1];
			Error[i*3+2]=Error[i*3+2]-all_VC[i]*X[i*3+2];
		}
	}

	void Jacobi_Constraints(TYPE* next_X, TYPE t)
	{
		Get_Error(t);
		for(int i=0; i<number; i++)
		{
			TYPE inv_c=1/all_VC[i];			
			next_X[i*3+0]=Error[i*3+0]*inv_c+X[i*3+0];
			next_X[i*3+1]=Error[i*3+1]*inv_c+X[i*3+1];
			next_X[i*3+2]=Error[i*3+2]*inv_c+X[i*3+2];
		}
	}
	
	void Apply_Constraints(TYPE t)
	{
		// Legacy function here.
		// Was used for PBD.
	}

	void Direct_Constraints(TYPE* next_X, TYPE t)
	{
		Eigen::VectorXd bx(number);
		Eigen::VectorXd by(number);
		Eigen::VectorXd bz(number);

		for(int i=0; i<number; i++)		
		{
			TYPE c=(M[i]+fixed[i])/(t*t);
			bx(i)=c*old_X[i*3+0];
			by(i)=c*old_X[i*3+1];
			bz(i)=c*old_X[i*3+2];
		}

		for(int i=0; i<number; i++)		
		{
			for(int index=all_vv_num[i]; index<all_vv_num[i+1]; index++)
			{
				if(all_VL[index]==-1)	continue;
				int j=all_VV[index];

				TYPE d[3];
				d[0]=X[i*3+0]-X[j*3+0];
				d[1]=X[i*3+1]-X[j*3+1];
				d[2]=X[i*3+2]-X[j*3+2];
				TYPE new_L=spring_k*all_VL[index]/sqrt(DOT(d, d));				
				bx(i)+=d[0]*new_L;
				by(i)+=d[1]*new_L;
				bz(i)+=d[2]*new_L;
			}
		}		
		
		Eigen::VectorXd x = solver.solve(bx);
		Eigen::VectorXd y = solver.solve(by);
		Eigen::VectorXd z = solver.solve(bz);
		for(int i=0; i<number; i++)
		{
			next_X[i*3+0]=x(i);
			next_X[i*3+1]=y(i);
			next_X[i*3+2]=z(i);
		}		
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Dynamic simulation functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Damping(TYPE* V, TYPE* next_V, const TYPE* fixed,  const int* all_VV, const int* all_vv_num, const TYPE air_damping, const TYPE lap_damping, const int lap_damping_loop)
	{
		//First: Laplacian damping 
		for(int l=0; l<lap_damping_loop; l++)
		{
			for(int i=0; i<number; i++)
			{
				next_V[i*3+0]=0;
				next_V[i*3+1]=0;
				next_V[i*3+2]=0;		
				if(fixed[i]!=0)	continue;

				for(int index=all_vv_num[i]; index<all_vv_num[i+1]; index++)
				{				
					int j=all_VV[index];
					next_V[i*3+0]+=V[j*3+0]-V[i*3+0];
					next_V[i*3+1]+=V[j*3+1]-V[i*3+1];
					next_V[i*3+2]+=V[j*3+2]-V[i*3+2];
				}
				next_V[i*3+0]=V[i*3+0]+next_V[i*3+0]*lap_damping;
				next_V[i*3+1]=V[i*3+1]+next_V[i*3+1]*lap_damping;
				next_V[i*3+2]=V[i*3+2]+next_V[i*3+2]*lap_damping;
			}
			Swap(V, next_V);
		}

		//Second: Air damping 
		for(int i=0; i<number; i++)
		{
			V[i*3+0]*=air_damping;
			V[i*3+1]*=air_damping;
			V[i*3+2]*=air_damping;
		}
	}

	void Update(TYPE t)
	{			
		for(int i=0; i<number; i++)
		{	
			if(fixed[i])	continue;			
			//Apply Gravity
			V[i*3+1]+=gravity*t;

			//Position update
			X[i*3+0]=X[i*3+0]+V[i*3+0]*t;
			X[i*3+1]=X[i*3+1]+V[i*3+1]*t;
			X[i*3+2]=X[i*3+2]+V[i*3+2]*t;
		}
	}

	void Update(TYPE t, int iterations)
	{
		//Damp the velocity
		Damping(V, next_X, fixed, all_VV, all_vv_num, air_damping, lap_damping, lap_damping_loop);		
		
		memcpy(backup_X, X, sizeof(TYPE)*number*3);
		Update(t);		
		Begin_Constraints();

		TYPE omega;
		for(int l=0; l<iterations; l++)
		{
			// Apply constraints
			// Direct_Constraints(next_X, t);
			Jacobi_Constraints(next_X, t);
		
			// Apply Chebyshev acceleration
			if(l<=10)		omega=1;
			else if(l==11)	omega=2/(2-rho*rho);
			else			omega=4/(4-rho*rho*omega);
			//	omega=1;			
			for(int i=0; i<number*3; i++)
			{
				next_X[i]=(next_X[i]-X[i])*under_relax+X[i];
				next_X[i]=omega*(next_X[i]-prev_X[i])+prev_X[i];
			}
			Swap(X, prev_X);
			Swap(X, next_X);
		}

		End_Constraints(1/t);
	}
};


#endif
