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
//  (imported from Huamin's old mesh library since 2003. Updated in 2014. Simplified in 2015)
//  Class BASE_MESH  (without connectivity information)
//  Class MESH		(use this for more regular purposes)
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_MESH_H__
#define __WHMIN_MESH_H__
#include <gl/glut.h>

#include <stdio.h>
#include <fstream>
#include <vector>
#include "IO_FUNC.h"
#include "MY_MATH.h"
#include "INTERSECTION.h"
#include "DISTANCE.h"


template <class TYPE>
class BASE_MESH
{
public:
	int		max_number;
	int		number;
	int		t_number;

	TYPE*	X;
	TYPE*	M;
	int*	T;
	TYPE*	VN;
	TYPE*	TN;

	int* e_v1;
	int* e_v2;
	TYPE* e_k;
	
	BASE_MESH(int _max_number=320000)
	{
		max_number	= _max_number;
		number		= 0;
		t_number	= 0;

		X	= new TYPE	[max_number*3  ];
		M	= new TYPE	[max_number	   ];
		T	= new int	[max_number*2*3];
		VN	= new TYPE	[max_number*3  ];
		TN	= new TYPE	[max_number*2*3];


		// Default mass
		for(int i=0; i<max_number; i++)	M[i]=1;
	}
	
	~BASE_MESH()
	{
		if(X)		delete[] X;
		if(M)		delete[] M;
		if(T)		delete[] T;
		if(TN)		delete[] TN;
		if(VN)		delete[] VN;
	}	

///////////////////////////////////////////////////////////////////////////////////////////
//  Mass functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Compute_Mass(TYPE density=1)
	{
		memset(M, 0, sizeof(TYPE)*max_number);
		for(int t=0; t<t_number; t++)
		{
			int v0=T[t*3+0];
			int v1=T[t*3+1];
			int v2=T[t*3+2];
			TYPE area=sqrt(Area_Squared(&X[v0*3], &X[v1*3], &X[v2*3]))*0.5/3;
			M[v0]+=area;
			M[v1]+=area;
			M[v2]+=area;
		}
		for(int i=0; i<number; i++)	M[i]*=density;
	}

	void Set_Mass(TYPE mass)
	{
		for(int i=0; i<number; i++)	M[i]=mass;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Add mesh (Can be used to copy a mesh as well)
///////////////////////////////////////////////////////////////////////////////////////////
	void Clear_Mesh()
	{
		number=0;
		t_number=0;
	}
	
	void Add_Mesh(BASE_MESH &mesh)
	{
		//Add the mesh 
		for(int t=0; t<mesh.t_number; t++)
		{
			T[t_number*3+0]=mesh.T[t*3+0]+number;
			T[t_number*3+1]=mesh.T[t*3+1]+number;
			T[t_number*3+2]=mesh.T[t*3+2]+number;
			t_number++;
		}
		for(int v=0; v<mesh.number; v++)
		{
			X[number*3+0]=mesh.X[v*3+0];
			X[number*3+1]=mesh.X[v*3+1];
			X[number*3+2]=mesh.X[v*3+2];
			M[number	]=mesh.M[v];
			number++;
		}
	}

	void Merge(int *m_list, int m_number, int *v_color=0, bool color_strategy=true)
	{
		if(v_color && color_strategy)
		{
			//Determine the color of the merged vertices
			for(int m=0; m<m_number; m++) v_color[m_list[m*2+0]]=v_color[m_list[m*2+1]];
		}
		int *map=new int[number];
		for(int i=0; i<number; i++)		map[i]=i;
		for(int m=0; m<m_number; m++)	map[m_list[m*2+1]]=m_list[m*2+0];
		int new_number=0;
		for(int i=0; i<number; i++)	if(map[i]==i)
		{
			X[new_number*3+0]=X[i*3+0];
			X[new_number*3+1]=X[i*3+1];
			X[new_number*3+2]=X[i*3+2];
			if(v_color)	v_color[new_number]=v_color[i];
			map[i]=new_number++;
		}
		number=new_number;		
		for(int m=0; m<m_number; m++)	map[m_list[m*2+1]]=map[m_list[m*2+0]];			

		int new_t_number=0;
		for(int t=0; t<t_number; t++)
		{
			T[new_t_number*3+0]=map[T[t*3+0]];
			T[new_t_number*3+1]=map[T[t*3+1]];
			T[new_t_number*3+2]=map[T[t*3+2]];			

			if(T[new_t_number*3+0]==T[new_t_number*3+1])	continue;
			if(T[new_t_number*3+0]==T[new_t_number*3+2])	continue;
			if(T[new_t_number*3+1]==T[new_t_number*3+2])	continue;
			new_t_number++;
		}
		t_number=new_t_number;

		delete[] map;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Builder functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Make_A_Plane(int ni, int nj, TYPE x, TYPE y, TYPE z)
	{
		TYPE s=1.0/(ni-1);
		for(int i=0; i<ni; i++)	for(int j=0; j<nj; j++)
		{
			X[number*3+(i*nj+j)*3+0]=j*s+x;//x+i* 0.01;
			X[number*3+(i*nj+j)*3+1]=y-i*s;//y+0.1-j*0.01;
			X[number*3+(i*nj+j)*3+2]=z;//0.01;
		}

		for(int i=0; i<ni-1; i+=2)	for(int j=0; j<nj-1; j+=2)
		{
			T[t_number*3+0]=number+i*nj+j;
			T[t_number*3+2]=number+i*nj+(j+1);
			T[t_number*3+1]=number+(i+1)*nj+j;
			t_number++;
			T[t_number*3+0]=number+i*nj+(j+1);
			T[t_number*3+2]=number+(i+1)*nj+(j+1);
			T[t_number*3+1]=number+(i+1)*nj+j;
			t_number++;
		}

		for(int i=1; i<ni-1; i+=2)	for(int j=1; j<nj-1; j+=2)
		{
			T[t_number*3+0]=number+i*nj+j;
			T[t_number*3+2]=number+i*nj+(j+1);
			T[t_number*3+1]=number+(i+1)*nj+j;
			t_number++;
			T[t_number*3+0]=number+i*nj+(j+1);
			T[t_number*3+2]=number+(i+1)*nj+(j+1);
			T[t_number*3+1]=number+(i+1)*nj+j;
			t_number++;
		}

		for (int i = 1; i < ni - 1; i += 2)	for (int j = 0; j < nj - 1; j += 2)
		{
			T[t_number * 3 + 0] = number + i * nj + j;
			T[t_number * 3 + 2] = number + i * nj + (j + 1);
			T[t_number * 3 + 1] = number + (i + 1)*nj + (j + 1) ;
			t_number++;
			T[t_number * 3 + 0] = number + i * nj + j;
			T[t_number * 3 + 2] = number + (i + 1)*nj + (j + 1);
			T[t_number * 3 + 1] = number + (i + 1)*nj + j;
			t_number++;
		}

		for(int i=0; i<ni-1; i+=2)	for(int j=1; j<nj-1; j+=2)
		{
			T[t_number*3+0]=number+i*nj+j;
			T[t_number*3+2]=number+i*nj+(j+1);
			T[t_number*3+1]=number+(i+1)*nj+(j+1);
			t_number++;
			T[t_number*3+0]=number+i*nj+j;
			T[t_number*3+2]=number+(i+1)*nj+(j+1);
			T[t_number*3+1]=number+(i+1)*nj+j;
			t_number++;	
		}
		number+=ni*nj;

		for (int i = 0; i < number; i++)
			M[i] = 1.0 / number;
	}

	void Make_Cylinder(TYPE x, TYPE y, TYPE z, TYPE r, TYPE h, int m, int n)
	{
		int old_number=number;
		for(int i=0; i<m; i++)
		for(int j=0; j<n; j++)
		{
			TYPE angle;
			if(i%2==0)	angle=(2.0*j*MY_PI)/n;
			else		angle=(2.0*j*MY_PI-MY_PI)/n;

			X[number*3+0]=x+cosf(angle)*r;
			X[number*3+1]=y-h*i;
			X[number*3+2]=z+sinf(angle)*r;
			number++;
		}

		for(int i=0; i<m-1; i++)
		for(int j=0; j<n;   j++)
		{
            if(i%2==0)
			{
                T[t_number*3+0]=old_number+(i  )*n+j;
                T[t_number*3+1]=old_number+(i+1)*n+(j+1)%n;
                T[t_number*3+2]=old_number+(i+1)*n+j;
                t_number++;
                T[t_number*3+0]=old_number+(i  )*n+j;
                T[t_number*3+1]=old_number+(i  )*n+(j+1)%n;
                T[t_number*3+2]=old_number+(i+1)*n+(j+1)%n;
                t_number++;
            }
            else
            {
                T[t_number*3+0]=old_number+(i  )*n+(j+1)%n;
                T[t_number*3+1]=old_number+(i+1)*n+j;
                T[t_number*3+2]=old_number+(i  )*n+j;
                t_number++;
                T[t_number*3+0]=old_number+(i  )*n+(j+1)%n;
                T[t_number*3+1]=old_number+(i+1)*n+(j+1)%n;
                T[t_number*3+2]=old_number+(i+1)*n+j;
                t_number++;
            }
		}
	}

	void Make_Sphere(TYPE x, TYPE y, TYPE z, TYPE r, int m, int n)
	{
		int old_number=number;
		for(int i=0; i<=n; i++)
		for(int j=0; j<=m; j++)
		{
			TYPE angle_i=(2.0*i*MY_PI)/n;			
			TYPE angle_j=(j*MY_PI)/m - MY_PI/2;			
			
			X[number*3+0]=x+cosf(angle_j)*cosf(angle_i)*r;
			X[number*3+1]=y+sinf(angle_j)*r;
			X[number*3+2]=z+cosf(angle_j)*sinf(angle_i)*r;
			number++;
		}

		for(int i=0; i<n; i++)
		for(int j=0; j<m; j++)
		{
			{
				T[t_number*3+0]=old_number+(i      )*(m+1)+j;
				T[t_number*3+1]=old_number+((i+1)%n)*(m+1)+j;
				T[t_number*3+2]=old_number+((i+1)%n)*(m+1)+j+1;
				t_number++;
				T[t_number*3+0]=old_number+(i      )*(m+1)+j;
				T[t_number*3+1]=old_number+((i+1)%n)*(m+1)+j+1;
				T[t_number*3+2]=old_number+(i      )*(m+1)+j+1;
				t_number++;
			}
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Normal functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Build_TN()
	{
		memset(TN, 0, sizeof(TYPE)*number*3);
		for(int i=0; i<t_number; i++)
		{
			TYPE *p0=&X[T[i*3+0]*3];
			TYPE *p1=&X[T[i*3+1]*3];
			TYPE *p2=&X[T[i*3+2]*3];
			if(p0==p1 || p0==p2 || p1==p2)	continue;
			Normal(p0, p1, p2, &TN[i*3]);
		}
	}

	void Build_VN()
	{
		memset(TN, 0, sizeof(TYPE)*number*3);
		for(int i=0; i<t_number; i++)
		{
			TYPE *p0=&X[T[i*3+0]*3];
			TYPE *p1=&X[T[i*3+1]*3];
			TYPE *p2=&X[T[i*3+2]*3];
			if(p0==p1 || p0==p2 || p1==p2)	continue;			
			TYPE p10[3], p20[3];
			p10[0]=p1[0]-p0[0];
			p10[1]=p1[1]-p0[1];
			p10[2]=p1[2]-p0[2];
			p20[0]=p2[0]-p0[0];
			p20[1]=p2[1]-p0[1];
			p20[2]=p2[2]-p0[2];
			Cross(p10, p20, &TN[i*3]);
		}

		memset(VN, 0, sizeof(TYPE)*number*3);
		for(int i=0; i<t_number; i++)
		{
			int v0=T[i*3+0];
			int v1=T[i*3+1];
			int v2=T[i*3+2];
			VN[v0*3+0]+=TN[i*3+0];
			VN[v0*3+1]+=TN[i*3+1];
			VN[v0*3+2]+=TN[i*3+2];
			VN[v1*3+0]+=TN[i*3+0];
			VN[v1*3+1]+=TN[i*3+1];
			VN[v1*3+2]+=TN[i*3+2];
			VN[v2*3+0]+=TN[i*3+0];
			VN[v2*3+1]+=TN[i*3+1];
			VN[v2*3+2]+=TN[i*3+2];			
		}
		for(int t=0; t<t_number; t++)
			Normalize(&TN[t*3]);
		for(int i=0; i<number; i++)
			Normalize(&VN[i*3]);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Editing functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Add_T(int t0, int t1, int t2)
	{
		T[t_number*3+0]=t0;
		T[t_number*3+1]=t1;
		T[t_number*3+2]=t2;
		t_number++;
	}

	void Remove_V(int vid)
	{		
		for(int v=vid; v<number; v++)
		{
			X[v*3+0]=X[v*3+3];
			X[v*3+1]=X[v*3+4];
			X[v*3+2]=X[v*3+5];
		}
		number--;

		int write=0;
		for(int t=0; t<t_number; t++)
		{
			if(T[t*3+0]==vid || T[t*3+1]==vid || T[t*3+2]==vid)	continue;
			if(write!=t)	//write
			{
				T[write*3+0]=T[t*3+0];
				T[write*3+1]=T[t*3+1];
				T[write*3+2]=T[t*3+2];				
			}

			if(T[write*3+0]>vid)	T[write*3+0]--;
			if(T[write*3+1]>vid)	T[write*3+1]--;
			if(T[write*3+2]>vid)	T[write*3+2]--;
			write++;
		}
		t_number=write;		
	}

	void Remove_T(int tid)
	{
		for(int t=tid; t<t_number-1; t++)
		{
			T[t*3+0]=T[t*3+3];
			T[t*3+1]=T[t*3+4];
			T[t*3+2]=T[t*3+5];
		}
		t_number--;
	}

	void Invert()
	{
		for(int t=0; t<t_number; t++)
			Swap(T[t*3+0], T[t*3+1]);
	}

	void Select(TYPE p[], TYPE q[], int& select_v)
	{
		TYPE dir[3];
		dir[0]=q[0]-p[0];
		dir[1]=q[1]-p[1];
		dir[2]=q[2]-p[2];
		Normalize(dir);

		TYPE min_t=MY_INFINITE;
		int	 select_t;
		for(int t=0; t<t_number; t++)
		{
			TYPE _min_t=MY_INFINITE;
			if(Ray_Triangle_Intersection(&X[T[t*3+0]*3], &X[T[t*3+1]*3], &X[T[t*3+2]*3], p, dir, _min_t) && _min_t>0 && _min_t<min_t)
			{
				select_t = t;
				min_t = _min_t;
			}
		}

		if(min_t!=MY_INFINITE)	//Selection made
		{
			TYPE r;
			TYPE d0=Squared_VE_Distance(&X[T[select_t*3+0]*3], p, q, r);
			TYPE d1=Squared_VE_Distance(&X[T[select_t*3+1]*3], p, q, r);
			TYPE d2=Squared_VE_Distance(&X[T[select_t*3+2]*3], p, q, r);
			if(d0<d1 && d0<d2)	select_v=T[select_t*3+0];
			else if(d1<d2)		select_v=T[select_t*3+1];
			else				select_v=T[select_t*3+2];
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Geometric functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Scale(TYPE s)
	{
		Scale(s, s, s);
	}

	void Scale(TYPE sx, TYPE sy, TYPE sz, int _number=-1)
	{
		if(_number==-1)	_number=number;		
		for(int i=0; i<_number; i++)
		{
			X[i*3+0]*=sx;
			X[i*3+1]*=sy;
			X[i*3+2]*=sz;
		}
	}

	void New_Scale(TYPE sx, TYPE sy, TYPE sz)
	{
		for(int i=0; i<number; i++)
		if(X[i*3+1]<1.0 && fabs(X[i*3+0])<0.2)
		{
			X[i*3+0]*=sx;
			X[i*3+1]*=sy;
			X[i*3+2]*=sz;
		}
	}

	void Center(TYPE c[])
	{
		c[0]=c[1]=c[2]=0;
		TYPE mass_sum=0;
		for(int i=0; i<number; i++)
		{
			c[0]	 += M[i]*X[i*3+0];
			c[1]	 += M[i]*X[i*3+1];
			c[2]	 += M[i]*X[i*3+2];
			mass_sum += M[i];
		}
		c[0]/=mass_sum;
		c[1]/=mass_sum;
		c[2]/=mass_sum;
	}

	void Centralize()
	{
		TYPE c[3];
		Center(c);
		for(int i=0; i<number; i++)
		{
			X[i*3+0]-=c[0];
			X[i*3+1]-=c[1];
			X[i*3+2]-=c[2];
		}
	}

	void Rotate_Z(TYPE angle, int _number=-1)
	{
		if(_number==-1)	_number=number;
		for(int i=0; i<_number; i++)
		{
			TYPE tx=X[i*3+0];
			TYPE ty=X[i*3+1];
			X[i*3+0]= tx*cosf(angle)-ty*sinf(angle);
			X[i*3+1]= tx*sinf(angle)+ty*cosf(angle);
		}
	}

	void Rotate_Y(TYPE angle, int _number=-1)
	{
		if(_number==-1)	_number=number;
		for(int i=0; i<_number; i++)
		{
			TYPE tx=X[i*3+0];
			TYPE ty=X[i*3+2];
			X[i*3+0]= tx*cosf(angle)+ty*sinf(angle);
			X[i*3+2]=-tx*sinf(angle)+ty*cosf(angle);
		}
	}

	void Rotate_X(TYPE angle, int _number=-1)
	{
		if(_number==-1)	_number=number;
		for(int i=0; i<_number; i++)
		{
			TYPE tx=X[i*3+1];
			TYPE ty=X[i*3+2];
			X[i*3+1]= tx*cosf(angle)+ty*sinf(angle);
			X[i*3+2]=-tx*sinf(angle)+ty*cosf(angle);
		}
	}

	void Translate(TYPE tx, TYPE ty, TYPE tz, int _number=-1)
	{
		if(_number==-1)	_number=number;
		for(int i=0; i<_number; i++)
		{
			X[i*3+0]+=tx;
			X[i*3+1]+=ty;
			X[i*3+2]+=tz;
		}
	}

	void Range(TYPE &min_x, TYPE &min_y, TYPE &min_z, TYPE &max_x, TYPE &max_y, TYPE &max_z)
	{
		min_x= MY_INFINITE;
		min_y= MY_INFINITE;
		min_z= MY_INFINITE;
		max_x=-MY_INFINITE;
		max_y=-MY_INFINITE;
		max_z=-MY_INFINITE;
		for(int i=0; i<number; i++)
		{
			if(X[i*3+0]<min_x)	min_x=X[i*3+0];
			if(X[i*3+1]<min_y)	min_y=X[i*3+1];
			if(X[i*3+2]<min_z)	min_z=X[i*3+2];
			if(X[i*3+0]>max_x)	max_x=X[i*3+0];
			if(X[i*3+1]>max_y)	max_y=X[i*3+1];
			if(X[i*3+2]>max_z)	max_z=X[i*3+2];
		}
	}

	void Pertubation()
	{
		for(int v=0; v<number; v++)
		{
			X[v*3+1]+=RandomFloat()*0.0024f;
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Curvatures function: obtain mean curvature lists
///////////////////////////////////////////////////////////////////////////////////////////
	void Curvatures(TYPE c[])
	{
		TYPE *area=new TYPE[number];
		TYPE *curv=new TYPE[number*3];
		memset(area, 0, sizeof(TYPE)*number);
		memset(curv, 0, sizeof(TYPE)*number*3);

		for(int t=0; t<t_number; t++)
		{
			int v0=T[t*3+0];
			int v1=T[t*3+1];
			int v2=T[t*3+2];
			TYPE cot0=Cotangent(&X[v0*3], &X[v1*3], &X[v2*3]);
			TYPE cot1=Cotangent(&X[v1*3], &X[v2*3], &X[v0*3]);
			TYPE cot2=Cotangent(&X[v2*3], &X[v0*3], &X[v1*3]);
			TYPE a=sqrt(Area_Squared(&X[v0*3], &X[v1*3], &X[v2*3]))*2/3;

			curv[v0*3+0]-=cot2*(X[v1*3+0]-X[v0*3+0])+cot1*(X[v2*3+0]-X[v0*3+0]);
			curv[v0*3+1]-=cot2*(X[v1*3+1]-X[v0*3+1])+cot1*(X[v2*3+1]-X[v0*3+1]);
			curv[v0*3+2]-=cot2*(X[v1*3+2]-X[v0*3+2])+cot1*(X[v2*3+2]-X[v0*3+2]);

			curv[v1*3+0]-=cot2*(X[v0*3+0]-X[v1*3+0])+cot0*(X[v2*3+0]-X[v1*3+0]);
			curv[v1*3+1]-=cot2*(X[v0*3+1]-X[v1*3+1])+cot0*(X[v2*3+1]-X[v1*3+1]);
			curv[v1*3+2]-=cot2*(X[v0*3+2]-X[v1*3+2])+cot0*(X[v2*3+2]-X[v1*3+2]);

			curv[v2*3+0]-=cot0*(X[v1*3+0]-X[v2*3+0])+cot1*(X[v0*3+0]-X[v2*3+0]);
			curv[v2*3+1]-=cot0*(X[v1*3+1]-X[v2*3+1])+cot1*(X[v0*3+1]-X[v2*3+1]);
			curv[v2*3+2]-=cot0*(X[v1*3+2]-X[v2*3+2])+cot1*(X[v0*3+2]-X[v2*3+2]);

			//Here we use a simple area model
			area[v0]+=a;
			area[v1]+=a;
			area[v2]+=a;
		}

		Build_VN();
		for(int i=0; i<number; i++)
		{
			curv[i*3+0]/=area[i];
			curv[i*3+1]/=area[i];
			curv[i*3+2]/=area[i];

			c[i]=Magnitude(&curv[i*3]);
			if(Dot(&curv[i*3], &VN[i*3])<0)	c[i]=-c[i];
		}		

		delete[] area;
		delete[] curv;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Read and Write functions
///////////////////////////////////////////////////////////////////////////////////////////
	bool Read_Mesh(const char *filename)
	{
		std::fstream input; 
		input.open(filename,std::ios::in|std::ios::binary);
		if(!input.is_open())	return false;		
		number	= 0; 
		t_number= 0;
		Read_Binary(input, number);
		Read_Binaries(input, X, number*3);
		Read_Binary(input, t_number);		
		Read_Binaries(input, T, t_number*3);
		input.close();
		return true;
	}

	bool Read_Float_Mesh(const char *filename)
	{
		std::fstream input; 
		input.open(filename,std::ios::in|std::ios::binary);
		if(!input.is_open())	return false;		
		number	= 0; 
		t_number= 0;
		Read_Binary(input, number);
		float* _X=new float[number*3];
		Read_Binaries(input, _X, number*3);
		for(int i=0; i<number*3; i++) X[i]=_X[i];
		Read_Binary(input, t_number);		
		Read_Binaries(input, T, t_number*3);
		input.close();
		delete[] _X;
		return true;
	}

	bool Read_More_Mesh(const char *filename)
	{
		BASE_MESH mesh;
		if(mesh.Read_Mesh(filename)==false)	return false;
		Add_Mesh(mesh);
		return true;
	}

	bool Write_Mesh(const char *file_name)
	{
		std::fstream output; 
		output.open(file_name,std::ios::out|std::ios::binary);
		if(!output.is_open())	return false;
		Write_Binary(output, number);
		Write_Binaries(output, X, number*3);
		Write_Binary(output, t_number);
		Write_Binaries(output, T, t_number*3);
		output.close();
		return true;
	}

	void Write_OBJ(const char *filename)
	{		
		FILE *fp=fopen(filename, "w+");
		for(int v=0; v<number; v++)
			fprintf(fp, "v %f %f %f\n", X[v*3+0], X[v*3+1], X[v*3+2]);
				
		for(int t=0; t<t_number; t++)
		{
			fprintf(fp, "f %d %d %d\n", 
			T[t*3+0]+1, T[t*3+1]+1, T[t*3+2]+1);
		}
		fclose(fp);
	}

	void Read_OBJ(char *filename)
	{
		number=0;
		t_number=0;
		int vertex_normal_number=0;
		FILE *fp=0;
		fopen_s(&fp, filename, "r+");	
		if(fp==0)	{printf("ERROR: cannot open %s\n", filename); getchar();}

		while(feof(fp)==0)
		{
			char token[1024];
			fscanf_s(fp, "%s", &token, 1024);				
			if(token[0]=='#' && token[1]=='\0')
			{
				int c;
				while(feof(fp)==0)
					if((c=fgetc(fp))=='\r' || c=='\n')	break;
			}
			else if(token[0]=='v' && token[1]=='\0')	//vertex
			{
				if(sizeof(TYPE)==sizeof(float))	fscanf_s(fp, "%f %f %f\n", &X[number*3], &X[number*3+1], &X[number*3+2]); 
				else							fscanf_s(fp, "%lf %lf %lf\n", &X[number*3], &X[number*3+1], &X[number*3+2]); 
				X[number*3+0]=X[number*3+0];
				X[number*3+1]=X[number*3+1];
				X[number*3+2]=X[number*3+2];
				number++;
			}
			else if(token[0]=='v' && token[1]=='t')
			{
				TYPE temp[2];
				fscanf_s(fp, "%f %f\n", &temp[0], &temp[1]);
			}
			else if(token[0]=='v' && token[1]=='n')
			{
				fscanf_s(fp, "%f %f %f\n", 
					   &VN[0*3], 
					   &VN[0*3+1], 
					   &VN[0*3+2]); 
				//printf("vn: %d/%d\n", vertex_normal_number, max_number);
				//vertex_normal_number++;
			}
			else if(token[0]=='f' && token[1]=='\0')
			{
				int data[16];
				int data_number=0;
				int c;

				fscanf_s(fp, "%s", &token, 1024);
				sscanf_s(token, "%d", &data[0], 1024);

				fscanf_s(fp, "%s", &token, 1024);
				sscanf_s(token, "%d", &data[3], 1024);

				fscanf_s(fp, "%s", &token, 1024);
				sscanf_s(token, "%d", &data[6], 1024);
				
				/*while(1)
				{
					fscanf(fp, "%d", &data[data_number++]);

					c=fgetc(fp);
					if(c==' ' || c=='\t')	continue;
					if(c=='\r' || c=='\n' || feof(fp))	break;
					if(c=='/')	fscanf(fp, "%d", &c);
					
					c=fgetc(fp);
					
					if(c==' ' || c=='\t')	continue;
					if(c=='\r' || c=='\n' || feof(fp))	break;
					if(c=='/')	fscanf(fp, "%d", &c);
				}
				printf("over %d\n", t_number);*/
				//data[9]=data[10]=data[11]=data[12]=0;
				//fscanf(fp, "%d/%d/%d %d/%d/%d %d/%d/%d\n",
				//	&data[0], &data[1], &data[2],
				//	&data[3], &data[4], &data[5],
				//	&data[6], &data[7], &data[8],
				//	&data[9], &data[10], &data[11],
				//	&data[12], &data[13], &data[14]);				
				//if(data[0]==490 || data[3]==490 || data[6]==490 || data[9]==490)
				//	printf("data: %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n", 
				//	   data[0], data[1], data[2],
				//	   data[3], data[4], data[5],
				//	   data[6], data[7], data[8],
				//	   data[9], data[10], data[11]);
								
				T[t_number*3+0]=data[0]-1;
				T[t_number*3+1]=data[3]-1;
				T[t_number*3+2]=data[6]-1;
				t_number++;

			/*	if(data[9]!=0)
				{
					T[t_number*3+0]=data[0]-1;
					T[t_number*3+1]=data[6]-1;
					T[t_number*3+2]=data[9]-1;
					t_number++;				
				}
				if(data[9]!=0 && data[12]!=0)
				{
					T[t_number*3+0]=data[0]-1;
					T[t_number*3+1]=data[9]-1;
					T[t_number*3+2]=data[12]-1;
					t_number++;		
				}*/
			}			
		}
		fclose(fp);		
		printf("v: %d %d; t: %d\n", number, vertex_normal_number, t_number);
	}
};


///////////////////////////////////////////////////////////////////////////////////////////
//  class MESH
//  with connectivity information
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
class MESH: public BASE_MESH<TYPE>
{
public:
    using   BASE_MESH<TYPE>::max_number;
    using   BASE_MESH<TYPE>::number;
    using   BASE_MESH<TYPE>::t_number;
    using   BASE_MESH<TYPE>::X;
    using   BASE_MESH<TYPE>::T;
    using   BASE_MESH<TYPE>::M;
    using   BASE_MESH<TYPE>::VN;
        
	// Edge Connectivity
	int		e_number;
	int*	E;
	int*	ET;	//Triangle lists of each edge (-1 means empty)
	int*	TE;	//Edge lists of each triangle

	// VV/VE neighborhood
	int*	VV;
	int*	VE;
	int*	vv_num;

	// Vertex connectivity
	int*	vt_num;

	// Boundary information
	int		*boundary;

	MESH(int _max_number=160000): BASE_MESH(_max_number)
	{
		E			= new int	[max_number*3*2];
		ET			= new int	[max_number*3*2];
		TE			= new int	[max_number*2*3];
		vt_num		= new int	[max_number];

		VV			= new int	[max_number*3*2];
		VE			= new int	[max_number*3*2];
		vv_num		= new int	[max_number];

		boundary	= new int	[max_number];
		e_number	= 0;
	}

	~MESH()
	{
		if(E)		delete[] E;
		if(ET)		delete[] ET;
		if(TE)		delete[] TE;
		if(vt_num)	delete[] vt_num;

		if(VV)		delete[] VV;
		if(VE)		delete[] VE;
		if(vv_num)	delete[] vv_num;
	}
		
///////////////////////////////////////////////////////////////////////////////////////////
//  Add mesh with edges (Can be used to copy a mesh as well)
///////////////////////////////////////////////////////////////////////////////////////////
	void Add_Mesh_with_Edges(MESH &mesh)
	{
		//Add the mesh 
		for(int e=0; e<mesh.e_number; e++)
		{
			E[e_number*2+0]=mesh.E[e*2+0]+number;
			E[e_number*2+1]=mesh.E[e*2+1]+number;
			e_number++;
		}
		for(int t=0; t<mesh.t_number; t++)
		{
			T[t_number*3+0]=mesh.T[t*3+0]+number;
			T[t_number*3+1]=mesh.T[t*3+1]+number;
			T[t_number*3+2]=mesh.T[t*3+2]+number;
			t_number++;
		}
		for(int v=0; v<mesh.number; v++)
		{
			X[number*3+0]=mesh.X[v*3+0];
			X[number*3+1]=mesh.X[v*3+1];
			X[number*3+2]=mesh.X[v*3+2];
			M[number	]=mesh.M[v];
			number++;
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Build topological functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Build_Boundary()
	{
		memset(boundary, 0, sizeof(int)*number);
		for(int e=0; e<e_number; e++)
		{
			boundary[E[e*2+0]]++;
			boundary[E[e*2+1]]++;
		}
		for(int v=0; v<number; v++)
		{			
			if(vt_num[v]==boundary[v])							
				boundary[v]=0;		
			else boundary[v]=1;
		}
	}

	void Build_Neighborhood()
	{
		//First set vv_num
		memset(vv_num, 0, sizeof(int)*max_number);
		for(int i=0; i<e_number; i++)
		{
			vv_num[E[i*2+0]]++;
			vv_num[E[i*2+1]]++;
		}
		for(int i=1; i<number; i++)
			vv_num[i]+=vv_num[i-1];
		for(int i=number; i>0; i--)
			vv_num[i]=vv_num[i-1];
		vv_num[0]=0;

		//Then set VV and VE
		int *_vv_num=new int[max_number];
		memcpy(_vv_num, vv_num, sizeof(int)*max_number);
		for(int i=0; i<e_number; i++)
		{
			VV[_vv_num[E[i*2+0]]]=E[i*2+1];
			VV[_vv_num[E[i*2+1]]]=E[i*2+0];
			VE[_vv_num[E[i*2+0]]++]=i;
			VE[_vv_num[E[i*2+1]]++]=i;
		}
		delete []_vv_num;
	}

    void Build_Connectivity()
    {
        Build_Edges();
		Build_Neighborhood();
    }

    void Build_VT_Num()
    {		
		memset(vt_num, 0, sizeof(int)*number);
		for(int t=0; t<t_number; t++)
		{
			vt_num[T[t*3+0]]++;
			vt_num[T[t*3+1]]++;
			vt_num[T[t*3+2]]++;
		}
    }
   
	void Build_Edges()
	{
		//Build E, ET and TE
		e_number=0;
		int *_RE=new int[t_number*9];	
		for(int t=0; t<t_number; t++)
		{
			int v0=T[t*3+0];
			int v1=T[t*3+1];
			int v2=T[t*3+2];
			
			if(v0<v1)	{ _RE[t*9+0]=v0; _RE[t*9+1]=v1; _RE[t*9+2]=t;}
			else		{ _RE[t*9+0]=v1; _RE[t*9+1]=v0; _RE[t*9+2]=t;}
			if(v1<v2)	{ _RE[t*9+3]=v1; _RE[t*9+4]=v2; _RE[t*9+5]=t;}
			else		{ _RE[t*9+3]=v2; _RE[t*9+4]=v1; _RE[t*9+5]=t;}
			if(v2<v0)	{ _RE[t*9+6]=v2; _RE[t*9+7]=v0; _RE[t*9+8]=t;}
			else		{ _RE[t*9+6]=v0; _RE[t*9+7]=v2; _RE[t*9+8]=t;}	
		}
		//Quicksort
		Quick_Sort_RE(_RE, 0, t_number*3-1);		
				
		for(int i=0; i<t_number*3; i++)
		{
			//printf("edge: %d, %d\n", _RE[i*3+0], _RE[i*3+1]);

			if(i!=0 && _RE[i*3]==_RE[(i-1)*3] && _RE[i*3+1]== _RE[(i-1)*3+1])
			{
				//Add the edge to ET
				ET[e_number*2-2]=_RE[i*3+2];
				ET[e_number*2-1]=_RE[i*3-1];
				
				//Add the edge to TE
				int v0=T[_RE[i*3+2]*3+0];
				int v1=T[_RE[i*3+2]*3+1];
				int v2=T[_RE[i*3+2]*3+2];
				if(v0==_RE[i*3+0] && v1==_RE[i*3+1] || v1==_RE[i*3+0] && v0==_RE[i*3+1])	TE[_RE[i*3+2]*3+0]=e_number-1;
				if(v1==_RE[i*3+0] && v2==_RE[i*3+1] || v2==_RE[i*3+0] && v1==_RE[i*3+1])	TE[_RE[i*3+2]*3+1]=e_number-1;
				if(v2==_RE[i*3+0] && v0==_RE[i*3+1] || v0==_RE[i*3+0] && v2==_RE[i*3+1])	TE[_RE[i*3+2]*3+2]=e_number-1;
			}
			else
			{
				//Add the edge to E
				E[e_number*2+0]=_RE[i*3+0];
				E[e_number*2+1]=_RE[i*3+1];
				//Add the edge to ET
				ET[e_number*2+0]=_RE[i*3+2];
				ET[e_number*2+1]=-1;				
				//Add the edge to  TE
				int v0=T[_RE[i*3+2]*3+0];
				int v1=T[_RE[i*3+2]*3+1];
				int v2=T[_RE[i*3+2]*3+2];
				if(v0==_RE[i*3+0] && v1==_RE[i*3+1] || v1==_RE[i*3+0] && v0==_RE[i*3+1])	TE[_RE[i*3+2]*3+0]=e_number;
				if(v1==_RE[i*3+0] && v2==_RE[i*3+1] || v2==_RE[i*3+0] && v1==_RE[i*3+1])	TE[_RE[i*3+2]*3+1]=e_number;
				if(v2==_RE[i*3+0] && v0==_RE[i*3+1] || v0==_RE[i*3+0] && v2==_RE[i*3+1])	TE[_RE[i*3+2]*3+2]=e_number;	
				e_number++;
			}
		}
		delete []_RE;		
	}
	
	void Quick_Sort_RE( int a[], int l, int r)
	{
		if(l<r)
		{
			int j=Quick_Sort_Partition_RE( a, l, r);			

			Quick_Sort_RE( a, l, j-1);
			Quick_Sort_RE( a, j+1, r);
		}
	}
	
	int Quick_Sort_Partition_RE( int a[], int l, int r) 
	{
		int pivot[3], i, j, t[3];
		pivot[0] = a[l*3+0];
		pivot[1] = a[l*3+1];
		pivot[2] = a[l*3+2];	
		i = l; j = r+1;		
		while( 1)
		{
			do ++i; while( (a[i*3]<pivot[0] || a[i*3]==pivot[0] && a[i*3+1]<=pivot[1]) && i <= r );
			do --j; while(  a[j*3]>pivot[0] || a[j*3]==pivot[0] && a[j*3+1]> pivot[1] );
			if( i >= j ) break;
			//Swap i and j			
			t[0]=a[i*3+0];
			t[1]=a[i*3+1];
			t[2]=a[i*3+2];
			a[i*3+0]=a[j*3+0];
			a[i*3+1]=a[j*3+1];
			a[i*3+2]=a[j*3+2];
			a[j*3+0]=t[0];
			a[j*3+1]=t[1];
			a[j*3+2]=t[2];
		}
		//Swap l and j
		t[0]=a[l*3+0];
		t[1]=a[l*3+1];
		t[2]=a[l*3+2];
		a[l*3+0]=a[j*3+0];
		a[l*3+1]=a[j*3+1];
		a[l*3+2]=a[j*3+2];
		a[j*3+0]=t[0];
		a[j*3+1]=t[1];
		a[j*3+2]=t[2];
		return j;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Rendering functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Draw_Triangles(int normal_mode=0)
	{
		glEnable(GL_LIGHTING);	
		for(int i=0; i<t_number; i++)
		{	
			int v0=T[i*3+0];
			int v1=T[i*3+1];
			int v2=T[i*3+2];
			if(v0==v1 || v0==v2 || v1==v2)	continue;
			glBegin(GL_TRIANGLES);
			glNormal3d(TN[i*3], TN[i*3+1], TN[i*3+2]);
			if(normal_mode)	glNormal3d(VN[v0*3], VN[v0*3+1], VN[v0*3+2]);
			glVertex3d(X[v0*3], X[v0*3+1], X[v0*3+2]);
			if(normal_mode)	glNormal3d(VN[v1*3], VN[v1*3+1], VN[v1*3+2]);
			glVertex3d(X[v1*3], X[v1*3+1], X[v1*3+2]);
			if(normal_mode)	glNormal3d(VN[v2*3], VN[v2*3+1], VN[v2*3+2]);
			glVertex3d(X[v2*3], X[v2*3+1], X[v2*3+2]);
			glEnd();
		}
		glDisable(GL_LIGHTING);
	}

	void Draw_Edges()
	{		
		glDisable(GL_LIGHTING);
		glColor3f(0, 0, 0);
		TYPE offset=0.0005f;		
		
		for(int i=0; i<t_number; i++)
		{
			glBegin(GL_LINE_LOOP);
			glVertex3d(
				X[T[i*3+0]*3+0]+VN[T[i*3+0]*3+0]*offset,
				X[T[i*3+0]*3+1]+VN[T[i*3+0]*3+1]*offset,
				X[T[i*3+0]*3+2]+VN[T[i*3+0]*3+2]*offset);
			glVertex3d(
				X[T[i*3+1]*3+0]+VN[T[i*3+1]*3+0]*offset,
				X[T[i*3+1]*3+1]+VN[T[i*3+1]*3+1]*offset,
				X[T[i*3+1]*3+2]+VN[T[i*3+1]*3+2]*offset);
			glVertex3d(
				X[T[i*3+2]*3+0]+VN[T[i*3+2]*3+0]*offset,
				X[T[i*3+2]*3+1]+VN[T[i*3+2]*3+1]*offset,
				X[T[i*3+2]*3+2]+VN[T[i*3+2]*3+2]*offset);
			glEnd();
		}
		glEnable(GL_LIGHTING);
	}

	void Draw_Vertices()
	{
		glDisable(GL_LIGHTING);
		glColor3f(1, 0, 0);
		for(int v=0; v<number; v++) //if(boundary[v])
			if(X[v*3+1]>1.42)
		{				
			glPushMatrix();
			glTranslatef(X[v*3+0], X[v*3+1], X[v*3+2]);
			glutSolidSphere(0.001, 10, 10);
			glPopMatrix();
		}
		glEnable(GL_LIGHTING);
	}
	
	void Render(int render_mode)
	{
		Build_VN();
		if(render_mode==0)
		{
			Draw_Triangles(true);
		}
		if(render_mode==1)
		{
			Draw_Triangles(true);
			Draw_Edges();
		}
		if(render_mode==2)
		{
			Draw_Triangles(true);
			Draw_Edges();
			Draw_Vertices();
		}
	}
};


#endif
