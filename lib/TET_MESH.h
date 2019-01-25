/////////////////////////////////////////////////////////////////////////////////////////////
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
//  TET mesh
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef	__WHMIN_TET_H__
#define __WHMIN_TET_H__
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "IO_FUNC.h"
#include "MY_MATH.h"
#include "INTERSECTION.h"
#include "DISTANCE.h"


template <class TYPE>
class TET_MESH
{
public:
	int		max_number;

	// Vertices
	int		number;
	TYPE*	X;
	TYPE*	M;
	// Tetrahedra
	int*	Tet;
	int		tet_number;
	TYPE*	Dm;
	TYPE*	inv_Dm;
	TYPE*	Vol;
	
	// triangles (for rendering)
	int		t_number;
	int*	T;
	TYPE*	VN;		//Vertex Normal
	TYPE*	TN;		//Triangle Normal


	TET_MESH(): number(0)
	{
		max_number	= 100000;
		X			= new TYPE	[max_number*3];
		M			= new TYPE	[max_number  ];
		Tet			= new int	[max_number*4];
		Dm			= new TYPE	[max_number*9];
		inv_Dm		= new TYPE	[max_number*9];
		Vol			= new TYPE	[max_number  ];
		T			= new int	[max_number*3];
		VN			= new TYPE	[max_number*3];
		TN			= new TYPE	[max_number*3];
	}
	
	~TET_MESH()
	{
		if(X)		delete[] X;
		if(M)		delete[] M;
		if(Tet)		delete[] Tet;
		if(Dm)		delete[] Dm;
		if(inv_Dm)	delete[] inv_Dm;
		if(Vol)		delete[] Vol;
		if(T)		delete[] T;
		if(VN)		delete[] VN;
		if(TN)		delete[] TN;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Initialize
///////////////////////////////////////////////////////////////////////////////////////////
	void Initialize()	
	{
		TYPE density=1;

		memset(M, 0, sizeof(TYPE)*number);
		for(int t=0; t<tet_number; t++)
		{
			int p0=Tet[t*4+0]*3;
			int p1=Tet[t*4+1]*3;
			int p2=Tet[t*4+2]*3;
			int p3=Tet[t*4+3]*3;

			Dm[t*9+0]=X[p1+0]-X[p0+0];
			Dm[t*9+3]=X[p1+1]-X[p0+1];
			Dm[t*9+6]=X[p1+2]-X[p0+2];
			Dm[t*9+1]=X[p2+0]-X[p0+0];
			Dm[t*9+4]=X[p2+1]-X[p0+1];
			Dm[t*9+7]=X[p2+2]-X[p0+2];
			Dm[t*9+2]=X[p3+0]-X[p0+0];
			Dm[t*9+5]=X[p3+1]-X[p0+1];
			Dm[t*9+8]=X[p3+2]-X[p0+2];
			Vol[t]=fabs(Matrix_Inverse_3(&Dm[t*9], &inv_Dm[t*9]))/6.0;

			M[p0/3]+=Vol[t]*density;
			M[p1/3]+=Vol[t]*density;
			M[p2/3]+=Vol[t]*density;
			M[p3/3]+=Vol[t]*density;
		}		
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Creator
///////////////////////////////////////////////////////////////////////////////////////////
	void Create_A_Tet()
	{
		X[ 0]=-0.05;
		X[ 1]=0;
		X[ 2]=-0.05;
		X[ 3]=0.1;
		X[ 4]=0;
		X[ 5]=0;
		X[ 6]=0;
		X[ 7]=0;
		X[ 8]=0.1;
		X[ 9]=0;
		X[10]=0.1;
		X[11]=0;
		X[12]=0;
		X[13]=-0.1;
		X[14]=0;
		number=5;

		Tet[0]=0;
		Tet[1]=1;
		Tet[2]=2;
		Tet[3]=3;
		Tet[4]=0;
		Tet[5]=1;
		Tet[6]=4;
		Tet[7]=2;
		tet_number=2;

		Build_Boundary_Triangles();
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Surface mesh builder
///////////////////////////////////////////////////////////////////////////////////////////
	void Build_Boundary_Triangles()
	{
		int *temp_T=new int[tet_number*4*4];

		for(int i=0; i<tet_number; i++)
		{
			temp_T[i*16+0]=Tet[i*4+0];
			temp_T[i*16+1]=Tet[i*4+1];
			temp_T[i*16+2]=Tet[i*4+2];
			temp_T[i*16+3]=1;

			temp_T[i*16+4]=Tet[i*4+0];
			temp_T[i*16+5]=Tet[i*4+2];
			temp_T[i*16+6]=Tet[i*4+3];
			temp_T[i*16+7]=1;

			temp_T[i*16+8]=Tet[i*4+0];
			temp_T[i*16+9]=Tet[i*4+3];
			temp_T[i*16+10]=Tet[i*4+1];
			temp_T[i*16+11]=1;

			temp_T[i*16+12]=Tet[i*4+1];
			temp_T[i*16+13]=Tet[i*4+3];
			temp_T[i*16+14]=Tet[i*4+2];
			temp_T[i*16+15]=1;
		}

		for(int i=0; i<tet_number*4; i++)
		{
			if(temp_T[i*4+1]<temp_T[i*4+0])
			{
				Swap(temp_T[i*4+0], temp_T[i*4+1]);
				temp_T[i*4+3]=(temp_T[i*4+3]+1)%2;
			}
			if(temp_T[i*4+2]<temp_T[i*4+0])
			{
				Swap(temp_T[i*4+0], temp_T[i*4+2]);
				temp_T[i*4+3]=(temp_T[i*4+3]+1)%2;
			}
			if(temp_T[i*4+2]<temp_T[i*4+1])
			{
				Swap(temp_T[i*4+1], temp_T[i*4+2]);
				temp_T[i*4+3]=(temp_T[i*4+3]+1)%2;
			}
		}

		QuickSort(temp_T, 0, tet_number*4-1);

		t_number=0;
		for(int i=0; i<tet_number*4; i++)
		{
			if(i!=tet_number*4-1 && temp_T[i*4+0]==temp_T[i*4+4] && temp_T[i*4+1]==temp_T[i*4+5] && temp_T[i*4+2]==temp_T[i*4+6])
			{
				i++;
				continue;
			}

			if(temp_T[i*4+3]==1)
			{
				T[t_number*3+0]=temp_T[i*4+0];
				T[t_number*3+1]=temp_T[i*4+1];
				T[t_number*3+2]=temp_T[i*4+2];
			}
			else
			{
				T[t_number*3+0]=temp_T[i*4+1];
				T[t_number*3+1]=temp_T[i*4+0];
				T[t_number*3+2]=temp_T[i*4+2];
			}
			t_number++;
		}
		delete []temp_T;
	}

	void QuickSort( int a[], int l, int r)
	{
		if( l < r ) 
		{
			int j=QuickSort_Partition(a, l, r);
			QuickSort(a, l, j-1);
			QuickSort(a, j+1, r);
		}
	}
	
	int QuickSort_Partition( int a[], int l, int r) 
	{
		int pivot[4], i, j, t[4];
		pivot[0] = a[l*4+0];
		pivot[1] = a[l*4+1];
		pivot[2] = a[l*4+2];
		pivot[3] = a[l*4+3];
		i = l; j = r+1;
		
		while( 1)
		{
			do ++i; while( (a[i*4+0]<pivot[0] || a[i*4+0]==pivot[0] && a[i*4+1]<pivot[1] || a[i*4+0]==pivot[0] && a[i*4+1]==pivot[1] && a[i*4+2]<=pivot[2]) && i <= r );
			do --j; while(  a[j*4+0]>pivot[0] || a[j*4+0]==pivot[0] && a[j*4+1]>pivot[1] || a[j*4+0]==pivot[0] && a[j*4+1]==pivot[1] && a[j*4+2]> pivot[2]);
			if( i >= j ) break;
			//Swap i and j
			t[0]=a[i*4+0];
			t[1]=a[i*4+1];
			t[2]=a[i*4+2];
			t[3]=a[i*4+3];
			a[i*4+0]=a[j*4+0];
			a[i*4+1]=a[j*4+1];
			a[i*4+2]=a[j*4+2];
			a[i*4+3]=a[j*4+3];
			a[j*4+0]=t[0];
			a[j*4+1]=t[1];
			a[j*4+2]=t[2];
			a[j*4+3]=t[3];
		}
		//Swap l and j
		t[0]=a[l*4+0];
		t[1]=a[l*4+1];
		t[2]=a[l*4+2];
		t[3]=a[l*4+3];
		a[l*4+0]=a[j*4+0];
		a[l*4+1]=a[j*4+1];
		a[l*4+2]=a[j*4+2];
		a[l*4+3]=a[j*4+3];
		a[j*4+0]=t[0];
		a[j*4+1]=t[1];
		a[j*4+2]=t[2];
		a[j*4+3]=t[3];
		return j;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Geometric functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Center(TYPE c[])
	{
		c[0]=c[1]=c[2]=0;
		TYPE mass_sum=0;
		for(int i=0; i<number; i++)
		{
			c[0]	 += X[i*3+0];
			c[1]	 += X[i*3+1];
			c[2]	 += X[i*3+2];
			mass_sum += 1;
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

	void Scale(TYPE s)
	{
		for(int i=0; i<number*3; i++)	X[i]*=s;
	}

	void Scale(TYPE sx, TYPE sy, TYPE sz)
	{
		for(int i=0; i<number; i++)
		{
			X[i*3+0]*=sx;
			X[i*3+1]*=sy;
			X[i*3+2]*=sz;
		}
	}

	void Rotate_X(TYPE angle)
	{
		for(int i=0; i<number; i++)
		{
			TYPE y=X[i*3+1];
			TYPE z=X[i*3+2];
			X[i*3+1]= y*cos(angle)+z*sin(angle);
			X[i*3+2]=-y*sin(angle)+z*cos(angle);
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Reader and Writer
///////////////////////////////////////////////////////////////////////////////////////////
	void Read_Original_File(char *name)
	{
		char filename[1024];
		int temp_value;
		int bound;

		sprintf(filename, "%s.node", name);
		FILE *fp;
		fp=fopen(filename, "r+");
		if(fp==NULL)	{printf("ERROR: file %s not open.\n", filename); return;}
		fscanf(fp, "%d %d %d %d\n", &number, &temp_value, &temp_value, &bound);
		if(bound==0)
			for(int i=0; i<number; i++)
			{
				float temp_x0, temp_x1, temp_x2;
				fscanf(fp, "%d %f %f %f\n", &temp_value, &temp_x0, &temp_x1, &temp_x2);
				X[i*3+0]=temp_x0;
				X[i*3+1]=temp_x1;
				X[i*3+2]=temp_x2;
			}
		else
			for(int i=0; i<number; i++)
			{
				float temp_x0, temp_x1, temp_x2;
				fscanf(fp, "%d %f %f %f %d\n", &temp_value, &temp_x0, &temp_x1, &temp_x2, &temp_value);
				X[i*3+0]=temp_x0;
				X[i*3+1]=temp_x1;
				X[i*3+2]=temp_x2;
			}

		fclose(fp);

		sprintf(filename, "%s.ele", name);
		fp=fopen(filename, "r+");
		if(fp==NULL)	{printf("ERROR: file %s not open.\n", filename); return;}
		fscanf(fp, "%d %d %d\n", &tet_number, &temp_value, &bound);
		
		if(bound==0)
			for(int i=0; i<tet_number; i++)
				fscanf(fp, "%d %d %d %d %d\n", &temp_value, &Tet[i*4+0], &Tet[i*4+1], &Tet[i*4+2], &Tet[i*4+3]);
		else if(bound==1)
			for(int i=0; i<tet_number; i++)
				fscanf(fp, "%d %d %d %d %d %d\n", &temp_value, &Tet[i*4+0], &Tet[i*4+1], &Tet[i*4+2], &Tet[i*4+3], &temp_value);
		fclose(fp);

		for(int i=0; i<tet_number; i++)
		{
			Tet[i*4+0]-=1;
			Tet[i*4+1]-=1;
			Tet[i*4+2]-=1;
			Tet[i*4+3]-=1;
		}
		Build_Boundary_Triangles();
	}

	void Write_Original_File(char *name)
	{
		char filename[1024];

		sprintf(filename, "%s.node", name);
		FILE *fp=fopen(filename, "w+");
		if(fp==NULL)	{printf("ERROR: file %s not open.\n", filename); return;}
		fprintf(fp, "%d %d %d %d\n", number, 3, 0, 0);
		for(int i=0; i<number; i++)
			fprintf(fp, "%d %f %f %f\n", i+1, X[i*3+0], X[i*3+1], X[i*3+2]);
		fclose(fp);

		sprintf(filename, "%s.ele", name);
		fp=fopen(filename, "w+");
		if(fp==NULL)	{printf("ERROR: file %s not open.\n", filename); return;}
		fprintf(fp, "%d %d %d\n", tet_number, 4, 0);

		for(int i=0; i<tet_number; i++)
			fprintf(fp, "%d %d %d %d %d\n", i+1, Tet[i*4+0]+1, Tet[i*4+1]+1, Tet[i*4+2]+1, Tet[i*4+3]+1);
		fclose(fp);
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Permutation using matlab input
///////////////////////////////////////////////////////////////////////////////////////////
	void Permutation(char *filename)
	{
		int* p=new int[number];
		int *q=new int[number];
		FILE *fp=fopen(filename, "r+");
		if(fp==NULL)	{printf("ERROR: file %s not open.\n", filename); return;}
		for(int i=0; i<number; i++)
		{
			fscanf(fp, "%d", &p[i]);
			p[i]-=1;
			//printf("read %d: %d\n", i, p[i]);
			q[p[i]]=i;
		}
		fclose(fp);

		TYPE*	new_X=new TYPE[number*3];
		int*	new_Tet=new int[tet_number*4];

		for(int i=0; i<number; i++)
		{
			int old_i=p[i];
			new_X[i*3+0]=X[old_i*3+0];
			new_X[i*3+1]=X[old_i*3+1];
			new_X[i*3+2]=X[old_i*3+2];
		}
				
		for(int t=0; t<tet_number; t++)
		{
			new_Tet[t*4+0]=q[Tet[t*4+0]];
			new_Tet[t*4+1]=q[Tet[t*4+1]];
			new_Tet[t*4+2]=q[Tet[t*4+2]];
			new_Tet[t*4+3]=q[Tet[t*4+3]];
		}

		memcpy(X, new_X, sizeof(TYPE)*number*3);
		memcpy(Tet, new_Tet, sizeof(int)*tet_number*4);
		delete[] new_X;
		delete[] new_Tet;
		delete[] p;
		delete[] q;
		
		Build_Boundary_Triangles();
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Rendering functions
///////////////////////////////////////////////////////////////////////////////////////////
	void Render(int visual_mode=0, int render_mode=0)
	{
		Build_VN();

		if(visual_mode==0)
		{			
			/*glDisable(GL_LIGHTING);
			glColor3f(1, 0, 0);
			for(int v=0; v<number; v++)
			{
				if(v!=0)	continue;

				glPushMatrix();
				glTranslatef(X[v*3+0], X[v*3+1], X[v*3+2]);
				glutSolidSphere(0.01, 5, 5);
				glPopMatrix();
			}			
			for(int t=0; t<tet_number; t++)
			{
				int v0=Tet[t*4+0];
				int v1=Tet[t*4+1];
				int v2=Tet[t*4+2];
				int v3=Tet[t*4+3];
				glBegin(GL_LINES);
				glVertex3dv(&X[v0*3]);	glVertex3dv(&X[v1*3]);
				glVertex3dv(&X[v0*3]);	glVertex3dv(&X[v2*3]);
				glVertex3dv(&X[v0*3]);	glVertex3dv(&X[v3*3]);
				glVertex3dv(&X[v1*3]);	glVertex3dv(&X[v2*3]);
				glVertex3dv(&X[v1*3]);	glVertex3dv(&X[v3*3]);
				glVertex3dv(&X[v2*3]);	glVertex3dv(&X[v3*3]);
				glEnd();
			}*/

			glEnable(GL_LIGHTING);		
			float diffuse_color[3]={0.7, 0.7, 1.0};
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse_color);
			for(int i=0; i<t_number; i++)
			{
				TYPE *p0=&X[T[i*3+0]*3];
				TYPE *p1=&X[T[i*3+1]*3];
				TYPE *p2=&X[T[i*3+2]*3];

				if(render_mode==0)	glNormal3f(TN[i*3+0], TN[i*3+1], TN[i*3+2]);
				glBegin(GL_TRIANGLES);
				if(render_mode)		glNormal3f(VN[T[i*3+0]*3+0], VN[T[i*3+0]*3+1], VN[T[i*3+0]*3+2]);
				glVertex3d(p0[0], p0[1], p0[2]);
				if(render_mode)		glNormal3f(VN[T[i*3+1]*3+0], VN[T[i*3+1]*3+1], VN[T[i*3+1]*3+2]);
				glVertex3d(p1[0], p1[1], p1[2]);
				if(render_mode)		glNormal3f(VN[T[i*3+2]*3+0], VN[T[i*3+2]*3+1], VN[T[i*3+2]*3+2]);
				glVertex3d(p2[0], p2[1], p2[2]);
				glEnd();
			}
		}
	}

	void Build_TN()
	{
		memset(TN, 0, sizeof(TYPE)*t_number*3);
		for(int i=0; i<t_number; i++)
		{
			TYPE *p0=&X[T[i*3+0]*3];
			TYPE *p1=&X[T[i*3+1]*3];
			TYPE *p2=&X[T[i*3+2]*3];
			Normal(p0, p1, p2, &TN[i*3]);
		}
	}

	void Build_VN()
	{
		memset(VN, 0, sizeof(TYPE)*number*3);
		Build_TN();

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

		TYPE length2, inv_length;
		for(int i=0; i<number; i++)
		{
			length2=VN[i*3+0]*VN[i*3+0]+VN[i*3+1]*VN[i*3+1]+VN[i*3+2]*VN[i*3+2];
			if(length2<1e-16f)	continue;
			inv_length=1.0f/sqrtf(length2);
			
			VN[i*3+0]*=inv_length;
			VN[i*3+1]*=inv_length;
			VN[i*3+2]*=inv_length;
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Select functions
///////////////////////////////////////////////////////////////////////////////////////////
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

};


#endif