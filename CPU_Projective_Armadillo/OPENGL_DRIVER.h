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
//  Class OPENGL_DRIVER
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __OPENGL_DRIVER_H__
#define __OPENGL_DRIVER_H__
#include <GL/glut.h>
#include "ARMADILLO.h"

#define NO_MOTION			0
#define ZOOM_MOTION			1
#define ROTATE_MOTION		2
#define TRANSLATE_MOTION	3

bool	idle_run=false;
int		file_id=0;
double	time_step=1/30.0;

ARMADILLO<double>		armadillo;
int		select_v=-1;
double	target[3];


///////////////////////////////////////////////////////////////////////////////////////////
//  Simulation entry here...
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
void Update(TYPE t)
{
	armadillo.Update(t, 64, select_v, target);
}


///////////////////////////////////////////////////////////////////////////////////////////
//  class OPENGL_DRIVER
///////////////////////////////////////////////////////////////////////////////////////////
class OPENGL_DRIVER
{
public:
	static int		file_id;

	//3D display configuration.
	static int		screen_width, screen_height;
	static int		mesh_mode;
	static int		render_mode;
	static double	zoom, swing_angle, elevate_angle;
	static double	center[3];
	static int		motion_mode, mouse_x, mouse_y;

	
	OPENGL_DRIVER(int *argc,char **argv)
	{
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
		glutInitWindowPosition(50,50);
		glutInitWindowSize(screen_width, screen_height);
		glutCreateWindow ("Experimental Cloth");
		glutDisplayFunc(Handle_Display);
		glutReshapeFunc(Handle_Reshape);
		glutKeyboardFunc(Handle_Keypress);
		glutMouseFunc(Handle_Mouse_Click);
		glutMotionFunc(Handle_Mouse_Move);
		glutSpecialFunc(Handle_SpecialKeypress);
		glutIdleFunc(Handle_Idle);
		Handle_Reshape(screen_width, screen_height);
		
		armadillo.Initialize(time_step);

		glutMainLoop();
	}

	static void Handle_Display()
	{	
		glLoadIdentity();
		glClearColor(0.8,0.8,0.8,0);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
						
		gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0);
		glRotated(elevate_angle, 1, 0, 0);
		glRotated(swing_angle, 0, 1, 0);
		glTranslatef(-center[0], -center[1], -center[2]);

		armadillo.Render();

		if(select_v!=-1)
		{
			glDisable(GL_LIGHTING);
			glPushMatrix();
			glColor3f(1, 0, 0);
			glTranslatef(armadillo.X[select_v*3+0], armadillo.X[select_v*3+1], armadillo.X[select_v*3+2]);
			glutSolidSphere(0.005, 10, 10);		
			glPopMatrix();

			glPushMatrix();
			glColor3f(0, 0, 1);
			glTranslatef(target[0], target[1], target[2]);
			glutSolidSphere(0.005, 10, 10);		
			glPopMatrix();
			glEnable(GL_LIGHTING);
		}
		glutSwapBuffers();
	}

	static void Handle_Idle()
	{
		if(idle_run)	Update(time_step);
		glutPostRedisplay();
	}
	
	static void Handle_Reshape(int w,int h)
	{
		screen_width=w,screen_height=h;
		glViewport(0,0,w, h);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(4, (double)w/(double)h, 1, 100);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glEnable(GL_DEPTH_TEST);
		{
			GLfloat LightDiffuse[] = { 1.0, 1.0, 1.0, 1};
			GLfloat LightPosition[] = { 0, 0, -100};
			glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
			glLightfv(GL_LIGHT0, GL_POSITION,LightPosition);
			glEnable(GL_LIGHT0);
		}			
		{
			GLfloat LightDiffuse[] = { 1.0, 1.0, 1.0, 1};
			GLfloat LightPosition[] = { 0, 0, 100};
			glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
			glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);
			glEnable(GL_LIGHT1);
		}
		glEnable(GL_LIGHTING);
		glShadeModel(GL_SMOOTH);		
		glutPostRedisplay();
	}
	
	static void Handle_Keypress(unsigned char key,int mousex,int mousey)
	{
		switch(key)
		{
			case 27: exit(0);
			case 'a':
			{ 
				zoom-=2; 
				if(zoom<0.3) zoom=0.3; 
				break;
			}
			case 'z':
			{
				zoom+=2; 
				break;
			}
			case 't':
			{
				render_mode=(render_mode+1)%4;
				break;
			}
			case 's':
			{
				//char filename[1024];
                int n=1;
                //TIMER timer;
				for(int i=0; i<n; i++)
				{					
					//printf("I: %d\n", i);
					Update(time_step);
					//clothing.Clear_Velocity();
					//if(0)
					//if(i%10==0)
					//{
					//	sprintf(filename, "results/shirt_%04d.state", i);
					//	clothing.Write_File(filename);
					//	sprintf(filename, "results/skirt%04d.state", i);
					//	skirt.Write_File(filename);
					//}
				}
               // printf("time: %f\n", timer.Get_Time());
				break;
			}
			case 'd':
			{
				armadillo.X[0]=8;
				armadillo.X[1]= 0;
				armadillo.X[2]= 0;

				break;
			}
			case '1':
			{
				idle_run=true;
				break;
			}
			case 'r':
			{
                //printf("dist: %f", Distance(&skirt.X[0], &skirt.X[40*3]));
				if(armadillo.Read_File("armadillo_float.state"))
					printf("Read armadillo.state successfully.\n");
				break;
			}
			case 'w':
			{
				if(armadillo.Write_File("armadillo_float.state"))
					printf("Write armadillo.state successfully.\n");
				//if(skirt.Write_File("skirt.state"))
				//	printf("Write skirt.state successfully.\n");
				break;
			}
		}
		glutPostRedisplay();
	}

	template <class TYPE>
	static void Get_Selection_Ray(int mouse_x, int mouse_y, TYPE* p, TYPE* q)
	{
		// Convert (x, y) into the 2D unit space
		double new_x = (double)(2*mouse_x)/(double)screen_width-1;
		double new_y = 1-(double)(2*mouse_y)/(double)screen_height;

		// Convert (x, y) into the 3D viewing space
		double M[16];
		glGetDoublev(GL_PROJECTION_MATRIX, M);
		//M is in column-major but inv_m is in row-major
		double inv_M[16];
		memset(inv_M, 0, sizeof(double)*16);
		inv_M[ 0]=1/M[0];
		inv_M[ 5]=1/M[5];
		inv_M[14]=1/M[14];
		inv_M[11]=-1;
		inv_M[15]=M[10]/M[14];
		double p0[4]={new_x, new_y, -1, 1}, p1[4];
		double q0[4]={new_x, new_y,  1, 1}, q1[4];
		Matrix_Vector_Product_4(inv_M, p0, p1);
		Matrix_Vector_Product_4(inv_M, q0, q1);
		
		// Convert (x ,y) into the 3D world space		
		glLoadIdentity();
		glTranslatef(center[0], center[1], center[2]);
		glRotatef(-swing_angle, 0, 1, 0);
		glRotatef(-elevate_angle, 1, 0, 0);
		glTranslatef(0, 0, zoom);
		glGetDoublev(GL_MODELVIEW_MATRIX, M);
		Matrix_Transpose_4(M, M);
		Matrix_Vector_Product_4(M, p1, p0);
		Matrix_Vector_Product_4(M, q1, q0);

		p[0]=p0[0]/p0[3];
		p[1]=p0[1]/p0[3];
		p[2]=p0[2]/p0[3];
		q[0]=q0[0]/q0[3];
		q[1]=q0[1]/q0[3];
		q[2]=q0[2]/q0[3];
	}


	static void Handle_SpecialKeypress(int key, int x, int y)
	{		
		if(key==100)		swing_angle+=3;
		else if(key==102)	swing_angle-=3;
		else if(key==103)	elevate_angle-=3;
		else if(key==101)	elevate_angle+=3;
		Handle_Reshape(screen_width, screen_height); 
		glutPostRedisplay();
	}

	static void Handle_Mouse_Move(int x, int y)
	{
		if(motion_mode!=NO_MOTION)
		{
			if(motion_mode==ROTATE_MOTION) 
			{
				swing_angle   += (double)(x - mouse_x)*360/(double)screen_width;
				elevate_angle += (double)(y - mouse_y)*180/(double)screen_height;
		        if     (elevate_angle> 90)	elevate_angle = 90;
				else if(elevate_angle<-90)	elevate_angle = -90;
			}
			if(motion_mode==ZOOM_MOTION)	zoom+=0.05 * (y-mouse_y);
			if(motion_mode==TRANSLATE_MOTION)
			{
				center[0] -= 0.1*(mouse_x - x);
				center[2] += 0.1*(mouse_y - y);
			}
			mouse_x=x;
			mouse_y=y;
			glutPostRedisplay();
		}
		if(select_v!=-1)
		{
			float	p[3], q[3];
			Get_Selection_Ray(x, y, p, q);
			double dir[3];
			dir[0]=q[0]-p[0];
			dir[1]=q[1]-p[1];
			dir[2]=q[2]-p[2];
			Normalize(dir);
			double diff[3];
			diff[0]=armadillo.X[select_v*3+0]-p[0];
			diff[1]=armadillo.X[select_v*3+1]-p[1];
			diff[2]=armadillo.X[select_v*3+2]-p[2];
			double dist=DOT(diff, dir);
			target[0]=p[0]+dist*dir[0];
			target[1]=p[1]+dist*dir[1];
			target[2]=p[2]+dist*dir[2];
			glutPostRedisplay();
		}
	}

	static void Handle_Mouse_Click(int button, int state, int x, int y)
	{		
		if(state==GLUT_UP) 
		{
			select_v	= -1;
			motion_mode	= NO_MOTION;
		}
		if(state==GLUT_DOWN)
		{
			double	p[3], q[3];
			Get_Selection_Ray(x, y, p, q);
			select_v=-1;
			armadillo.Select(p, q, select_v);

			// Set up the motion target
			if(select_v!=-1)
			{
				double dir[3];
				dir[0]=q[0]-p[0];
				dir[1]=q[1]-p[1];
				dir[2]=q[2]-p[2];
				Normalize(dir);
				double diff[3];
				diff[0]=armadillo.X[select_v*3+0]-p[0];
				diff[1]=armadillo.X[select_v*3+1]-p[1];
				diff[2]=armadillo.X[select_v*3+2]-p[2];
				double dist=DOT(diff, dir);
				target[0]=p[0]+dist*dir[0];
				target[1]=p[1]+dist*dir[1];
				target[2]=p[2]+dist*dir[2];
			}
			else //No selection, perform camera change
			{
				int modif = glutGetModifiers();
				if (modif & GLUT_ACTIVE_SHIFT)		motion_mode = ZOOM_MOTION;
				else if (modif & GLUT_ACTIVE_CTRL)	motion_mode = TRANSLATE_MOTION;
				else								motion_mode = ROTATE_MOTION;
				mouse_x=x;
				mouse_y=y;
			}
		}		

		glutPostRedisplay();
	}
};


int		OPENGL_DRIVER::file_id			=	0;
int		OPENGL_DRIVER::screen_width		=	1024;
int		OPENGL_DRIVER::screen_height	=	768;
int		OPENGL_DRIVER::mesh_mode		=	0;
int		OPENGL_DRIVER::render_mode		=	0;
double	OPENGL_DRIVER::zoom				=	30;
double	OPENGL_DRIVER::swing_angle		=	-0;
double	OPENGL_DRIVER::elevate_angle	=	0; 
double	OPENGL_DRIVER::center[3]		=	{0, 0, 0};
int		OPENGL_DRIVER::motion_mode		=	NO_MOTION;
int		OPENGL_DRIVER::mouse_x			=	0;
int		OPENGL_DRIVER::mouse_y			=	0;


#endif

