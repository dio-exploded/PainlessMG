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
// Class OPENGL_DRIVER
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __OPENGL_DRIVER_H__
#define __OPENGL_DRIVER_H__
#include "../lib/MY_GLSL.h"
#include "../lib/BMP_IO.h"

//#define BENCHMARKG
#define SETTINGF
//#define enable_PD
//#define WHM

#include "CLOTHING.h"
#define	USE_CUDA
#define NO_MOTION			0
#define ZOOM_MOTION			1
#define ROTATE_MOTION		2
#define TRANSLATE_MOTION	3

int		screen_width	= 1024;
int		screen_height	= 768;
float	zoom			= 28;
float	swing_angle		= -52;
float	elevate_angle	= 10;
float	center[3]		={0, -0.4, 0};
bool	idle_run=false;
bool	idle_scrnshot = false;
bool	idle_export_obj = false;
int		file_id=0;
float	time_step=1/30.0;
int sub_step = 1;
//int pd_iters = 1;

//#ifdef BENCHMARKG
//FILE	*benchmark = fopen("benchmark\\TEST.txt", "w");
//#endif

CLOTHING<float>		clothing;
int		select_v=-1;
float	target[3]={0, 0, 0};


///////////////////////////////////////////////////////////////////////////////////////////
//  Simulation entry here!
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
void Update(TYPE t)
{
	TYPE dir[3];
	if(select_v!=-1)
	{
		dir[0]=target[0]-clothing.X[select_v*3+0];
		dir[1]=target[1]-clothing.X[select_v*3+1];
		dir[2]=target[2]-clothing.X[select_v*3+2];
		TYPE dir_length=Normalize(dir);
		if(dir_length>0.2)	dir_length=0.2;
		dir[0]=dir_length*dir[0];
		dir[1]=dir_length*dir[1];
		dir[2]=dir_length*dir[2];
	}
	clothing.Update(t, 400, dir);

}

///////////////////////////////////////////////////////////////////////////////////////////
//  Shader functions
///////////////////////////////////////////////////////////////////////////////////////////
GLuint depth_FBO		= 0;
GLuint depth_texture	= 0;

GLuint shadow_program	= 0;
GLuint phong_program	= 0;

GLuint vertex_handle	= 0;
GLuint normal_handle	= 0;
GLuint triangle_handle	= 0;

float	light_position[3]={-2, 2, 4};

void Init_GLSL()
{
	//Init GLEW
	GLenum err = glewInit(); 
	if(err!= GLEW_OK)  printf(" Error initializing GLEW! \n"); 
	else printf("Initializing GLEW succeeded!\n");

	//Init depth texture and FBO
	glGenFramebuffers(1, &depth_FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
	// Depth texture. Slower than a depth buffer, but you can sample it later in your shader
	glGenTextures(1, &depth_texture);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT, 1024, 1024, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );	
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_texture, 0);
	glDrawBuffer(GL_NONE);
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) printf("Init_Shadow_Map failed.\n");
	
	//Load shader program
	shadow_program	= Setup_GLSL("shadow");
	phong_program	= Setup_GLSL("phong");

	//Create VBO
	glGenBuffers(1, &vertex_handle);
	glGenBuffers(1, &normal_handle);
	glGenBuffers(1, &triangle_handle);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle); 
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*clothing.t_number*3, clothing.T, GL_STATIC_DRAW);
}

void Create_Shadow_Map(char* filename=0)
{
	glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
	glViewport(0, 0, 1024, 1024); // Render on the whole framebuffer, complete from the lower left corner to the upper right

	// glEnable(GL_CULL_FACE);
	// glCullFace(GL_BACK); // Cull back-facing triangles -> draw only front-facing triangles
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-2,2,-2,2, 0, 20);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(light_position[0], light_position[1], light_position[2], 0, 0, 0 , 0, 1, 0);
	//Use fixed program
	glUseProgram(0);

	glPushMatrix();
	glRotated(elevate_angle, 1, 0, 0);
	glRotated(swing_angle, 0, 1, 0);
	glTranslatef(-center[0], -center[1], -center[2]);
	clothing.Render(0);
	glPopMatrix();

	//Also we need to set up the projection matrix for shadow texture	
	// This is matrix transform every coordinate x,y,z
	// Moving from unit cube [-1,1] to [0,1]  
	float bias[16] = {	0.5, 0.0, 0.0, 0.0, 
						0.0, 0.5, 0.0, 0.0,
						0.0, 0.0, 0.5, 0.0,
						0.5, 0.5, 0.5, 1.0};
	
	// Grab modelview and transformation matrices
	float	modelView[16];
	float	projection[16];
	float	biased_MVP[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glGetFloatv(GL_PROJECTION_MATRIX, projection);	
		
	glMatrixMode(GL_MODELVIEW);	
	glLoadIdentity();	
	glLoadMatrixf(bias);
	// concatating all matrice into one.
	glMultMatrixf(projection);
	glMultMatrixf(modelView);

	glGetFloatv(GL_MODELVIEW_MATRIX, biased_MVP);

	glUseProgram(shadow_program);
	GLuint m = glGetUniformLocation(shadow_program, "biased_MVP"); // get the location of the biased_MVP matrix
	glUniformMatrix4fv(m, 1, GL_FALSE, biased_MVP); 
}


///////////////////////////////////////////////////////////////////////////////////////////
//  class OPENGL_DRIVER
///////////////////////////////////////////////////////////////////////////////////////////
class OPENGL_DRIVER
{
public:
	static int		file_id;

	//3D display configuration.
	static int		mesh_mode;
	static int		render_mode;
	static int		motion_mode, mouse_x, mouse_y;

	
	OPENGL_DRIVER(int *argc,char **argv)
	{
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
		glutInitWindowPosition(300,100);
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
		
		clothing.Initialize(time_step/sub_step);

		Init_GLSL();
		glutMainLoop();
	}

	~OPENGL_DRIVER()
	{}

	static void Handle_Display()
	{	
		GLuint uniloc;
		Create_Shadow_Map();
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0,0, screen_width, screen_height);
				
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(4, (double)screen_width/(double)screen_height, 1, 100);
		glMatrixMode(GL_MODELVIEW);
		glShadeModel(GL_SMOOTH);
		
		glLoadIdentity();
		glClearColor(1,1,1,0);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);		
		gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0);

		glDisable(GL_LIGHTING);		
		glEnable(GL_DEPTH_TEST);
		glUseProgram(shadow_program);
		uniloc = glGetUniformLocation(shadow_program, "shadow_texture");
		glUniform1i(uniloc, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, depth_texture);
		uniloc = glGetUniformLocation(shadow_program, "light_position");
		glUniform3fv(uniloc, 1, light_position);

		glBegin(GL_POLYGON);
		glVertex3f(-10, -10, -1);
		glVertex3f( 10, -10, -1);
		glVertex3f( 10,  10, -1);
		glVertex3f(-10,  10, -1);
		glEnd();
				
		
		glRotated(elevate_angle, 1, 0, 0);
		glRotated(swing_angle, 0, 1, 0);
		glTranslatef(-center[0], -center[1], -center[2]);
		//glUseProgram(0);
		//clothing.Render(0);

		clothing.Build_VN();
		glUseProgram(phong_program);
		uniloc = glGetUniformLocation(phong_program, "light_position");
		glUniform3fv(uniloc, 1, light_position);

		GLuint c0=glGetAttribLocation(phong_program, "position");
		GLuint c1=glGetAttribLocation(phong_program, "normal");
		glEnableVertexAttribArray(c0); 
		glEnableVertexAttribArray(c1); 
		glBindBuffer(GL_ARRAY_BUFFER, vertex_handle); 
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*clothing.number*3, clothing.X, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(c0,3,GL_FLOAT, GL_FALSE, sizeof(float)*3,(char*) NULL+0);		
		glBindBuffer(GL_ARRAY_BUFFER, normal_handle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*clothing.number*3, clothing.VN, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(c1,3,GL_FLOAT, GL_FALSE, sizeof(float)*3,(char*) NULL+0); 
	
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle);
		glDrawElements(GL_TRIANGLES, clothing.t_number*3, GL_UNSIGNED_INT, (char*) NULL+0);
		

		if(select_v!=-1)
		{
			glUseProgram(0);
			glDisable(GL_LIGHTING);
			glEnable(GL_BLEND);
			glPushMatrix();
			glColor4f(1, 0, 0, 0.5);
			glTranslatef(clothing.X[select_v*3+0], clothing.X[select_v*3+1], clothing.X[select_v*3+2]);
			glutWireSphere(sqrt(RADIUS_SQUARED), 12, 12);
			glPopMatrix();
		}

		if (clothing.objects_num)
		{
			for (int i = 0; i != clothing.objects_num; i++)
			{
				auto obj = clothing.objects[i];
				switch (obj.type)
				{
				case Cylinder:
					glUseProgram(0);
					glDisable(GL_LIGHTING);
					glEnable(GL_BLEND);
					glPushMatrix();
					glColor4f(0, 0, 1, 0.5);
					glTranslatef(obj.s_cx, obj.s_cy, 0);
					glutWireSphere(obj.s_r, 12, 12);
					glPopMatrix();
					break;
				case Sphere:
					glUseProgram(0);
					glDisable(GL_LIGHTING);
					glEnable(GL_BLEND);
					glPushMatrix();
					glColor4f(0, 0, 1, 0.5);
					glTranslatef(obj.s_cx, obj.s_cy, obj.s_cz);
					glutWireSphere(obj.s_r, 12, 12);
					glPopMatrix();
					break;
				default:
					break;
				}
			}
		}

		//Draw FPS
		glLoadIdentity();
		glDisable(GL_DEPTH_TEST);
		glUseProgram(0);
		glColor3f(0, 0, 0);
		glRasterPos3f(-1.35, 0.95, -30);
		char text[1024];
		sprintf_s(text, "FPS: %4.2f", clothing.fps);
		if (idle_scrnshot)
		{
			strcat(text, " | Recording");
		}
		if (idle_export_obj)
		{
			strcat(text, " | Exporting OBJ");
		}
		if (idle_scrnshot | idle_export_obj)
		{
			char frame_text[1024];
			sprintf_s(frame_text, " | Frame#: %d", file_id);
			strcat(text, frame_text);
		}
		for(int i=0; text[i]!='\0'; i++)
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);
		glEnable(GL_DEPTH_TEST);

		glutSwapBuffers();
	}

	static void Handle_Idle()
	{
		if (idle_run)	Update(time_step / sub_step);
		//if (idle_run&idle_scrnshot)
		//{
		//	char filename[1024];
		//	float* pixels = new float[screen_width*screen_height * 3];
		//	glReadPixels(0, 0, screen_width, screen_height, GL_RGB, GL_FLOAT, pixels);
		//	sprintf_s(filename, "output/clothing_%04d.bmp", file_id);
		//	BMP_Write(filename, pixels, screen_width, screen_height);
		//	delete[] pixels;

		//	file_id += 1;
		//}
		if (idle_run)
		{
			char filename[1024];

			if (idle_scrnshot)
			{
				float* pixels = new float[screen_width*screen_height * 3];
				glReadPixels(0, 0, screen_width, screen_height, GL_RGB, GL_FLOAT, pixels);
				sprintf_s(filename, "output/screenshots/screenshot_%04d.bmp", file_id);
				BMP_Write(filename, pixels, screen_width, screen_height);
				delete[] pixels;
			}
			if (idle_export_obj)
			{
				sprintf_s(filename, "output/mesh/mesh_%04d.obj", file_id);
				clothing.Write_OBJ(filename);
			}

			file_id += 1;
		}

		glutPostRedisplay();
	}
	
	static void Handle_Reshape(int w,int h)
	{
		screen_width=w,screen_height=h;

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
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
				break;
			}
			case '1':
			{
				idle_run=true;
				break;
			}
			case ' ':
			{
				idle_run = !idle_run;
				break;
			}
			case 'r':
			{
				if(clothing.Read_File("clothing_float.state"))
					printf("Read armadillo.state successfully.\n");
				cudaMemcpy(clothing.dev_X,	clothing.X, sizeof(float)*3*clothing.number, cudaMemcpyHostToDevice);
				cudaMemcpy(clothing.dev_V,	clothing.V, sizeof(float)*3*clothing.number, cudaMemcpyHostToDevice);
				break;
			}
			case 'w':
			{
				if(clothing.Write_File("clothing_float.state"))
					printf("Write armadillo.state successfully.\n");
				//if(skirt.Write_File("skirt.state"))
				//	printf("Write skirt.state successfully.\n");
				break;
			}
			case '.':case '>':
			{
				file_id+=10;
				//char filename[1024];
				//sprintf(filename, "results/shirt_%04d.state", file_id);
				//clothing.Read_File(filename);
				//sprintf(filename, "results/skirt%04d.state", file_id);
				//skirt.Read_File(filename);
				//printf("fid: %d\n", file_id);
				break;
			}
			case ',':case '<':
			{
				file_id-=10;
				//char filename[1024];
				//sprintf(filename, "results/shirt_%04d.state", file_id);
				//clothing.Read_File(filename);
				//sprintf(filename, "results/skirt%04d.state", file_id);
				//skirt.Read_File(filename);	
				//printf("fid: %d\n", file_id);
				break;
			}
			case '9':
			{
				//idle_render=!idle_render;
				idle_scrnshot = !idle_scrnshot;
				break;
			}
			case '0':
			{
				idle_export_obj = !idle_export_obj;
				break;
			}	
			case 'p':
			{
				char filename[1024];

				float* pixels = new float[screen_width*screen_height * 3];
				glReadPixels(0, 0, screen_width, screen_height, GL_RGB, GL_FLOAT, pixels);
				sprintf_s(filename, "output/screenshot.bmp");
				BMP_Write(filename, pixels, screen_width, screen_height);

				delete[] pixels;
				sprintf_s(filename, "output/mesh.obj");
				clothing.Write_OBJ(filename);

				break;
			}
			case 'y':
			{
				if (clothing.Write_File("clothing_target_float.state"))
					printf("Write clothing_target successfully.\n");
				break;
			}
			case 'u':
			{
				if (clothing.Read_File("clothing_target_float.state"))
					printf("Read clothing_target successfully.\n");
				cudaMemcpy(clothing.dev_target_X, clothing.X, sizeof(float) * 3 * clothing.number, cudaMemcpyHostToDevice);
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
				center[0] -= 0.01*(mouse_x - x);
				center[2] += 0.01*(mouse_y - y);
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
			diff[0]=clothing.X[select_v*3+0]-p[0];
			diff[1]=clothing.X[select_v*3+1]-p[1];
			diff[2]=clothing.X[select_v*3+2]-p[2];
			double dist=DOT(diff, dir);
			target[0]=p[0]+dist*dir[0];
			target[1]=p[1]+dist*dir[1];
			target[2]=p[2]+dist*dir[2];
			glutPostRedisplay();
		}
	}

	static void Handle_Mouse_Click(int button, int state, int x, int y)
	{	
		select_v=-1;
		if(state==GLUT_UP)	motion_mode	= NO_MOTION;		
		if(state==GLUT_DOWN)
		{
			float	p[3], q[3];
			Get_Selection_Ray(x, y, p, q);
			clothing.Select(p, q, select_v);

			// Set up the motion target
			if(select_v!=-1)
			{
				double dir[3];
				dir[0]=q[0]-p[0];
				dir[1]=q[1]-p[1];
				dir[2]=q[2]-p[2];
				Normalize(dir);
				double diff[3];
				diff[0]=clothing.X[select_v*3+0]-p[0];
				diff[1]=clothing.X[select_v*3+1]-p[1];
				diff[2]=clothing.X[select_v*3+2]-p[2];
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
		
		clothing.Reset_More_Fixed(select_v);

		glutPostRedisplay();
	}
};


int		OPENGL_DRIVER::file_id			=	0;
int		OPENGL_DRIVER::mesh_mode		=	0;
int		OPENGL_DRIVER::render_mode		=	0;
int		OPENGL_DRIVER::motion_mode		=	NO_MOTION;
int		OPENGL_DRIVER::mouse_x			=	0;
int		OPENGL_DRIVER::mouse_y			=	0;


#endif

