//**************************************************************************************
//  Copyright (C) 2002 - 2013, Huamin Wang
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
//**************************************************************************************
// MY_GLSL
//**************************************************************************************
#ifndef __MY_GLSL_H__
#define __MY_GLSL_H__
#include <GL/glew.h> 
#include <GL/glut.h> 
#include <stdio.h>

//**************************************************************************************
// Check the status of the GPU
//**************************************************************************************
void Check_GPU_Status()
{
	const GLubyte *renderer = glGetString(GL_RENDERER); 
	const GLubyte *vendor = glGetString(GL_VENDOR); 
	const GLubyte *version = glGetString(GL_VERSION); 
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION); 
	GLint major, minor; 
	glGetIntegerv(GL_MAJOR_VERSION, &major); 
	glGetIntegerv(GL_MINOR_VERSION, &minor); 
  
	printf(" ----------------- Checking graphics capability ...\n"); 
	printf(" GL Vendor: %s \n", vendor); 
	printf(" GL Renderer: %s \n", renderer); 
	printf(" GL version: %s\n", version); 
	printf(" GL version: %d.%d\n", major, minor); 
	printf(" GLSL version: %s\n", glslVersion); 

	//Now check the availability of shaders 
	if (! GLEW_ARB_vertex_program) printf(" ARB vertex program is not supported!!\n");  
	else printf(" ARB vertex program is supported!!\n");
	if (! GLEW_ARB_vertex_shader) printf(" ARB vertex shader is not supported!!\n");  
	else printf(" ARB vertex shader is supported!!\n");
	if (! GLEW_ARB_fragment_program) printf(" ARB fragment program is not supported!!\n");  
	else printf(" ARB fragment program is supported!!\n");
	if (! GLEW_ARB_fragment_shader) printf(" ARB fragment shader is not supported!!\n");  
	else printf(" ARB fragment shader is supported!!\n"); 

	//Another way to query the shaders support 
	if(glewGetExtension("GL_ARB_fragment_shader")      != GL_TRUE ||
       glewGetExtension("GL_ARB_vertex_shader")        != GL_TRUE ||
       glewGetExtension("GL_ARB_shader_objects")       != GL_TRUE ||
       glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
	{
		fprintf(stderr, "Driver does not support OpenGL Shading Language\n");
		exit(1);
    }
	else fprintf(stderr, "GLSL supported and ready to go\n");

	printf(" -----------------  checking graphics capability done. \n");
} 

//**************************************************************************************
//  Read shaders from the disk into the main memory
//**************************************************************************************
int Read_Shader(char *name, char **shader_text)
{	
    FILE* fp = fopen(name, "r+");
    if(fp==NULL) return 0;
	//Calculate the file size and allocate the shader_content
	fseek(fp, 0L, SEEK_END);
    int size=ftell(fp)+1;
	if(*shader_text)	delete[] (*shader_text);
	*shader_text = new char[size];
	//Read the shader file
	int count=0;
	fseek(fp, 0, SEEK_SET);
	count = fread(*shader_text, 1, size, fp);
	(*shader_text)[count] = '\0';
	//The end
    fclose(fp);
    return count;
}

bool Read_Shader_Source(char *shader_name, GLchar **vertexShader, GLchar **fragmentShader)
{
	char vert_shader_name[1024];
	sprintf(vert_shader_name, "%s.vert", shader_name);
	if(!Read_Shader(vert_shader_name, vertexShader))
    {
        printf("Cannot read the file %s.\n", vert_shader_name);
        return false;
    }
	char frag_shader_name[1024];
	sprintf(frag_shader_name, "%s.frag", shader_name);
	if(!Read_Shader(frag_shader_name, fragmentShader))
    {
        printf("Cannot read the file %s.\n", frag_shader_name);
        return false;
    }
    return true;
}


//**************************************************************************************
//  GLSL setup routine
//**************************************************************************************
GLuint Setup_GLSL(char *name)
{
	//Step 1: Check the status of the GPU
	Check_GPU_Status();

	//Step 2: Create the objects
	GLuint programObject;
	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;
	if(!(programObject=glCreateProgram()))
	{   
		printf("Error creating shader program object.\n"); 
		exit(1); 
	} 
	else printf("Succeeded creating shader program object.\n"); 
	if(!(vertexShaderObject=glCreateShader(GL_VERTEX_SHADER)))
	{
	  printf("Error creating vertex shader object.\n"); 
	  exit(1); 
	} 
	else printf("Succeeded creating vertex shader object.\n"); 
	if(!(fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER)))
	{
		printf("Error creating fragment shader object.\n"); 
		exit(1); 
	} 
	else printf("Succeeded creating fragment shader object.\n"); 

	//Step 3: Load the shaders from the disk into the two shader objects
	GLchar* vertexShaderSource=0;
	GLchar* fragmentShaderSource=0;
	Read_Shader_Source(name, &vertexShaderSource, &fragmentShaderSource); 
	glShaderSource(vertexShaderObject,1,(const GLchar**)&vertexShaderSource,NULL);
	glShaderSource(fragmentShaderObject,1,(const GLchar**)&fragmentShaderSource,NULL);
	delete[] vertexShaderSource;
	delete[] fragmentShaderSource;

	//Step 4: Compile the vertex shader 
	glCompileShader(vertexShaderObject);
	//If there is any error, print out the error log
	GLint result; 
	glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &result); 
	if(result==GL_FALSE) 
	{
		printf(" vertex shader compilation failed!\n"); 
		GLint logLen; 
		glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &logLen); 
		if (logLen > 0) 
		{
			char* log=new char[logLen]; 
			GLsizei written; 
			glGetShaderInfoLog(vertexShaderObject, logLen, &written, log); 
			printf("Shader log: \n %s", log); 
			delete[] log;
		}
	}

	//Step 5: Compile the fragment shader
	glCompileShader(fragmentShaderObject);
	//If there is any error, print out the error log
	glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &result); 
	if(result==GL_FALSE) 
	{
		printf(" fragment shader compilation failed!\n"); 
		GLint logLen; 
		glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &logLen); 
		if(logLen > 0) 
		{
			char* log = new char[logLen]; 
			GLsizei written; 
			glGetShaderInfoLog(fragmentShaderObject, logLen, &written, log); 
			printf("Shader log: \n %s", log); 
			delete[] log;
		}
	}

	//Step 6: Attach the shader objects and link the program object
	glAttachShader(programObject, vertexShaderObject);
	glAttachShader(programObject, fragmentShaderObject);
	glLinkProgram(programObject);
	
	return(programObject); 
}




#endif