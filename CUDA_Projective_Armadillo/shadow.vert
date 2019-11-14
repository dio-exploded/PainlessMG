//attribute vec4 vertex_position;     //In the local space
//attribute vec4 vertex_normal;       //In the local space


varying vec4 fragment_position;

void main()
{ 

	fragment_position = gl_Vertex; 	//In the world space
//	fragment_normal   = vec3(gl_ModelViewMatrix * vertex_normal);   //In the eye space



	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;   // In the clip space
}
