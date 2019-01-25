attribute vec4 position; 
attribute vec4 normal; 


varying vec3 fragNormal;
varying vec4 fragPosition;


void main()
{ 
	
	fragNormal = normalize(gl_NormalMatrix * vec3(normal));
	
	gl_Position = gl_ModelViewProjectionMatrix * position; 

	fragPosition = gl_Position;
}
