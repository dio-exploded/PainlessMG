varying vec3 fragNormal;
varying vec4 fragPosition;

uniform vec3 light_position;

void main() 
{ 
	vec3 V=normalize(vec3(fragPosition));
	vec3 N=normalize(fragNormal);
	vec3 L=normalize(light_position);
	vec3 L2=normalize(vec3(2.0,  1.0, 1.0));
	vec3 L3=normalize(vec3(2.0, -1.0, 1.0));


	L=normalize(vec3(-1, 0, 4));


	vec3 base_color=0;

	if(dot(N, L)>0)
		base_color=abs(dot(N, L))*0.9*vec3(0.8, 1.0, 0.6);
	else
		base_color=abs(dot(N, L))*0.9*vec3(0.6, 0.8, 1.0);


	//base_color=base_color-(0, dot(N, L2))*0.1*vec3(0.8, 1.0, 0.7);
	

	//base_color=base_color+max(0, dot(N, L3))*0.5*vec3(0.8, 1.0, 0.7);



	gl_FragColor = vec4(base_color, 1);
		
	
 } 
