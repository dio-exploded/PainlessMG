varying vec3 fragNormal;
varying vec4 fragPosition;

uniform vec3 light_position;

void main() 
{ 
	vec3 V=normalize(vec3(fragPosition));
	vec3 N=-normalize(fragNormal);
	vec3 L=normalize(light_position);
	vec3 L2=normalize(vec3(2.0, 1.0, 1.0));
	vec3 L3=normalize(vec3(2.0, -1.0, 1.0));



	float base_color=(dot(N, L))*0.6;
	base_color=base_color+abs(dot(N, L2))*0.35;
	base_color=base_color+abs(dot(N, L3))*0.35;

	vec3 R;

	R=2*dot(L, N)*N-L;

	float color=pow(max(0, dot(R, V)), 60)*0.8;
	R=2*dot(L2, N)*N-L2;	
	color=color+pow(max(0, dot(R, V)), 60)*0.2;
	R=2*dot(L3, N)*N-L3;	
	color=color+pow(max(0, dot(R, V)), 60)*0.2;



	gl_FragColor = vec4(base_color*vec3(1.0, 0.8, 0.7)+color*vec3(1, 1, 1), 1);
	
 } 
