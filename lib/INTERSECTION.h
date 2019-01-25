///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2014, Huamin Wang
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
//  Intersection functions
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_INTERSECTION_H__
#define __WHMIN_INTERSECTION_H__
#include "MY_MATH.h"


///////////////////////////////////////////////////////////////////////////////////////////
////  Test whether a ray intersects a triangle
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
bool Ray_Triangle_Intersection(TYPE x0[], TYPE x1[], TYPE x2[], TYPE p0[], TYPE dir[], TYPE &min_t)
{
	TYPE e1[3], e2[3], s1[3];
	e1[0]=x1[0]-x0[0];
	e1[1]=x1[1]-x0[1];
	e1[2]=x1[2]-x0[2];
	e2[0]=x2[0]-x0[0];
	e2[1]=x2[1]-x0[1];
	e2[2]=x2[2]-x0[2];
	CROSS(dir, e2, s1);			
	TYPE divisor=DOT(s1, e1);
	if(divisor==0) return false;
	// Test the first barycentric coordinate
	TYPE tt[3];
	tt[0]=p0[0]-x0[0];
	tt[1]=p0[1]-x0[1];
	tt[2]=p0[2]-x0[2];
	TYPE b1=DOT(tt, s1);
	if(divisor>0 && (b1<0 || b1>divisor))		return false;
	if(divisor<0 && (b1>0 || b1<divisor))		return false;	
	// Test the second barycentric coordinate
	TYPE s2[3];
	CROSS(tt, e1, s2);
	TYPE b2=DOT(dir, s2);
	if(divisor>0 && (b2<0 || b1+b2>divisor))	return false;
	if(divisor<0 && (b2>0 || b1+b2<divisor))	return false;		
	// Compute t to intersection point
	min_t=DOT(e2, s2)/divisor;
	return min_t>0;
}



#endif

