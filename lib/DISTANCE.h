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
//  DISTANCE functions
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_DISTANCE_H__
#define __WHMIN_DISTANCE_H__
#include "MY_MATH.h"


///////////////////////////////////////////////////////////////////////////////////////////
////  Squared vertex-edge distance
////	r is the barycentric weight of the closest point: 1-r, r.
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
TYPE Squared_VE_Distance(TYPE xi[], TYPE xa[], TYPE xb[], TYPE &r, TYPE *N=0)
{
	TYPE xia[3], xba[3];
	for(int n=0; n<3; n++)
	{
		xia[n]=xi[n]-xa[n];
		xba[n]=xb[n]-xa[n];
	}
	TYPE xia_xba=DOT(xia, xba);
	TYPE xba_xba=DOT(xba, xba);
	if(xia_xba<0)				r=0;
	else if(xia_xba>xba_xba)	r=1;
	else						r=xia_xba/xba_xba;
	TYPE _N[3];
	if(N==0)	N=_N;
	N[0]=xi[0]-xa[0]*(1-r)-xb[0]*r;
	N[1]=xi[1]-xa[1]*(1-r)-xb[1]*r;
	N[2]=xi[2]-xa[2]*(1-r)-xb[2]*r;
	return DOT(N, N);
}

///////////////////////////////////////////////////////////////////////////////////////////
////  Squared edge-edge distance
////	r and s are the barycentric weights of the closest point: 1-r, r; 1-s, s
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
TYPE Squared_EE_Distance(TYPE xi[], TYPE xj[], TYPE xa[], TYPE xb[], TYPE &r, TYPE &s, TYPE *N=0)
{
	TYPE  xba[3],  xji[3],  xai[3];
	for(int n=0; n<3; n++)
	{
		xba[n]=xb[n]-xa[n];
		xji[n]=xj[n]-xi[n];
		xai[n]=xa[n]-xi[n];
	}

	TYPE _N[3];
	if(N==0)	N=_N;
	CROSS(xji, xba, N);
	TYPE nn=DOT(N, N);

	TYPE temp[3];
	CROSS(xai, xji, temp);
	TYPE weight_aiji=DOT(N, temp);
	CROSS(xai, xba, temp);
	TYPE weight_aiba=DOT(N, temp);

	if(nn>1e-24f && weight_aiji>=0 && weight_aiji<=nn && weight_aiba>=0 && weight_aiba<=nn)
	{
		r=weight_aiba/nn;
		s=weight_aiji/nn;
	}
	else
	{
		TYPE min_distance=MY_INFINITE;
		TYPE distance, v;

		if(weight_aiba<0 && ((distance=Squared_VE_Distance(xi, xa, xb, v))<min_distance))
		{
			min_distance=distance;
			r=0;
			s=v;
		}
		if(weight_aiba>nn && ((distance=Squared_VE_Distance(xj, xa, xb, v))<min_distance))
		{
			min_distance=distance;
			r=1;
			s=v;
		}
		if(weight_aiji<0 && ((distance=Squared_VE_Distance(xa, xi, xj, v))<min_distance))
		{
			min_distance=distance;
			r=v;
			s=0;
		}
		if(weight_aiji>nn && ((distance=Squared_VE_Distance(xb, xi, xj, v))<min_distance))
		{
			min_distance=distance;
			r=v;
			s=1;
		}
	}
	N[0]=xi[0]*(1-r)+xj[0]*r-xa[0]*(1-s)-xb[0]*s;
	N[1]=xi[1]*(1-r)+xj[1]*r-xa[1]*(1-s)-xb[1]*s;
	N[2]=xi[2]*(1-r)+xj[2]*r-xa[2]*(1-s)-xb[2]*s;
	return DOT(N, N);
}

///////////////////////////////////////////////////////////////////////////////////////////
////  Squared vertex-triangle distance
////	bb and bc are the barycentric weights of the closest point: 1-bb-bc, bb, bc.
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
TYPE Squared_VT_Distance(TYPE xi[], TYPE xa[], TYPE xb[], TYPE xc[], TYPE &ba, TYPE &bb, TYPE &bc, TYPE *N=0)
{
	TYPE  xba[3],  xca[3],  xia[3];
	for(int n=0; n<3; n++)
	{
		xba[n]=xb[n]-xa[n];
		xca[n]=xc[n]-xa[n];
		xia[n]=xi[n]-xa[n];
	}

	TYPE _N[3];
	if(N==0)	N=_N;
	CROSS(xba, xca, N);
	TYPE nn=DOT(N, N);

	TYPE temp[3];
	CROSS(xia, xca, temp);
	TYPE weight_iaca=DOT(N, temp);
	CROSS(xba, xia, temp);
	TYPE weight_baia=DOT(N, temp);

	if(nn>1e-24f && weight_iaca>=0 && weight_baia>=0 && nn-weight_iaca-weight_baia>=0)
	{
		bb=weight_iaca/nn;
		bc=weight_baia/nn;
		ba=1-bb-bc;
	}
	else
	{
		TYPE min_distance=MY_INFINITE;
		TYPE r, distance;
		if(nn-weight_iaca-weight_baia<0 && ((distance=Squared_VE_Distance(xi, xb, xc, r))<min_distance))
		{
			min_distance=distance;
			bb=1-r;
			bc=r;
			ba=0;
		}
		if(weight_iaca<0 && ((distance=Squared_VE_Distance(xi, xa, xc, r))<min_distance))		
		{
			min_distance=distance;
			bb=0;
			bc=r;
			ba=1-bb-bc;
		}			
		if(weight_baia<0 && ((distance=Squared_VE_Distance(xi, xa, xb, r))<min_distance))
		{
			min_distance=distance;
			bb=r;
			bc=0;
			ba=1-bb-bc;
		}
	}

	N[0]=xi[0]-xa[0]*ba-xb[0]*bb-xc[0]*bc;
	N[1]=xi[1]-xa[1]*ba-xb[1]*bb-xc[1]*bc;
	N[2]=xi[2]-xa[2]*ba-xb[2]*bb-xc[2]*bc;
	return DOT(N, N);
}

template <class TYPE>
TYPE Simple_Squared_VT_Distance(TYPE xi[], TYPE xa[], TYPE xb[], TYPE xc[], TYPE &ba, TYPE &bb, TYPE &bc, TYPE *N=0)
{
	TYPE  xba[3],  xca[3],  xia[3];
	for(int n=0; n<3; n++)
	{
		xba[n]=xb[n]-xa[n];
		xca[n]=xc[n]-xa[n];
		xia[n]=xi[n]-xa[n];
	}

	TYPE _N[3];
	if(N==0)	N=_N;
	CROSS(xba, xca, N);
	TYPE nn=DOT(N, N);

	TYPE temp[3];
	CROSS(xia, xca, temp);
	TYPE weight_iaca=DOT(N, temp);
	CROSS(xba, xia, temp);
	TYPE weight_baia=DOT(N, temp);

	if(nn>1e-24f && weight_iaca>=0 && weight_baia>=0 && nn-weight_iaca-weight_baia>=0)
	{
		bb=weight_iaca/nn;
		bc=weight_baia/nn;
		ba=1-bb-bc;
	}
	else	return 999999;	

	N[0]=xi[0]-xa[0]*ba-xb[0]*bb-xc[0]*bc;
	N[1]=xi[1]-xa[1]*ba-xb[1]*bb-xc[1]*bc;
	N[2]=xi[2]-xa[2]*ba-xb[2]*bb-xc[2]*bc;
	return DOT(N, N);
}

#endif

