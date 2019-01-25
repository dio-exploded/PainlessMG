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
//  DYNAMIC_MESH
//  A virtual mesh class for dynamic simulation
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_DYNAMIC_MESH_H__
#define __WHMIN_DYNAMIC_MESH_H__
#include "MESH.h"


template <class TYPE>
class DYNAMIC_MESH: public MESH<TYPE>
{
public:
    using   BASE_MESH<TYPE>::max_number;
    using   BASE_MESH<TYPE>::number;
    using   BASE_MESH<TYPE>::X;

	TYPE*	V;
    TYPE*   old_X;

	DYNAMIC_MESH()
	{
		V       = new TYPE[max_number*3];
        old_X   = new TYPE[max_number*3];
		memset(V, 0, sizeof(TYPE)*max_number*3);
	}

	~DYNAMIC_MESH()
	{
		if(V)       delete[] V;
        if(old_X)   delete[] old_X;
	}

	void Clear_Velocity()
    {
        memset(V, 0, sizeof(TYPE)*number*3);
    }

///////////////////////////////////////////////////////////////////////////////////////////
//  Virtual simulation functions
///////////////////////////////////////////////////////////////////////////////////////////
	virtual void Initialize(TYPE t)=0;
	// Quick update without constraints
	virtual void Update(TYPE t)=0;
    //Enforce constraints
    virtual void Apply_Constraints(TYPE t)=0;
		
    void Begin_Constraints()
	{
		//Backup vertex positiones into old_X
		memcpy(old_X, X, sizeof(TYPE)*number*3);
	}
	
	void End_Constraints(TYPE inv_t)
	{
		for(int v=0; v<number; v++)
		{
			V[v*3+0]+=(X[v*3+0]-old_X[v*3+0])*inv_t;
			V[v*3+1]+=(X[v*3+1]-old_X[v*3+1])*inv_t;
			V[v*3+2]+=(X[v*3+2]-old_X[v*3+2])*inv_t;
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  IO functions.
///////////////////////////////////////////////////////////////////////////////////////////	
	void Write(std::fstream &output)
	{
		Write_Binaries(output, X, number*3);
		Write_Binaries(output, V, number*3);
	}

	bool Write_File(const char *file_name)
	{
		std::fstream output; 
		output.open(file_name,std::ios::out|std::ios::binary);
		if(!output.is_open())	{printf("Error, file not open.\n"); return false;}
		Write(output);
		output.close();
		return true;
	}
	
	void Read(std::fstream &input)
	{
		Read_Binaries(input, X, number*3);
		Read_Binaries(input, V, number*3);
	}

	bool Read_File(const char *file_name)
	{
		std::fstream input; 
		input.open(file_name,std::ios::in|std::ios::binary);
		if(!input.is_open())	{printf("Error, file not open.\n");	return false;}
		Read(input);
		input.close();
		return true;
	}

};

#endif