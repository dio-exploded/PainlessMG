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
//  IO_FUNCTIONS
//  Only consider little endian here. Not for big endian architecture
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_IO_FUNCTIONS_H__
#define __WHMIN_IO_FUNCTIONS_H__
#include <fstream>


///////////////////////////////////////////////////////////////////////////////////////////
//  Reading Functions.
///////////////////////////////////////////////////////////////////////////////////////////
template<class T>
inline void Read_Binaries(std::istream &input, T *v, const int size)
{input.read((char*)v, sizeof(T)*size);}

template<class T>
inline void Read_Binary(std::istream &input, T &v)
{input.read((char*)&v, sizeof(T));}

template<class T1,class T2>
inline void Read_Binary(std::istream &input, T1 &v1,T2 &v2)
{Read_Binary(input,v1);Read_Binary(input,v2);}

template<class T1,class T2,class T3>
inline void Read_Binary(std::istream &input, T1 &v1,T2 &v2,T3 &v3)
{Read_Binary(input,v1);Read_Binary(input,v2);Read_Binary(input,v3);}

template<class T1,class T2,class T3,class T4>
inline void Read_Binary(std::istream &input, T1 &v1,T2 &v2,T3 &v3, T4 &v4)
{Read_Binary(input,v1);Read_Binary(input,v2);Read_Binary(input,v3);Read_Binary(input,v4);}

///////////////////////////////////////////////////////////////////////////////////////////
//  Writing Functions.
///////////////////////////////////////////////////////////////////////////////////////////
template<class T>
inline void Write_Binaries(std::ostream &output,const T *v, const int size)
{output.write((const char*)v, sizeof(T)*size);}

template<class T>
inline void Write_Binary(std::ostream &output,const T& v)
{output.write((const char*)&v, sizeof(T));}

template<class T1,class T2>
inline void Write_Binary(std::ostream &output,const T1& v1,const T2& v2)
{Write_Binary(output,v1);Write_Binary(output,v2);}

template<class T1,class T2,class T3>
inline void Write_Binary(std::ostream &output,const T1& v1,const T2& v2,const T3& v3)
{Write_Binary(output,v1);Write_Binary(output,v2);Write_Binary(output,v3);}

template<class T1,class T2,class T3,class T4>
inline void Write_Binary(std::ostream &output,const T1& v1,const T2& v2,const T3& v3,const T4& v4)
{Write_Binary(output,v1);Write_Binary(output,v2);Write_Binary(output,v3);Write_Binary(output,v4);}

///////////////////////////////////////////////////////////////////////////////////////////
//  Enforced double/float Type Reading Functions.
///////////////////////////////////////////////////////////////////////////////////////////
template<class T>
inline void Read_Binary_Double(std::istream &input,T &v)
{double temp; Read_Binary(input,temp); v=(T)temp;}

template<class T>
inline void Read_Binary_Double(std::istream &input,T &v1,T &v2)
{Read_Binary_Double(input,v1);Read_Binary_Double(input,v2);}

template<class T>
inline void Read_Binary_Double(std::istream &input,T &v1,T &v2,T &v3)
{Read_Binary_Double(input,v1);Read_Binary_Double(input,v2);Read_Binary_Double(input,v3);}

template<class T>
inline void Read_Binary_Double(std::istream &input,T &v1,T &v2,T &v3, T &v4)
{Read_Binary_Double(input,v1);Read_Binary_Double(input,v2);Read_Binary_Double(input,v3);Read_Binary_Double(input,v4);}

template<class T>
inline void Read_Binary_Float(std::istream &input,T &v)
{float temp; Read_Binary(input, temp); v=(T)temp;}

template<class T>
inline void Read_Binary_Float(std::istream &input,T &v1,T& v2)
{Read_Binary_Float(input,v1);Read_Binary_Float(input,v2);}

template<class T>
inline void Read_Binary_Float(std::istream &input,T &v1,T &v2,T &v3)
{Read_Binary_Float(input,v1);Read_Binary_Float(input,v2);Read_Binary_Float(input,v3);}

template<class T>
inline void Write_Binary_Double(std::ostream &output,const T &v)
{double temp=(double)v; Write_Binary(output,temp);}

template<class T>
inline void Write_Binary_Double(std::ostream &output,const T &v1,const T &v2)
{Write_Binary_Double(output,v1);Write_Binary_Double(output,v2);}

template<class T>
inline void Write_Binary_Double(std::ostream &output,const T &v1,const T &v2,const T &v3)
{Write_Binary_Double(output,v1);Write_Binary_Double(output,v2);Write_Binary_Double(output,v3);}

template<class T>
inline void Write_Binary_Float(std::ostream &output,const T &v)
{float temp=(float)v; Write_Binary(output, temp);}

template<class T>
inline void Write_Binary_Float(std::ostream &output,const T &v1,const T &v2)
{Write_Binary_Float(output,v1);Write_Binary_Float(output,v2);}

template<class T>
inline void Write_Binary_Float(std::ostream &output,const T &v1,const T &v2,const T &v3)
{Write_Binary_Float(output,v1);Write_Binary_Float(output,v2);Write_Binary_Float(output,v3);}

#endif
