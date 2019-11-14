#pragma once
#include<cstdio>
template<typename TYPE>
void outputBoo(const char *filename, int m, int n, int *rowInd, int *colInd, TYPE *val, int nnz, int blockdim, int base=0)
{
	FILE *f = fopen(filename, "w");
	fprintf(f, "%d %d %d\n", m*blockdim, n*blockdim, nnz*blockdim*blockdim);
	for (int i = 0; i < nnz; i++)
		for (int di = 0; di < blockdim; di++)
			for (int dj = 0; dj < blockdim; dj++)
				fprintf(f, "%d %d %f\n", rowInd[i] * blockdim + di+base, colInd[i] * blockdim + dj+base, val[blockdim*blockdim*i + di * blockdim + dj]);
	fclose(f);
}
template<typename TYPE>
void outputCsr(const char *filename, int m, int n, int *rowPtr, int *colInd, TYPE *val, int nnz, int base = 0)
{
	FILE *f = fopen(filename, "w");
	fprintf(f, "%d %d %d\n", m, n, nnz);
	for (int i = 0; i < m; i++)
		for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++)
			fprintf(f, "%d %d %f\n", i+base, colInd[j]+base, val[j]);
	fclose(f);
}
template<typename TYPE>
void outputVector(const char *filename, int m, TYPE *val, int base = 0)
{
	FILE *f = fopen(filename, "w");
	fprintf(f, "%d %d %d\n", m, 1, m);
	for (int i = 0; i < m; i++)
	{
		if (typeid(TYPE) == typeid(float))
			fprintf(f, "%d %d %.10f\n", i + base, base, val[i]);
		if (typeid(TYPE) == typeid(int))
			fprintf(f, "%d %d %d\n", i + base, base, val[i]);
	}
	fclose(f);
}
template<typename TYPE>
void outputDense(const char *filename, int m, int n, TYPE *val, int base = 0)
{
	FILE *f = fopen(filename, "w");
	fprintf(f, "%d %d %d\n", m, n, m*n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			fprintf(f, "%d %d %f\n", i + base, j + base, val[j*m + i]);
	fclose(f);
}
