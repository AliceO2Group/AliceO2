#include "qmultialloc.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

qMultiAlloc::qMultiAlloc()
{
	p = NULL;
	np = npalloc = 0;
	maxalign = 1024;
	ptr = NULL;
}

qMultiAlloc::~qMultiAlloc()
{
	if (p) free(p);
	if (ptr) cudaFree(ptr);
	p = NULL;
	ptr = NULL;
}

void qMultiAlloc::AddAlloc(void** ptr, size_t size, size_t align)
{
	if (np == npalloc)
	{
		if (npalloc == 0)
		{
			npalloc = 8;
		}
		else
		{
			npalloc *= 2;
		}
		
		if (p)
		{
			p = (ptr_struct*) realloc(p, npalloc * sizeof(ptr_struct));
		}
		else
		{
			p = (ptr_struct*) malloc(npalloc * sizeof(ptr_struct));
		}
		if (p == NULL)
		{
			printf("Memory Allocation Error\n");
			exit(1);
		}
	}

	p[np].ptr = ptr;
	p[np].size = size;
	np++;
	if (align > maxalign) maxalign = align;
}

size_t qMultiAlloc::Allocate()
{
	size_t size = 0;
	for (int i = 0;i < np;i++)
	{
		size += p[i].size;
		if (size % maxalign) size += maxalign - size % maxalign;
	}
	size += maxalign;

	checkCudaErrors(cudaMalloc(&ptr, size));
	if (ptr == NULL)
	{
		np = 0;
		maxalign = 1024;
		return(0);
	}
	char* tmpp = (char*) ptr;
	for (int i = 0;i < np;i++)
	{
		if (((size_t) tmpp) % maxalign) tmpp += maxalign - ((size_t) tmpp) % maxalign;
		*p[i].ptr = tmpp;
		tmpp += p[i].size;
	}
	return(size);
}

void qMultiAlloc::Free()
{
	cudaFree(ptr);
	ptr = 0;
	np = 0;
	maxalign = 1024;
}
