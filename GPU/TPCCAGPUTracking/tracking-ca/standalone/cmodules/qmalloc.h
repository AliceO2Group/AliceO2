#include <stdlib.h>

#ifndef QMALLOC_H
#define QMALLOC_H

#if !defined(_WIN32) & !defined(__cdecl)
#define __cdecl
#endif

class qmalloc
{
public:
	static void* __cdecl qMalloc(size_t size, bool huge, bool executable, bool locked, void* alloc_addr = NULL, int interleave = false);
	static int __cdecl qFree(void* ptr);

private:	
	static int qMallocCount;
	static int qMallocUsed;
	struct qMallocData
	{
		void* addr;
		size_t size;
	};
	static qMallocData* qMallocs;
};

#endif
