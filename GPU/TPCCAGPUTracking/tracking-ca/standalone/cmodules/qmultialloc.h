#ifndef QMULTIALLOC_H
#define QMULTIALLOC_H
#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else
#include <stddef.h>
#endif

class qMultiAlloc
{
public:
	qMultiAlloc();
	~qMultiAlloc();

	void AddAlloc(void** ptr, size_t size, size_t align = 1024);
	size_t Allocate();
	void Free();

private:
	struct ptr_struct
	{
		void** ptr;
		size_t size;
	};

	ptr_struct* p;
	int np;
	int npalloc;
	size_t maxalign;
	void* ptr;
};

#endif
