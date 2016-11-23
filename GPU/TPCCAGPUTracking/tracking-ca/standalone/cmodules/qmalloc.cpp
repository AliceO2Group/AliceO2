#include "qmalloc.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else //_WIN32
#include <unistd.h>
#include <sys/mman.h>
#include <syscall.h>
#ifdef _NUMAIF_H
#include <numaif.h>
#endif

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000 /* arch specific */
#endif
#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT 0
#endif
#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED 1
#endif
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
#endif //!_WIN32

#ifndef STD_OUT
#define STD_OUT stdout
#endif

int qmalloc::qMallocCount = 0;
int qmalloc::qMallocUsed = 0;
qmalloc::qMallocData* qmalloc::qMallocs = NULL;

#ifdef _WIN32
static void Privilege(TCHAR* pszPrivilege, BOOL bEnable)
{
	HANDLE           hToken;
	TOKEN_PRIVILEGES tp;
	BOOL             status;
	DWORD            error;

	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
	{
		fprintf(STD_OUT, "Error obtaining process token\n");
	}

	if (!LookupPrivilegeValue(NULL, pszPrivilege, &tp.Privileges[0].Luid))
	{
		fprintf(STD_OUT, "Error looking up priviledge value\n");
	}

	tp.PrivilegeCount = 1;
	tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

	status = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);

	error = GetLastError();
	if (!status || (error != ERROR_SUCCESS))
	{
		fprintf(STD_OUT, "Error obtaining Priviledge %d\n", GetLastError());
	}

	CloseHandle(hToken);
}
#endif

void* qmalloc::qMalloc(size_t size, bool huge, bool executable, bool locked, void* alloc_addr, int interleave)
{
	int pagesize;
	void* addr;
	if (huge)
	{
#ifdef _WIN32
		static int tokenObtained = 0;
#ifdef _AMD64_
		pagesize = GetLargePageMinimum();
#else
		pagesize = 1024 * 2048;
#endif
		if (tokenObtained == 0)
		{
			fprintf(STD_OUT, "Obtaining security token\n");
			Privilege(TEXT("SeLockMemoryPrivilege"), TRUE);
			tokenObtained = 1;
		}
#else
		pagesize = 1024 * 2048;
#endif
	}
	else
	{
#ifdef _WIN32
		SYSTEM_INFO si;
		GetSystemInfo(&si);
		pagesize = si.dwPageSize;
#else
		pagesize = sysconf(_SC_PAGESIZE);
#endif
	}
	if (size % pagesize) size += pagesize - size % pagesize;
#ifdef _WIN32
	DWORD flags = MEM_COMMIT;
	if (huge)
	{
		flags |= MEM_LARGE_PAGES;
	}
	DWORD protect = PAGE_READWRITE;
	if (executable)
	{
		protect = PAGE_EXECUTE_READWRITE;
	}
	if (interleave)
	{
		fprintf(stderr, "Interleaved allocation not supported on Windows\n");
		return(NULL);
	}
	if (alloc_addr != NULL)
	{
		if (VirtualAlloc(alloc_addr, size, (flags & ~MEM_COMMIT) | MEM_RESERVE, protect) != alloc_addr)
		{
			return(NULL);
		}
	}
	addr = VirtualAlloc(alloc_addr, size, flags, protect);
#else
	int flags = MAP_ANONYMOUS | MAP_PRIVATE;
	int prot = PROT_READ | PROT_WRITE;
	if (huge) flags |= MAP_HUGETLB;
	if (executable) prot |= PROT_EXEC;
	if (locked) flags |= MAP_LOCKED;
	//unsigned long oldnodemask;
	//int oldpolicy;
	if (interleave && locked) //mmap will perform a memory lock, so we have to change memory policy beforehand
	{
/*		if (syscall(SYS_get_mempolicy, &oldpolicy, &oldnodemask, sizeof(oldnodemask) * 8, NULL, 0) != 0)
		{
		    fprintf(stderr, "Error obtaining memory policy\n");
		    exit(1);
		}*/
		unsigned long nodemask = 0xffffff;
		if (syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8) != 0)
		{
		    fprintf(stderr, "Error setting memory policy\n");
		}
	}
	addr = mmap(alloc_addr, size, prot, flags, 0, 0);
	if (addr == MAP_FAILED) addr = NULL;
	if (interleave)
	{
		if (locked)	//Restore old memory policy
		{
			//syscall(SYS_set_mempolicy, oldpolicy, &oldnodemask, sizeof(oldnodemask) * 8);
			if (syscall(SYS_set_mempolicy, MPOL_DEFAULT, NULL) != 0)
			{
			    fprintf(stderr, "Error setting memory policy\n");
			}
		}
		else if (addr) //Set memory policy for region
		{
#ifndef _NUMAIF_H
			fprintf(stderr, "Interleaved memory can only be used with non-locked memory if numaif.h is present\n");
			exit(1);
#else
			unsigned long nodemask = 0xffffff;
			mbind(addr, size, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8, 0);
#endif
		}
	}
#endif

	if (alloc_addr != NULL && addr != alloc_addr)
	{
		fprintf(stderr, "Could not allocate memory at desired address\n");
#ifdef _WIN32
		VirtualFree(addr, 0, MEM_RELEASE);
#else
		munmap(addr, size);
#endif
		return(NULL);
	}

	if (addr == NULL)
	{
#ifdef _WIN32
		DWORD error = GetLastError();
#endif
		fprintf(stderr, "Failed to allocate memory\n");
		return(NULL);
	}
	
	if (qMallocCount == qMallocUsed)
	{
		if (qMallocCount == 0) qMallocCount = 8;
		else if (qMallocCount < 1024) qMallocCount *= 2;
		else qMallocCount += 1024;
		if (qMallocUsed == 0)
		{
			qMallocs = (qMallocData*) malloc(qMallocCount * sizeof(qMallocData));
		}
		else
		{
			qMallocs = (qMallocData*) realloc(qMallocs, qMallocCount * sizeof(qMallocData));
		}
	}
	qMallocs[qMallocUsed].addr = addr;
	qMallocs[qMallocUsed].size = size;
	qMallocUsed++;

#ifdef _WIN32
	if (locked)
	{
		size_t minp, maxp;
		HANDLE pid = GetCurrentProcess();
		if (GetProcessWorkingSetSize(pid, (PSIZE_T) &minp, (PSIZE_T) &maxp) == 0) fprintf(STD_OUT, "Error getting minimum working set size\n");
		if (SetProcessWorkingSetSize(pid, minp + size, maxp + size) == 0) fprintf(STD_OUT, "Error settings maximum working set size\n");
		if (VirtualLock(addr, size) == 0)
		{
			fprintf(STD_OUT, "Error locking memory\n");
			DWORD error = GetLastError();
			VirtualFree(addr, 0, MEM_RELEASE);
			if (SetProcessWorkingSetSize(pid, minp, maxp) == 0) fprintf(STD_OUT, "Error settings maximum working set size\n");
			addr = NULL;
		}
	}
#endif

	return(addr);
}

int qmalloc::qFree(void* ptr)
{
	for (int i = 0;i < qMallocUsed;i++)
	{
		if (qMallocs[i].addr == ptr)
		{
#ifdef _WIN32
			if (VirtualFree(ptr, 0, MEM_RELEASE) == 0) return(1);
#else
			if (munmap(ptr, qMallocs[i].size)) return(1);
#endif
			qMallocUsed--;
			if (i < qMallocUsed) memcpy(&qMallocs[i], &qMallocs[qMallocUsed], sizeof(qMallocData));
			if (qMallocUsed == 0)
			{
				free(qMallocs);
				qMallocCount = 0;
			}
			return(0);
		}
	}
	return(1);
}