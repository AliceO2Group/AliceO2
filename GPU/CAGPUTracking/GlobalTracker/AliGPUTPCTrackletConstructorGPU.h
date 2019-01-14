#include "AliGPUTPCGPUConfig.h"

GPUdi() int AliGPUTPCTrackletConstructor::FetchTracklet(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker, GPUsharedref() MEM_LOCAL(AliGPUTPCSharedMemory) &sMem)
{
	const int nativeslice = get_group_id(0) % 36;
	const int nTracklets = *tracker.NTracklets();
	GPUsync();
	if (get_local_id(0) == 0)
	{
		if (sMem.fNextTrackletFirstRun == 1)
		{
			sMem.fNextTrackletFirst = (get_group_id(0) - nativeslice) / 36 * GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR;
			sMem.fNextTrackletFirstRun = 0;
		}
		else
		{
			if (tracker.GPUParameters()->fNextTracklet < nTracklets)
			{
				const int firstTracklet = CAMath::AtomicAdd(&tracker.GPUParameters()->fNextTracklet, GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR);
				if (firstTracklet < nTracklets) sMem.fNextTrackletFirst = firstTracklet;
				else sMem.fNextTrackletFirst = -2;
			}
			else
			{
				sMem.fNextTrackletFirst = -2;
			}
		}
	}
	GPUsync();
	return (sMem.fNextTrackletFirst);
}

GPUdi() void AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGPU(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) *pTracker, GPUsharedref() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory)& sMem)
{
	const int nSlices = 36;
	int mySlice = get_group_id(0) % nSlices;
	int currentSlice = -1;

	if (get_local_id(0) == 0)
	{
		sMem.fNextTrackletFirstRun = 1;
	}

	for (int iSlice = 0;iSlice < nSlices;iSlice++)
	{
		GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker = pTracker[mySlice];

		AliGPUTPCThreadMemory rMem;

		while ((rMem.fItr = FetchTracklet(tracker, sMem)) != -2)
		{
			if (rMem.fItr >= 0 && get_local_id(0) < GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR)
			{
				rMem.fItr += get_local_id(0);
			}
			else
			{
				rMem.fItr = -1;
			}

			if (mySlice != currentSlice)
			{
				if (get_local_id(0) == 0)
				{
					sMem.fNTracklets = *tracker.NTracklets();
				}

				for (int i = get_local_id(0);i < GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(AliGPUTPCRow)) / sizeof(int);i += get_local_size(0))
				{
					reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
				}
				GPUsync();
				currentSlice = mySlice;
			}

			if (rMem.fItr >= 0 && rMem.fItr < sMem.fNTracklets)
			{
				rMem.fGo = true;
				DoTracklet(tracker, sMem, rMem);
			}
		}
		if (++mySlice >= nSlices) mySlice = 0;
	}
}

GPUdi() void AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorSingleSlice(GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) *pTracker, GPUsharedref() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory)& sMem)
{
	GPUconstant() MEM_CONSTANT(AliGPUTPCTracker) &tracker = *pTracker;

	if (get_local_id(0) == 0) sMem.fNTracklets = *tracker.NTracklets();
	for (int i = get_local_id(0);i < GPUCA_ROW_COUNT * sizeof(MEM_PLAIN(AliGPUTPCRow)) / sizeof(int);i += get_local_size(0))
	{
		reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
	}
	GPUsync();

	AliGPUTPCThreadMemory rMem;
	for (rMem.fItr = get_global_id(0);rMem.fItr < sMem.fNTracklets;rMem.fItr += get_global_size(0))
	{
		rMem.fGo = 1;
		DoTracklet(tracker, sMem, rMem);
	}
}

#ifndef __OPENCL__
GPUg() void AliGPUTPCTrackletConstructorGPU()
{
	//GPU Wrapper for AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGPU
	AliGPUTPCTracker *pTracker = gGPUConstantMem.tpcTrackers;
	GPUshared() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory) sMem;
	AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGPU(pTracker, sMem);
}

GPUg() void AliGPUTPCTrackletConstructorSingleSlice(int iSlice)
{
	GPUshared() AliGPUTPCTrackletConstructor::MEM_LOCAL(AliGPUTPCSharedMemory) sMem;
	AliGPUTPCTracker *pTracker = &gGPUConstantMem.tpcTrackers[iSlice];
	AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorSingleSlice(pTracker, sMem);
}
#endif
