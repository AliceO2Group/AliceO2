#ifndef ALIGPURECONSTRUCTIONIMPL_H
#define ALIGPURECONSTRUCTIONIMPL_H

#include "AliGPUReconstruction.h"

//struct method1 {static const int x = 1;};
//struct method2 {static const int x = 2;};

namespace
{
	template <class T> class classArgument {};
}

template <class T> class AliGPUReconstructionImpl : public T
{
public:
	virtual ~AliGPUReconstructionImpl() = default;
	AliGPUReconstructionImpl(const AliGPUCASettingsProcessing& cfg) : T(cfg) {}
	AliGPUReconstructionImpl() = default;

	template <class S, int nBlocks, int nThreads, bool device = false, typename... Args> void runKernel(Args&... args) {runKernelImpl(classArgument<S>(), nBlocks, nThreads, device, args...);}

protected:
	//virtual void runKernelImpl(classArgument<method1>, int nBlocks, int nThreads, bool device) {T::template runKernelBackend<method1>(nBlocks, nThreads, device);}
	//virtual void runKernelImpl(classArgument<method2>, int nBlocks, int nThreads, bool device) {T::template runKernelBackend<method2>(nBlocks, nThreads, device);}
};

class AliGPUReconstructionCPUBackend
{
public:
	virtual ~AliGPUReconstructionCPUBackend() = default;
	
protected:
	template <class T, typename... Args> void runKernelBackend(int nBlocks, int nThreads, bool device, Args&... args)
	{
		printf("Running kernel Host\n");
	};
};

class AliGPUReconstructionCPU : public AliGPUReconstruction, public AliGPUReconstructionImpl<AliGPUReconstructionCPUBackend>
{
	friend class AliGPUReconstruction;
	
public:
	virtual ~AliGPUReconstructionCPU() = default;
	
	//runKernel<method1, 256, 256>();
	//runKernel<method2, 256, 256>();
	
	virtual int RunTPCTrackingSlices();
	virtual int RunTPCTrackingMerger();
	virtual int RunTRDTracking();
	
protected:
	AliGPUReconstructionCPU(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstruction(cfg) {}
};

#endif
