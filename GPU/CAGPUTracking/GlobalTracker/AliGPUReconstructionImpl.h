#ifndef ALIGPURECONSTRUCTIONIMPL_H
#define ALIGPURECONSTRUCTIONIMPL_H

#include "AliGPUReconstruction.h"
#include "AliGPUCADataTypes.h"
#include <stdexcept>

class AliGPUTPCNeighboursFinder;
class AliGPUTPCNeighboursCleaner;
class AliGPUTPCStartHitsFinder;
class AliGPUTPCStartHitsSorter;
class AliGPUTPCTrackletConstructor;
class AliGPUTPCTrackletSelector;

namespace
{
	template <class T> class classArgument {};
	
	enum class krnlDeviceType : int {CPU = 0, Device = 1, Auto = -1};
	struct krnlExec
	{
		krnlExec(int b, int t, int s, krnlDeviceType d = krnlDeviceType::Auto) : nBlocks(b), nThreads(t), stream(s), device(d) {}
		int nBlocks;
		int nThreads;
		int stream;
		krnlDeviceType device;
	};
	struct krnlRunRange
	{
		krnlRunRange() : start(0), end(0) {}
		krnlRunRange(unsigned int a) : start(a), end(a) {}
		krnlRunRange(unsigned int s, unsigned int e) : start(s), end(e) {}
		
		unsigned int start;
		unsigned int end;
	};
}

template <class T> class AliGPUReconstructionImpl : public T
{
public:
	virtual ~AliGPUReconstructionImpl() = default;
	AliGPUReconstructionImpl(const AliGPUCASettingsProcessing& cfg) : T(cfg) {}
	AliGPUReconstructionImpl() = default;

	template <class S, typename... Args> int runKernel(const krnlExec& x, const krnlRunRange& y, const Args&... args)
	{
		return runKernelImpl(classArgument<S>(), x, y, args...);
	}

protected:
	virtual int runKernelImpl(classArgument<AliGPUTPCNeighboursFinder>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCNeighboursFinder>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCNeighboursCleaner>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCNeighboursCleaner>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCStartHitsFinder>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCStartHitsFinder>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCStartHitsSorter>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCStartHitsSorter>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCTrackletConstructor>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCTrackletConstructor>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCTrackletSelector>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCTrackletSelector>(x, y);}
};

class AliGPUReconstructionCPUBackend
{
public:
	virtual ~AliGPUReconstructionCPUBackend() = default;
	
protected:
	template <class T, typename... Args> int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const Args&... args)
	{
		if (x.device == krnlDeviceType::Device) throw std::runtime_error("Cannot run device kernel on host");
		printf("Running kernel Host\n");
		return 0;
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
