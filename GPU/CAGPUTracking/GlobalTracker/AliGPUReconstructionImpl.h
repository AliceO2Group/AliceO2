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

#include "AliGPUTPCNeighboursFinder.h"
#include "AliGPUTPCNeighboursCleaner.h"
#include "AliGPUTPCStartHitsFinder.h"
#include "AliGPUTPCStartHitsSorter.h"
#include "AliGPUTPCTrackletConstructor.h"
#include "AliGPUTPCTrackLinearisation.h"
#include "AliGPUTPCTrackletSelector.h"

namespace AliGPUReconstruction_krnlHelpers {
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
} //End Namespace

using namespace AliGPUReconstruction_krnlHelpers;

template <class T> class AliGPUReconstructionImpl : public T
{
public:
	virtual ~AliGPUReconstructionImpl() = default;
	AliGPUReconstructionImpl(const AliGPUCASettingsProcessing& cfg) : T(cfg) {}
	AliGPUReconstructionImpl() = default;

protected:
	virtual int runKernelImpl(classArgument<AliGPUTPCNeighboursFinder>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCNeighboursFinder>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCNeighboursCleaner>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCNeighboursCleaner>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCStartHitsFinder>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCStartHitsFinder>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCStartHitsSorter>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCStartHitsSorter>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCTrackletConstructor>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCTrackletConstructor>(x, y);}
	virtual int runKernelImpl(classArgument<AliGPUTPCTrackletSelector>, const krnlExec& x, const krnlRunRange& y) {return T::template runKernelBackend<AliGPUTPCTrackletSelector>(x, y);}
};

class AliGPUReconstructionCPUBackend : public AliGPUReconstruction
{
public:
	virtual ~AliGPUReconstructionCPUBackend() = default;
	
protected:
	AliGPUReconstructionCPUBackend(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstruction(cfg) {}
	template <class T, typename... Args> inline int runKernelBackend(const krnlExec& x, const krnlRunRange& y, const Args&... args)
	{
		if (x.device == krnlDeviceType::Device) throw std::runtime_error("Cannot run device kernel on host");
		for (unsigned int k = y.start;k < y.end;k++)
		{
			for (int iB = 0; iB < x.nBlocks; iB++)
			{
				typename T::AliGPUTPCSharedMemory smem;
				for (int iS = 0; iS <= T::NThreadSyncPoints(); iS++)
				{
					T::Thread(x.nBlocks, 1, iB, 0, iS, smem, mWorkers->tpcTrackers[k]);
				}
			}
		}
		return 0;
	}
};

class AliGPUReconstructionCPU : public AliGPUReconstructionImpl<AliGPUReconstructionCPUBackend>
{
	friend class AliGPUReconstruction;
	
public:
	virtual ~AliGPUReconstructionCPU() = default;
	
	template <class S, typename... Args> inline int runKernel(const krnlExec& x, const krnlRunRange& y, const Args&... args)
	{
		return runKernelImpl(classArgument<S>(), x, y, args...);
	}
	
	virtual int RunTPCTrackingSlices();
	virtual int RunTPCTrackingMerger();
	virtual int RunTRDTracking();
	
protected:
	AliGPUReconstructionCPU(const AliGPUCASettingsProcessing& cfg) : AliGPUReconstructionImpl(cfg) {}
};

#endif
