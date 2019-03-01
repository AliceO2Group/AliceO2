#ifndef GPURECONSTRUCTIONKERNELS_H
#define GPURECONSTRUCTIONKERNELS_H

#ifdef GPUCA_GPURECONSTRUCTIONCPU_DECLONLY
template <> class GPUReconstructionKernels<GPUReconstructionCPUBackend> : public GPUReconstructionCPUBackend
#define GPUCA_KRNL(...) ;
#define GPUCA_KRNL_CLASS GPUReconstructionCPUBackend
#else
template <class T> class GPUReconstructionKernels : public T
#define GPUCA_EXPAND(...) __VA_ARGS__
#define GPUCA_KRNL(X) GPUCA_EXPAND X
#define GPUCA_KRNL_CLASS T
#endif
{
public:
	virtual ~GPUReconstructionKernels() = default;
	GPUReconstructionKernels(const GPUSettingsProcessing& cfg) : GPUCA_KRNL_CLASS(cfg) {}

protected:
	virtual int runKernelImpl(classArgument<GPUTPCNeighboursFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCNeighboursFinder>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUTPCNeighboursCleaner>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCNeighboursCleaner>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUTPCStartHitsFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCStartHitsFinder>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUTPCStartHitsSorter>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCStartHitsSorter>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUTPCTrackletConstructor>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCTrackletConstructor>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUTPCTrackletConstructor, 1>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCTrackletConstructor, 1>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUTPCTrackletSelector>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCTrackletSelector>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUMemClean16>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, void* ptr, unsigned long size) GPUCA_KRNL(({return T::template runKernelBackend<GPUMemClean16>(x, y, z, ptr, size);}))
	virtual int runKernelImpl(classArgument<GPUTPCGMMergerTrackFit>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTPCGMMergerTrackFit>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUTRDTrackerGPU>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUTRDTrackerGPU>(x, y, z);}))
	virtual int runKernelImpl(classArgument<GPUITSFitterKernel>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<GPUITSFitterKernel>(x, y, z);}))
};

#undef GPUCA_KRNL
#undef GPUCA_KRNL_CLASS

#endif
