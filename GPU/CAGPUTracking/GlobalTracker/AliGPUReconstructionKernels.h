#ifndef ALIGPURECONSTRUCTIONKERNELS_H
#define ALIGPURECONSTRUCTIONKERNELS_H

#ifdef GPUCA_ALIGPURECONSTRUCTIONCPU_DECLONLY
template <> class AliGPUReconstructionKernels<AliGPUReconstructionCPUBackend> : public AliGPUReconstructionCPUBackend
#define GPUCA_KRNL(...) ;
#define GPUCA_KRNL_CLASS AliGPUReconstructionCPUBackend
#else
template <class T> class AliGPUReconstructionKernels : public T
#define GPUCA_EXPAND(...) __VA_ARGS__
#define GPUCA_KRNL(X) GPUCA_EXPAND X
#define GPUCA_KRNL_CLASS T
#endif
{
public:
	virtual ~AliGPUReconstructionKernels() = default;
	AliGPUReconstructionKernels(const AliGPUCASettingsProcessing& cfg) : GPUCA_KRNL_CLASS(cfg) {}

protected:
	virtual int runKernelImpl(classArgument<AliGPUTPCNeighboursFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCNeighboursFinder>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUTPCNeighboursCleaner>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCNeighboursCleaner>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUTPCStartHitsFinder>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCStartHitsFinder>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUTPCStartHitsSorter>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCStartHitsSorter>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUTPCTrackletConstructor>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCTrackletConstructor>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUTPCTrackletConstructor, 1>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCTrackletConstructor, 1>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUTPCTrackletSelector>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCTrackletSelector>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUMemClean16>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, void* ptr, unsigned long size) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUMemClean16>(x, y, z, ptr, size);}))
	virtual int runKernelImpl(classArgument<AliGPUTPCGMMergerTrackFit>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTPCGMMergerTrackFit>(x, y, z);}))
	virtual int runKernelImpl(classArgument<AliGPUTRDTrackerGPU>, const krnlExec& x, const krnlRunRange& y, const krnlEvent& z) GPUCA_KRNL(({return T::template runKernelBackend<AliGPUTRDTrackerGPU>(x, y, z);}))
};

#undef GPUCA_KRNL
#undef GPUCA_KRNL_CLASS

#endif
