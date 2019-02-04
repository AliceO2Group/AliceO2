#ifndef ALIGPURECONSTRUCTIONHIPWORKAROUNDS_H
#define ALIGPURECONSTRUCTIONHIPWORKAROUNDS_H

//Needs one workaround per kernel parameters, currently there are the folloing possible paramters:
// (void) //No parameters
// (void* ptr, unsigned long size)

//Kernel templates for no arguments
template <class TProcess, int I, typename... Args> GPUg() void runKernelHIP(int iSlice)
{
	runKernelHIP_a<TProcess, I>(iSlice);
}

template <class TProcess, int I, typename... Args> GPUg() void runKernelHIPMulti(int firstSlice, int nSliceCount)
{
	runKernelHIPMulti_a<TProcess, I>(firstSlice, nSliceCount);
}

template <class T, int I> void runKernelBackend_a(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, hipStream_t stream)
{
	hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIP<T, I>), dim3(x.nBlocks), dim3(x.nThreads), 0, stream, y.start);
}

template <class T, int I> void runKernelBackendMulti_a(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, hipStream_t stream)
{
	hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIPMulti<T, I>), dim3(x.nBlocks), dim3(x.nThreads), 0, stream, y.start, y.num);
}

//Kernel templates for arguments void*, unsigned long
template <class TProcess, int I, typename... Args> GPUg() void runKernelHIP2(int iSlice, void* ptr, unsigned long size)
{
	runKernelHIP_a<TProcess, I>(iSlice, ptr, size);
}

template <class TProcess, int I, typename... Args> GPUg() void runKernelHIPMulti2(int firstSlice, int nSliceCount, void* ptr, unsigned long size)
{
	runKernelHIPMulti_a<TProcess, I>(firstSlice, nSliceCount, ptr, size);
}

template <class T, int I> void runKernelBackend_a(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, hipStream_t stream, void* ptr, unsigned long size)
{
	hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIP2<T, I>), dim3(x.nBlocks), dim3(x.nThreads), 0, stream, y.start, ptr, size);
}

template <class T, int I> void runKernelBackendMulti_a(const krnlExec& x, const krnlRunRange& y, const krnlEvent& z, hipStream_t stream, void* ptr, unsigned long size)
{
	hipLaunchKernelGGL(HIP_KERNEL_NAME(runKernelHIPMulti2<T, I>), dim3(x.nBlocks), dim3(x.nThreads), 0, stream, y.start, y.num, ptr, size);
}

#endif
