#ifndef ALITPCCOMMONDEFGPU_H
#define ALITPCCOMMONDEFGPU_H

#if !(defined(__CINT__) || defined(__ROOTCINT__) || defined(__CLING__) || defined(__ROOTCLING__) || defined(G__ROOT)) //No GPU code for ROOT
	#if defined(__CUDACC__) || defined(__OPENCL__) || defined(__HIPCC__)
		#define GPUCA_GPUCODE //Compiled by GPU compiler
	#endif

	#if defined(__CUDA_ARCH__) || defined(__OPENCL__) || defined(__HIP_DEVICE_COMPILE__)
		#define GPUCA_GPUCODE_DEVICE //Executed on device
	#endif
#endif

//Define macros for GPU keywords. i-version defines inline functions.
//All host-functions in GPU code are automatically inlined, to avoid duplicate symbols.
//For non-inline host only functions, use no keyword at all!
#if !defined(GPUCA_GPUCODE) //For host / ROOT dictionary
	#define GPUd()
	#define GPUdi() inline
	#define GPUh()
	#define GPUhi() inline
	#define GPUhd()
	#define GPUhdi() inline
	#define GPUg() INVALID_TRIGGER_ERROR_NO_HOST_CODE
	#define GPUshared()
	#define GPUglobal()
	#define GPUconstant()
	#define GPUbarrier()
	#define GPUAtomic(type) type

	struct float4 { float x, y, z, w; };
	struct float3 { float x, y, z; };
	struct float2 { float x; float y; };
	struct uchar2 { unsigned char x, y; };
	struct short2 { short x, y; };
	struct ushort2 { unsigned short x, y; };
	struct int2 { int x, y; };
	struct int3 { int x, y, z; };
	struct int4 { int x, y, z, w; };
	struct uint1 { unsigned int x; };
	struct uint2 { unsigned int x, y; };
	struct uint3 { unsigned int x, y, z; };
	struct uint4 { unsigned int x, y, z, w; };
	struct uint16 { unsigned int x[16]; };
	struct dim3 { unsigned int x, y, z; };
#elif defined(__OPENCL__) //Defines for OpenCL
	#define GPUd()
	#define GPUdi() inline
	#define GPUh() INVALID_TRIGGER_ERROR_NO_HOST_CODE
	#define GPUhi() INVALID_TRIGGER_ERROR_NO_HOST_CODE
	#define GPUhd() inline
	#define GPUhdi() inline
	#define GPUg() __kernel
	#define GPUshared() __local
	#define GPUglobal() __global
	//#define GPUconstant() __constant //TODO: Replace __constant by __global (possibly add const __restrict where possible later!)
	#define GPUconstant() __global
	#ifdef __OPENCLCPP__
		#define GPUbarrier() work_group_barrier(mem_fence::global | mem_fence::local);
		#define GPUAtomic(type) atomic<type>
		
		static_assert(sizeof(atomic<int>) == sizeof(int), "Invalid atomic type");
	#else
		#define GPUbarrier() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
		#define GPUAtomic(type) type
	#endif
#elif defined(__CUDACC__) //Defines for CUDA
	#define GPUd() __device__
	#define GPUdi() __device__ inline
	#define GPUh() __host__ inline
	#define GPUhi() __host__ inline
	#define GPUhd() __host__ __device__ inline
	#define GPUhdi() __host__ __device__ inline
	#define GPUg() __global__
	#define GPUshared() __shared__
	#define GPUglobal()
	#define GPUconstant()
	#define GPUbarrier() __syncthreads()
	#define GPUAtomic(type) type
#elif defined(__HIPCC__) //Defines for HIP
	#define GPUd() __device__
	#define GPUdi() __device__ inline
	#define GPUh() __host__ inline
	#define GPUhi() __host__ inline
	#define GPUhd() __host__ __device__ inline
	#define GPUhdi() __host__ __device__ inline
	#define GPUg() __global__
	#define GPUshared() __shared__
	#define GPUglobal()
	#define GPUconstant()
	#define GPUbarrier() __syncthreads()
	#define GPUAtomic(type) type
#endif

#if defined(__OPENCL__) && !defined(__OPENCLCPP__) //Other special defines for OpenCL
	#define GPUsharedref() GPUshared()
	#define GPUglobalref() __global
#else //Other defines for the rest
	#define GPUsharedref()
	#define GPUglobalref()
#endif

#endif
