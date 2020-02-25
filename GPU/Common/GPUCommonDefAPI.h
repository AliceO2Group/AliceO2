// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonDefAPI.h
/// \author David Rohr

#ifndef GPUCOMMONDEFAPI_H
#define GPUCOMMONDEFAPI_H
// clang-format off

#ifndef GPUCOMMONDEF_H
  #error Please include GPUCommonDef.h!
#endif

//Define macros for GPU keywords. i-version defines inline functions.
//All host-functions in GPU code are automatically inlined, to avoid duplicate symbols.
//For non-inline host only functions, use no keyword at all!
#if !defined(GPUCA_GPUCODE) || defined(__OPENCL_HOST__) // For host / ROOT dictionary
  #define GPUd()                                    // device function
  #define GPUdDefault()                             // default (constructor / operator) device function
  #define GPUdi() inline                            // to-be-inlined device function
  #define GPUdii()                                  // Only on GPU to-be-inlined device function
  #define GPUh()                                    // Host-only function
  #define GPUhi() inline                            // to-be-inlined host-only function
  #define GPUhd()                                   // Host and device function, inlined during GPU compilation to avoid symbol clashes in host code
  #define GPUhdi() inline                           // Host and device function, to-be-inlined on host and device
  #define GPUhdni()                                 // Host and device function, not to-be-inlined automatically
  #define GPUg() INVALID_TRIGGER_ERROR_NO_HOST_CODE // GPU kernel
  #define GPUshared()                               // shared memory variable declaration
  #define GPUglobal()                               // global memory variable declaration (only used for kernel input pointers)
  #define GPUconstant()                             // constant memory variable declaraion
  #define GPUconstexpr() static constexpr           // constexpr on GPU that needs to be instantiated for dynamic access (e.g. arrays), becomes __constant on GPU
  #define GPUprivate()                              // private memory variable declaration
  #define GPUgeneric()                              // reference / ptr to generic address space
  #define GPUbarrier()                              // synchronize all GPU threads in block
  #define GPUAtomic(type) type                      // atomic variable type
  #define GPUsharedref()                            // reference / ptr to shared memory
  #define GPUglobalref()                            // reference / ptr to global memory
  #define GPUconstantref()                          // reference / ptr to constant memory
  #define GPUconstexprref()                         // reference / ptr to variable declared as GPUconstexpr()

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
  struct dim3 { unsigned int x, y, z; };
#elif defined(__OPENCL__) // Defines for OpenCL
  #define GPUd()
  #define GPUdDefault()
  #define GPUdi() inline
  #define GPUdii() inline
  #define GPUh() INVALID_TRIGGER_ERROR_NO_HOST_CODE
  #define GPUhi() INVALID_TRIGGER_ERROR_NO_HOST_CODE
  #define GPUhd() inline
  #define GPUhdi() inline
  #define GPUhdni()
  #define GPUg() __kernel
  #define GPUshared() __local
  #define GPUglobal() __global
  #define GPUconstant() __constant // TODO: possibly add const __restrict where possible later!
  #define GPUconstexpr() __constant
  #define GPUprivate() __private
  #define GPUgeneric() __generic
  #define GPUconstexprref() GPUconstexpr()
  #if defined(__OPENCLCPP__) && !defined(__clang__)
    #define GPUbarrier() work_group_barrier(mem_fence::global | mem_fence::local);
    #define GPUAtomic(type) atomic<type>
    static_assert(sizeof(atomic<unsigned int>) == sizeof(unsigned int), "Invalid size of atomic type");
  #else
    #define GPUbarrier() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
    #if defined(__OPENCLCPP__) && defined(GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS)
      namespace GPUCA_NAMESPACE { namespace gpu {
      template <class T> struct oclAtomic;
      template <> struct oclAtomic<unsigned int> {typedef atomic_uint t;};
      static_assert(sizeof(oclAtomic<unsigned int>::t) == sizeof(unsigned int), "Invalid size of atomic type");
      }}
      #define GPUAtomic(type) GPUCA_NAMESPACE::gpu::oclAtomic<type>::t
    #else
      #define GPUAtomic(type) volatile type
    #endif
    #ifdef CONSTEXPR
      #undef CONSTEXPR
    #endif
    #if defined(__OPENCLCPP__)
      #define CONSTEXPR constexpr
    #else
      #define CONSTEXPR const
    #endif
  #endif
  #if !defined(__OPENCLCPP__) // Other special defines for OpenCL 1
    #define GPUCA_USE_TEMPLATE_ADDRESS_SPACES // TODO: check if we can make this (partially, where it is already implemented) compatible with OpenCL CPP
    #define GPUsharedref() GPUshared()
    #define GPUglobalref() GPUglobal()
  #endif
  #if (!defined(__OPENCLCPP__) || !defined(GPUCA_OPENCLCPP_NO_CONSTANT_MEMORY))
    #define GPUconstantref() GPUconstant()
  #endif
#elif defined(__HIPCC__) //Defines for HIP
  #define GPUd() __device__
  #define GPUdDefault() __device__
  #define GPUdi() __device__ inline
  #define GPUdii() __device__ __forceinline__
  #define GPUh() __host__ inline
  #define GPUhi() __host__ inline
  #define GPUhd() __host__ __device__ inline
  #define GPUhdi() __host__ __device__ inline
  #define GPUhdni() __host__ __device__
  #define GPUg() __global__
  #define GPUshared() __shared__
  #if defined(GPUCA_GPUCODE_DEVICE) && 0 // TODO: Fix for HIP
    #define GPUCA_USE_TEMPLATE_ADDRESS_SPACES
    #define GPUglobal() __attribute__((address_space(1)))
    #define GPUglobalref() GPUglobal()
    #define GPUconstantref() __attribute__((address_space(4)))
    #define GPUsharedref() __attribute__((address_space(3)))
  #else
    #define GPUglobal()
  #endif
  #define GPUconstant()
  #define GPUconstexpr() __constant__
  #define GPUprivate()
  #define GPUgeneric()
  #define GPUbarrier() __syncthreads()
  #define GPUAtomic(type) type
#elif defined(__CUDACC__) //Defines for CUDA
  #define GPUd() __device__
  #define GPUdDefault()
  #define GPUdi() __device__ inline
  #define GPUdii() __device__ inline
  #define GPUh() __host__ inline
  #define GPUhi() __host__ inline
  #define GPUhd() __host__ __device__ inline
  #define GPUhdi() __host__ __device__ inline
  #define GPUhdni() __host__ __device__
  #define GPUg() __global__
  #define GPUshared() __shared__
  #define GPUglobal()
  #define GPUconstant()
  #define GPUconstexpr() __constant__
  #define GPUprivate()
  #define GPUgeneric()
  #define GPUbarrier() __syncthreads()
  #define GPUAtomic(type) type
#endif

#if (defined(__CUDACC__) && defined(GPUCA_CUDA_NO_CONSTANT_MEMORY)) || (defined(__HIPCC__) && defined(GPUCA_HIP_NO_CONSTANT_MEMORY)) || (defined(__OPENCL__) && !defined(__OPENCLCPP__) && defined(GPUCA_OPENCL_NO_CONSTANT_MEMORY)) || (defined(__OPENCLCPP__) && defined(GPUCA_OPENCLCPP_NO_CONSTANT_MEMORY))
  #undef GPUconstant
  #define GPUconstant() GPUglobal()
#endif

#ifndef GPUsharedref
#define GPUsharedref()
#endif
#ifndef GPUglobalref
#define GPUglobalref()
#endif
#ifndef GPUconstantref
#define GPUconstantref()
#endif
#ifndef GPUconstexprref
#define GPUconstexprref()
#endif

// Macros for GRID dimension
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
  #define get_global_size(dim) (blockDim.x * gridDim.x)
  #define get_num_groups(dim) (gridDim.x)
  #define get_local_id(dim) (threadIdx.x)
  #define get_local_size(dim) (blockDim.x)
  #define get_group_id(dim) (blockIdx.x)
#elif defined(__OPENCL__)
  // Using OpenCL defaults
#else
  #define get_global_id(dim) iBlock
  #define get_global_size(dim) nBlocks
  #define get_num_groups(dim) nBlocks
  #define get_local_id(dim) 0
  #define get_local_size(dim) 1
  #define get_group_id(dim) iBlock
#endif

  // clang-format on
#endif
