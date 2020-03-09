// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonDefSettings.h
/// \author David Rohr

// clang
#ifndef GPUCOMMONDEFSETTINGS_H
#define GPUCOMMONDEFSETTINGS_H

// clang-format off

#ifndef GPUCOMMONDEF_H
  #error Please include GPUCommonDef.h!
#endif

//#define GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS     // Use C11 atomic instead of old style atomics for OpenCL C++ in clang (OpenCL 2.2 C++ will use C++11 atomics irrespectively)

//#define GPUCA_CUDA_NO_CONSTANT_MEMORY          // Do not use constant memory for CUDA
//#define GPUCA_HIP_NO_CONSTANT_MEMORY           // Do not use constant memory for HIP - MANDATORY for now since all AMD GPUs have insufficient constant memory with HIP
//#define GPUCA_OPENCL_NO_CONSTANT_MEMORY        // Do not use constant memory for OpenCL 1.2
#define GPUCA_OPENCLCPP_NO_CONSTANT_MEMORY       // Do not use constant memory for OpenCL C++ - MANDATORY as OpenCL cannot cast between __constant and __generic yet!

// clang-format on

#endif // GPUCOMMONDEFSETTINGS_H
