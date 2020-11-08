// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDDef.h
/// \author David Rohr

#ifndef O2_GPU_GPURECONSTRUCTIONCUDADEF_H
#define O2_GPU_GPURECONSTRUCTIONCUDADEF_H

#define GPUCA_UNROLL(CUDA, HIP) GPUCA_M_UNROLL_##CUDA
#define GPUdic(CUDA, HIP) GPUCA_GPUdic_select_##CUDA()

#include "GPUDef.h"

#ifndef GPUCA_NO_CONSTANT_MEMORY
#define GPUCA_CONSMEM_PTR
#define GPUCA_CONSMEM_CALL
#define GPUCA_CONSMEM (gGPUConstantMemBuffer.v)
#else
#define GPUCA_CONSMEM_PTR const GPUConstantMem *gGPUConstantMemBuffer,
#define GPUCA_CONSMEM_CALL me->mDeviceConstantMem,
#define GPUCA_CONSMEM ((GPUConstantMem&)(*gGPUConstantMemBuffer))
#endif
#define GPUCA_KRNL_BACKEND_CLASS GPUReconstructionCUDABackend

#endif
