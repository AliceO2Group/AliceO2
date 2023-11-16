// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCDecompression.h
/// \author Gabriele Cimador

#ifndef GPUTPCDECOMPRESSION_H
#define GPUTPCDECOMPRESSION_H

#include "GPUDef.h"
#include "GPUProcessor.h"
#include "GPUCommonMath.h"
#include "GPUParam.h"

namespace GPUCA_NAMESPACE::gpu
{

class GPUTPCDecompression : public GPUProcessor
{
  friend class GPUTPCDecmpressionKernels;
  friend class GPUChainTracking;

 public:
  unsigned int test = 42;
  unsigned int* testP = nullptr;

  void* SetPointersMemory(void* mem);

#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);
#endif
};
}
#endif // GPUTPCDECOMPRESSION_H
