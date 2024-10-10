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

/// \file GPUTPCGMO2Output.h
/// \author David Rohr

#ifndef GPUTPCGMO2OUTPUT_H
#define GPUTPCGMO2OUTPUT_H

#include "GPUTPCDef.h"
#include "GPUTPCGMMergerGPU.h"

namespace o2
{
namespace gpu
{

class GPUTPCGMO2Output : public GPUTPCGMMergerGeneral
{
 public:
  enum K { prepare = 0,
           sort = 1,
           output = 2,
           mc = 3 };
  template <int32_t iKernel = defaultKernel>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

} // namespace gpu
} // namespace o2

#endif
