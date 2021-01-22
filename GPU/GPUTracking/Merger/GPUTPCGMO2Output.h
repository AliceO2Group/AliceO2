// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  template <int iKernel = defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& merger);
};

} // namespace gpu
} // namespace o2

#endif
