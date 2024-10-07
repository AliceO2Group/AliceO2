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

/// \file ChargeMapFiller.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CHARGE_MAP_FILLER_H
#define O2_GPU_CHARGE_MAP_FILLER_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"

namespace o2::tpc
{
class Digit;
} // namespace o2::tpc

namespace GPUCA_NAMESPACE::gpu
{

struct ChargePos;

class GPUTPCCFChargeMapFiller : public GPUKernelTemplate
{
 public:
  enum K : int32_t {
    fillIndexMap,
    fillFromDigits,
    findFragmentStart,
  };

#ifdef GPUCA_HAVE_O2HEADERS
  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }
#endif

  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep()
  {
    return GPUDataTypes::RecoStep::TPCClusterFinding;
  }

  template <int32_t iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  static GPUd() void fillIndexMapImpl(int32_t, int32_t, int32_t, int32_t, const CfFragment&, const tpc::Digit*, Array2D<uint>&, size_t);

  static GPUd() void fillFromDigitsImpl(int32_t, int32_t, int32_t, int32_t, processorType&, const CfFragment&, size_t, const tpc::Digit*, ChargePos*, Array2D<PackedCharge>&);

 private:
  static GPUd() size_t findTransition(int32_t, const tpc::Digit*, size_t, size_t);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
