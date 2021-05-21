// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  enum K : int {
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

  template <int iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

  static GPUd() void fillIndexMapImpl(int, int, int, int, const CfFragment&, const tpc::Digit*, Array2D<uint>&, size_t);

  static GPUd() void fillFromDigitsImpl(int, int, int, int, processorType&, const CfFragment&, size_t, const tpc::Digit*, ChargePos*, Array2D<PackedCharge>&);

 private:
  static GPUd() size_t findTransition(int, const tpc::Digit*, size_t, size_t);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
