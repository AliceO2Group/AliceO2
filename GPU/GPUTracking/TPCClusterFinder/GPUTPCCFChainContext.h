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

/// \file GPUTPCCFChainContext.h
/// \author David Rohr

#ifndef O2_GPU_TPCCFCHAINCONTEXT_H
#define O2_GPU_TPCCFCHAINCONTEXT_H

#include "clusterFinderDefs.h"
#include "GPUDataTypes.h"
#include "GPUTPCClusterFinder.h"
#include "CfFragment.h"
#include <vector>
#include <utility>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct GPUTPCCFChainContext {
  struct FragmentData {
    uint32_t nDigits[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
    uint32_t nPages[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
    std::vector<uint16_t> pageDigits[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
    GPUTPCClusterFinder::MinMaxCN minMaxCN[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  };

  struct PtrSave {
    GPUTPCClusterFinder::ZSOffset* zsOffsetHost;
    GPUTPCClusterFinder::ZSOffset* zsOffsetDevice;
    uint8_t* zsDevice;
  };

  int32_t zsVersion;
  std::vector<FragmentData> fragmentData;
  uint32_t nPagesTotal;
  uint32_t nPagesFragmentMax;
  uint32_t nPagesSector[GPUCA_NSLICES];
  uint32_t nDigitsEndpointMax[GPUCA_NSLICES];
  uint32_t tpcMaxTimeBin;
  bool abandonTimeframe;
  uint32_t nFragments;
  CfFragment fragmentFirst;
  std::pair<uint32_t, uint32_t> nextPos[GPUCA_NSLICES];
  PtrSave ptrSave[GPUCA_NSLICES];
  const o2::tpc::ClusterNativeAccess* ptrClusterNativeSave;

  void prepare(bool tpcZS, const CfFragment& fragmentMax)
  {
    abandonTimeframe = false;
    nPagesTotal = nPagesFragmentMax = 0;
    for (uint32_t i = 0; i < GPUCA_NSLICES; i++) {
      nPagesSector[i] = 0;
      nDigitsEndpointMax[i] = 0;
    }

    if (tpcZS) {
      tpcMaxTimeBin = 0;
      nFragments = fragmentMax.count();
      if (fragmentData.size() < nFragments) {
        fragmentData.resize(nFragments);
      }

      for (uint32_t i = 0; i < nFragments; i++) {
        for (uint32_t j = 0; j < GPUCA_NSLICES; j++) {
          for (uint32_t k = 0; k < GPUTrackingInOutZS::NENDPOINTS; k++) {
            fragmentData[i].nDigits[j][k] = fragmentData[i].nPages[j][k] = 0;
            fragmentData[i].pageDigits[j][k].clear();
          }
        }
      }
    }
  }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
