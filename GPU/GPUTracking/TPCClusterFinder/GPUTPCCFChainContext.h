// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <vector>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct GPUTPCCFChainContext {
  struct FragmentData {
    unsigned int nDigits[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
    unsigned int nPages[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
    std::vector<unsigned short> pageDigits[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
    GPUTPCClusterFinder::MinMaxCN minMaxCN[GPUCA_NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  };

  std::vector<FragmentData> fragmentData;
  unsigned int nPagesTotal;
  unsigned int nPagesSectorMax;
  unsigned int nPagesFragmentMax;
  unsigned int nPagesSector[GPUCA_NSLICES];
  size_t nMaxDigitsFragment[GPUCA_NSLICES];
  unsigned int tpcMaxTimeBin;
  unsigned int nFragments;

  void prepare(bool tpcZS, const CfFragment& fragmentMax)
  {
    nPagesTotal = nPagesSectorMax = nPagesFragmentMax;
    for (unsigned int i = 0; i < GPUCA_NSLICES; i++) {
      nPagesSector[i] = 0;
      nMaxDigitsFragment[i] = 0;
    }

    if (tpcZS) {
      tpcMaxTimeBin = 0;
      CfFragment f = fragmentMax;
      while (!f.isEnd()) {
        f = f.next();
      }
      nFragments = f.index;
      if (fragmentData.size() < nFragments) {
        fragmentData.resize(nFragments);
      }

      for (unsigned int i = 0; i < nFragments; i++) {
        for (unsigned int j = 0; j < GPUCA_NSLICES; j++) {
          for (unsigned int k = 0; k < GPUTrackingInOutZS::NENDPOINTS; k++) {
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
