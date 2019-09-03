// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterErrorStat.h
/// \author David Rohr

#ifndef GPUTPCCLUSTERERRORSTAT_H
#define GPUTPCCLUSTERERRORSTAT_H

//#define EXTRACT_RESIDUALS

#if (defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_BUILD_QA)) && !defined(GPUCA_GPUCODE) && defined(EXTRACT_RESIDUALS)
#include "cagpu/GPUTPCGPURootDump.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterErrorStat {
  GPUTPCClusterErrorStat(int maxN) : fTupBuf(maxN) {}

  static GPUTPCGPURootDump<TNtuple> fTup;
  static long long int fCount;

  std::vector<std::array<float, 10>> fTupBuf;

  void Fill(float x, float y, float z, float alpha, float trkX, float* fP, float* fC, int ihit, int iWay)
  {
    if (iWay == 1) {
      fTupBuf[ihit] = {fP[0], fP[1], fP[2], fP[3], fP[4], fC[0], fC[2], fC[5], fC[9], fC[14]};
    } else if (iWay == 2) {
      fTup.Fill(x, y, z, alpha, trkX, (fP[0] * fTupBuf[ihit][5] + fTupBuf[ihit][0] * fC[0]) / (fTupBuf[ihit][5] + fC[0]), (fP[1] * fTupBuf[ihit][6] + fTupBuf[ihit][1] * fC[2]) / (fTupBuf[ihit][6] + fC[2]), (fP[2] * fTupBuf[ihit][7] + fTupBuf[ihit][2] * fC[5]) / (fTupBuf[ihit][7] + fC[5]),
                (fP[3] * fTupBuf[ihit][8] + fTupBuf[ihit][3] * fC[9]) / (fTupBuf[ihit][8] + fC[9]), (fP[4] * fTupBuf[ihit][9] + fTupBuf[ihit][4] * fC[14]) / (fTupBuf[ihit][9] + fC[14]), fC[0] * fTupBuf[ihit][5] / (fC[0] + fTupBuf[ihit][5]),
                fC[2] * fTupBuf[ihit][6] / (fC[2] + fTupBuf[ihit][6]), fC[5] * fTupBuf[ihit][7] / (fC[5] + fTupBuf[ihit][7]), fC[9] * fTupBuf[ihit][8] / (fC[9] + fTupBuf[ihit][8]), fC[14] * fTupBuf[ihit][9] / (fC[14] + fTupBuf[ihit][9]));
      if (++fCount == 2000000) {
        GPUInfo("Reached %lld clusters in error stat, exiting", fCount);
        fTup.~GPUTPCGPURootDump<TNtuple>();
        exit(0);
      }
    }
  }
};

GPUTPCGPURootDump<TNtuple> GPUTPCClusterErrorStat::fTup("clusterres.root", "clusterres", "clusterres", "clX:clY:clZ:angle:trkX:trkY:trkZ:trkSinPhi:trkDzDs:trkQPt:trkSigmaY2:trkSigmaZ2:trkSigmaSinPhi2:trkSigmaDzDs2:trkSigmaQPt2");
long long int GPUTPCClusterErrorStat::fCount = 0;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterErrorStat {
  GPUd() GPUTPCClusterErrorStat(int /*maxN*/) {}
  GPUd() void Fill(float /*x*/, float /*y*/, float /*z*/, float /*alpha*/, float /*trkX*/, float* /*fP*/, float* /*fC*/, int /*ihit*/, int /*iWay*/) {}
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif

#endif
