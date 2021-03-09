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

#define EXTRACT_RESIDUALS

#if (defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_BUILD_QA)) && !defined(GPUCA_GPUCODE) && defined(EXTRACT_RESIDUALS)
#include "GPUCommonRtypes.h"
#include "GPUROOTDump.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterErrorStat {
  GPUTPCClusterErrorStat(int maxN) : mTupBuf(maxN) {}

  static long long int mCount;

  std::vector<std::array<float, 10>> mTupBuf;

  void Fill(float x, float y, float z, float alpha, float trkX, float* fP, float* fC, int ihit, int iWay)
  {
    static GPUROOTDump<TNtuple> tup("clusterres", "clX:clY:clZ:angle:trkX:trkY:trkZ:trkSinPhi:trkDzDs:trkQPt:trkSigmaY2:trkSigmaZ2:trkSigmaSinPhi2:trkSigmaDzDs2:trkSigmaQPt2");

    if (iWay == 1) {
      mTupBuf[ihit] = {fP[0], fP[1], fP[2], fP[3], fP[4], fC[0], fC[2], fC[5], fC[9], fC[14]};
    } else if (iWay == 2) {
      tup.Fill(x, y, z, alpha, trkX, (fP[0] * mTupBuf[ihit][5] + mTupBuf[ihit][0] * fC[0]) / (mTupBuf[ihit][5] + fC[0]), (fP[1] * mTupBuf[ihit][6] + mTupBuf[ihit][1] * fC[2]) / (mTupBuf[ihit][6] + fC[2]), (fP[2] * mTupBuf[ihit][7] + mTupBuf[ihit][2] * fC[5]) / (mTupBuf[ihit][7] + fC[5]),
               (fP[3] * mTupBuf[ihit][8] + mTupBuf[ihit][3] * fC[9]) / (mTupBuf[ihit][8] + fC[9]), (fP[4] * mTupBuf[ihit][9] + mTupBuf[ihit][4] * fC[14]) / (mTupBuf[ihit][9] + fC[14]), fC[0] * mTupBuf[ihit][5] / (fC[0] + mTupBuf[ihit][5]),
               fC[2] * mTupBuf[ihit][6] / (fC[2] + mTupBuf[ihit][6]), fC[5] * mTupBuf[ihit][7] / (fC[5] + mTupBuf[ihit][7]), fC[9] * mTupBuf[ihit][8] / (fC[9] + mTupBuf[ihit][8]), fC[14] * mTupBuf[ihit][9] / (fC[14] + mTupBuf[ihit][9]));
    }
  }
};

long long int GPUTPCClusterErrorStat::mCount = 0;
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
