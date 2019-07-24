// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMOfflineStatisticalErrors.h
/// \author David Rohr

#ifndef GPUTPCGMOFFLINESTATISTICALERRORS
#define GPUTPCGMOFFLINESTATISTICALERRORS

#if defined(GPUCA_TPC_USE_STAT_ERROR)
#include "AliTPCcalibDB.h"
#include "AliTPCclusterMI.h"
#include "AliTPCtracker.h"
#include "AliTPCTransform.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "AliTPCReconstructor.h"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCGMMergedTrackHit;

#if defined(GPUCA_TPC_USE_STAT_ERROR)
struct GPUTPCGMOfflineStatisticalErrors {
  void SetCurCluster(GPUTPCGMMergedTrackHit* c) { mCurCluster = c; }

  void GetOfflineStatisticalErrors(float& err2Y, float& err2Z, float sinPhi, float dzds, unsigned char clusterState) const
  {
    float snp2 = sinPhi * sinPhi;
    if (snp2 > 1. - 1e-6) {
      snp2 = 1. - 1e-6;
    }
    float tgp2 = snp2 / (1.f - snp2);
    float tgp = sqrt(tgp2);
    double serry2 = 0, serrz2 = 0;
    AliTPCclusterMI cl;
    cl.SetRow(mCurCluster->mRow);
    cl.SetPad(mCurCluster->mPad);
    cl.SetTimeBin(mCurCluster->fTime);
    int type = 0;
    if (clusterState & GPUTPCGMMergedTrackHit::flagSplit) {
      type = 50;
    }
    if (clusterState & GPUTPCGMMergedTrackHit::flagEdge) {
      type = -type - 3;
    }
    cl.SetType(type);
    cl.SetSigmaY2(0.5);
    cl.SetSigmaZ2(0.5);
    cl.SetQ(mCurCluster->fAmp);
    cl.SetMax(25);
    cl.SetX(mCurCluster->mX);
    cl.SetY(mCurCluster->fY);
    cl.SetZ(mCurCluster->fZ);
    cl.SetDetector(mCurCluster->mSlice);
    if (mCurCluster->mRow >= 63) {
      cl.SetRow(mCurCluster->mRow - 63);
      cl.SetDetector(mCurCluster->mSlice + 36);
    }
    static AliTPCtracker trk;
    /*AliTPCRecoParam *par = const_cast<AliTPCRecoParam*>(AliTPCReconstructor::GetRecoParam()), *par2;
                AliTPCReconstructor rec;
                if (par == nullptr)
                {
                    par2 = new AliTPCRecoParam;
                    par2->SetUseSectorAlignment(false);
                    rec.SetRecoParam(par2);
                }*/
    // This needs AliTPCRecoParam::GetUseSectorAlignment(), so we should make sure that the RecoParam exists!
    // This is not true during HLT simulation by definition, so the above code should fix it, but it leads to non-understandable bug later on in TPC transformation.
    // Anyway, this is only a debugging class.
    // So in order to make this work, please temporarily outcomment any use of TPCRecoParam in AliTPCTracker::Transform (it is not needed here anyway...)
    trk.Transform(&cl);
    /*if (par == nullptr)
                {
                    delete par2;
                    rec.SetRecoParam(nullptr);
                }*/

    AliTPCcalibDB::Instance()->GetTransform()->ErrY2Z2Syst(&cl, tgp, dzds, serry2, serrz2);

    // GPUInfo("TEST Sector %d Row %d: X %f %f Y %f %f Z %f %f - Err Y %f + %f, Z %f + %f", mCurCluster->mSlice, mCurCluster->mRow, mCurCluster->mX, cl.GetX(), mCurCluster->fY, cl.GetY(), mCurCluster->fZ, cl.GetZ(), err2Y, serry2, err2Z, serrz2);

    err2Y += serry2;
    err2Z += serrz2;
  }

  GPUTPCGMMergedTrackHit* mCurCluster;
};
#else
struct GPUTPCGMOfflineStatisticalErrors {
  GPUd() void SetCurCluster(GPUTPCGMMergedTrackHit* /*c*/) {}
  GPUd() void GetOfflineStatisticalErrors(float& /*err2Y*/, float& /*err2Z*/, float /*sinPhi*/, float /*dzds*/, unsigned char /*clusterState*/) const {}
};
#endif
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
