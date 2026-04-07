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

/// \file GPUDisplay.cxx
/// \author David Rohr

#ifndef GPUCA_NO_ROOT
#include "Rtypes.h" // Include ROOT header first, to use ROOT and disable replacements
#endif

#include "GPUDisplay.h"
#include "frontend/GPUDisplayInfo.inc"
#include "GPUO2DataTypes.h"
#include "GPUTPCConvertImpl.h"
#include "GPUTRDGeometry.h"
#include "GPUTRDTrackletWord.h"
#include "GPUChainTracking.h"
#include "GPUParam.inc"

#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "TOFBase/Geo.h"
#include "ITSBase/GeometryTGeo.h"
#ifndef GPUCA_STANDALONE
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#endif

#include <oneapi/tbb.h>

using namespace o2::gpu;

void GPUDisplay::DrawGLScene_updateEventData()
{
  HighResTimer timer(mChain->GetProcessingSettings().debugLevel >= 2);
  if (mIOPtrs->clustersNative) {
    mCurrentClusters = mIOPtrs->clustersNative->nClustersTotal;
  } else {
    mCurrentClusters = 0;
    for (int32_t iSector = 0; iSector < NSECTORS; iSector++) {
      mCurrentClusters += mIOPtrs->nClusterData[iSector];
    }
  }
  if (mNMaxClusters < mCurrentClusters) {
    mNMaxClusters = mCurrentClusters;
    mGlobalPosPtr.reset(new float4[mNMaxClusters]);
    mGlobalPos = mGlobalPosPtr.get();
  }

  mCurrentSpacePointsTRD = mIOPtrs->nTRDTracklets;
  if (mCurrentSpacePointsTRD > mNMaxSpacePointsTRD) {
    mNMaxSpacePointsTRD = mCurrentSpacePointsTRD;
    mGlobalPosPtrTRD.reset(new float4[mNMaxSpacePointsTRD]);
    mGlobalPosPtrTRD2.reset(new float4[mNMaxSpacePointsTRD]);
    mGlobalPosTRD = mGlobalPosPtrTRD.get();
    mGlobalPosTRD2 = mGlobalPosPtrTRD2.get();
  }

  mCurrentClustersITS = mIOPtrs->itsClusters ? mIOPtrs->nItsClusters : 0;
  if (mNMaxClustersITS < mCurrentClustersITS) {
    mNMaxClustersITS = mCurrentClustersITS;
    mGlobalPosPtrITS.reset(new float4[mNMaxClustersITS]);
    mGlobalPosITS = mGlobalPosPtrITS.get();
  }

  mCurrentClustersTOF = mIOPtrs->nTOFClusters;
  if (mNMaxClustersTOF < mCurrentClustersTOF) {
    mNMaxClustersTOF = mCurrentClustersTOF;
    mGlobalPosPtrTOF.reset(new float4[mNMaxClustersTOF]);
    mGlobalPosTOF = mGlobalPosPtrTOF.get();
  }

  uint32_t nTpcMergedTracks = mConfig.showTPCTracksFromO2Format ? mIOPtrs->nOutputTracksTPCO2 : mIOPtrs->nMergedTracks;
  if ((size_t)nTpcMergedTracks > mTRDTrackIds.size()) {
    mTRDTrackIds.resize(nTpcMergedTracks);
  }
  if (mIOPtrs->nItsTracks > mITSStandaloneTracks.size()) {
    mITSStandaloneTracks.resize(mIOPtrs->nItsTracks);
  }
  for (uint32_t i = 0; i < nTpcMergedTracks; i++) {
    mTRDTrackIds[i] = -1;
  }
  auto tmpDoTRDTracklets = [&](auto* trdTracks) {
    for (uint32_t i = 0; i < mIOPtrs->nTRDTracks; i++) {
      if (trdTracks[i].getNtracklets()) {
        mTRDTrackIds[trdTracks[i].getRefGlobalTrackIdRaw()] = i;
      }
    }
  };
  if (mIOPtrs->trdTracksO2) {
    tmpDoTRDTracklets(mIOPtrs->trdTracksO2);
  } else {
    tmpDoTRDTracklets(mIOPtrs->trdTracks);
  }
  if (mIOPtrs->nItsTracks) {
    std::fill(mITSStandaloneTracks.begin(), mITSStandaloneTracks.end(), true);
    if (mIOPtrs->tpcLinkITS) {
      for (uint32_t i = 0; i < nTpcMergedTracks; i++) {
        if (mIOPtrs->tpcLinkITS[i] != -1) {
          mITSStandaloneTracks[mIOPtrs->tpcLinkITS[i]] = false;
        }
      }
    }
  }
  if (timer.IsRunning()) {
    GPUInfo("Display Time: Init:\t\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
  }

  if (mCfgH.trackFilter) {
    uint32_t nTracks = mConfig.showTPCTracksFromO2Format ? mIOPtrs->nOutputTracksTPCO2 : mIOPtrs->nMergedTracks;
    mTrackFilter.resize(nTracks);
    std::fill(mTrackFilter.begin(), mTrackFilter.end(), true);
    if (buildTrackFilter()) {
      SetInfo("Error running track filter from %s", mConfig.filterMacros[mCfgH.trackFilter - 1].c_str());
    } else {
      uint32_t nFiltered = 0;
      for (uint32_t i = 0; i < mTrackFilter.size(); i++) {
        nFiltered += !mTrackFilter[i];
      }
      if (mUpdateTrackFilter) {
        SetInfo("Applied track filter %s - filtered %u / %u", mConfig.filterMacros[mCfgH.trackFilter - 1].c_str(), nFiltered, (uint32_t)mTrackFilter.size());
      }
    }
  }
  mUpdateTrackFilter = false;
  if (timer.IsRunning()) {
    GPUInfo("Display Time: Track Filter:\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
  }

  mMaxClusterZ = tbb::parallel_reduce(tbb::blocked_range<int32_t>(0, NSECTORS, 1), float(0.f), [&](const tbb::blocked_range<int32_t>& r, float maxClusterZ) {
    for (int32_t iSector = r.begin(); iSector < r.end(); iSector++) {
      uint32_t row = 0;
      uint32_t nCls = mIOPtrs->clustersNative ? mIOPtrs->clustersNative->nClustersSector[iSector] : 0;
      for (uint32_t i = 0; i < nCls; i++) {
        int32_t cid;
        cid = mIOPtrs->clustersNative->clusterOffset[iSector][0] + i;
        while (row < GPUTPCGeometry::NROWS - 1 && mIOPtrs->clustersNative->clusterOffset[iSector][row + 1] <= (uint32_t)cid) {
          row++;
        }
        if (cid >= mNMaxClusters) {
          throw std::runtime_error("Cluster Buffer Size exceeded");
        }
        float4* ptr = &mGlobalPos[cid];
        float x, y, z;
        const auto& cln = mIOPtrs->clustersNative->clusters[iSector][0][i];
        GPUTPCConvertImpl::convert(*mCalib->fastTransform, *mParam, iSector, row, cln.getPad(), cln.getTime(), x, y, z);
        if (mCfgH.clustersOnNominalRow) {
          x = GPUTPCGeometry::Row2X(row);
        }
        mParam->Sector2Global(iSector, x + mCfgH.xAdd, y, z, &ptr->x, &ptr->y, &ptr->z);

        if (fabsf(ptr->z) > maxClusterZ) {
          maxClusterZ = fabsf(ptr->z);
        }
        ptr->z += iSector < 18 ? mCfgH.zAdd : -mCfgH.zAdd;
        ptr->x *= GL_SCALE_FACTOR;
        ptr->y *= GL_SCALE_FACTOR;
        ptr->z *= GL_SCALE_FACTOR;
        ptr->w = tCLUSTER;
      }
    }
    return maxClusterZ; // clang-format off
  }, [](const float a, const float b) { return std::max(a, b); }, tbb::simple_partitioner()); // clang-format on
  if (timer.IsRunning()) {
    GPUInfo("Display Time: Load TPC:\t\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
  }

  mMaxClusterZ = tbb::parallel_reduce(tbb::blocked_range<int32_t>(0, mCurrentSpacePointsTRD, 32), float(mMaxClusterZ), [&](const tbb::blocked_range<int32_t>& r, float maxClusterZ) {
    int32_t trdTriggerRecord = -1;
    float trdZoffset = 0;
    for (int i = r.begin(); i < r.end(); i++) {
      while (mParam->par.continuousTracking && trdTriggerRecord < (int32_t)mIOPtrs->nTRDTriggerRecords - 1 && mIOPtrs->trdTrackletIdxFirst[trdTriggerRecord + 1] <= i) {
        trdTriggerRecord++; // This requires to go through the data in order I believe
        float trdTime = mIOPtrs->trdTriggerTimes[trdTriggerRecord] * 1e3 / o2::constants::lhc::LHCBunchSpacingNS / o2::tpc::constants::LHCBCPERTIMEBIN;
        trdZoffset = fabsf(mCalib->fastTransform->convVertexTimeToZOffset(0, trdTime, mParam->continuousMaxTimeBin));
      }
      const auto& sp = mIOPtrs->trdSpacePoints[i];
      int32_t iSec = trdGeometry()->GetSector(mIOPtrs->trdTracklets[i].GetDetector());
      float4* ptr = &mGlobalPosTRD[i];
      mParam->Sector2Global(iSec, sp.getX() + mCfgH.xAdd, sp.getY(), sp.getZ(), &ptr->x, &ptr->y, &ptr->z);
      ptr->z += ptr->z > 0 ? trdZoffset : -trdZoffset;
      if (fabsf(ptr->z) > maxClusterZ) {
        maxClusterZ = fabsf(ptr->z);
      }
      ptr->x *= GL_SCALE_FACTOR;
      ptr->y *= GL_SCALE_FACTOR;
      ptr->z *= GL_SCALE_FACTOR;
      ptr->w = tTRDCLUSTER;
      ptr = &mGlobalPosTRD2[i];
      mParam->Sector2Global(iSec, sp.getX() + mCfgH.xAdd + 4.5f, sp.getY() + 1.5f * sp.getDy(), sp.getZ(), &ptr->x, &ptr->y, &ptr->z);
      ptr->z += ptr->z > 0 ? trdZoffset : -trdZoffset;
      if (fabsf(ptr->z) > maxClusterZ) {
        maxClusterZ = fabsf(ptr->z);
      }
      ptr->x *= GL_SCALE_FACTOR;
      ptr->y *= GL_SCALE_FACTOR;
      ptr->z *= GL_SCALE_FACTOR;
      ptr->w = tTRDCLUSTER;
    }
    return maxClusterZ; // clang-format off
  }, [](const float a, const float b) { return std::max(a, b); }, tbb::static_partitioner()); // clang-format on
  if (timer.IsRunning()) {
    GPUInfo("Display Time: Load TRD:\t\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
  }

  mMaxClusterZ = tbb::parallel_reduce(tbb::blocked_range<int32_t>(0, mCurrentClustersTOF, 32), float(mMaxClusterZ), [&](const tbb::blocked_range<int32_t>& r, float maxClusterZ) {
    for (int32_t i = r.begin(); i < r.end(); i++) {
      float4* ptr = &mGlobalPosTOF[i];
      mParam->Sector2Global(mIOPtrs->tofClusters[i].getSector(), mIOPtrs->tofClusters[i].getX() + mCfgH.xAdd, mIOPtrs->tofClusters[i].getY(), mIOPtrs->tofClusters[i].getZ(), &ptr->x, &ptr->y, &ptr->z);
      float ZOffset = 0;
      if (mParam->par.continuousTracking) {
        float tofTime = mIOPtrs->tofClusters[i].getTime() * 1e-3 / o2::constants::lhc::LHCBunchSpacingNS / o2::tpc::constants::LHCBCPERTIMEBIN;
        ZOffset = fabsf(mCalib->fastTransform->convVertexTimeToZOffset(0, tofTime, mParam->continuousMaxTimeBin));
        ptr->z += ptr->z > 0 ? ZOffset : -ZOffset;
      }
      if (fabsf(ptr->z) > maxClusterZ) {
        maxClusterZ = fabsf(ptr->z);
      }
      ptr->x *= GL_SCALE_FACTOR;
      ptr->y *= GL_SCALE_FACTOR;
      ptr->z *= GL_SCALE_FACTOR;
      ptr->w = tTOFCLUSTER;
    }
    return maxClusterZ; // clang-format off
  }, [](const float a, const float b) { return std::max(a, b); }); // clang-format on
  if (timer.IsRunning()) {
    GPUInfo("Display Time: Load TOF:\t\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
  }

  if (mCurrentClustersITS) {
    float itsROFhalfLen = 0;
#ifndef GPUCA_STANDALONE // Not available in standalone benchmark
    if (mParam->par.continuousTracking) {
      const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
      itsROFhalfLen = alpParams.roFrameLengthInBC / (float)o2::tpc::constants::LHCBCPERTIMEBIN / 2;
    }
#endif
    int32_t i = 0;
    for (uint32_t j = 0; j < mIOPtrs->nItsClusterROF; j++) {
      float ZOffset = 0;
      if (mParam->par.continuousTracking) {
        o2::InteractionRecord startIR = o2::InteractionRecord(0, mIOPtrs->settingsTF && mIOPtrs->settingsTF->hasTfStartOrbit ? mIOPtrs->settingsTF->tfStartOrbit : 0);
        float itsROFtime = mIOPtrs->itsClusterROF[j].getBCData().differenceInBC(startIR) / (float)o2::tpc::constants::LHCBCPERTIMEBIN;
        ZOffset = fabsf(mCalib->fastTransform->convVertexTimeToZOffset(0, itsROFtime + itsROFhalfLen, mParam->continuousMaxTimeBin));
      }
      if (i != mIOPtrs->itsClusterROF[j].getFirstEntry()) {
        throw std::runtime_error("Inconsistent ITS data, number of clusters does not match ROF content");
      }
      for (int32_t k = 0; k < mIOPtrs->itsClusterROF[j].getNEntries(); k++) {
        float4* ptr = &mGlobalPosITS[i];
        const auto& cl = mIOPtrs->itsClusters[i];
        auto* itsGeo = o2::its::GeometryTGeo::Instance();
        auto p = cl.getXYZGlo(*itsGeo);
        ptr->x = p.X();
        ptr->y = p.Y();
        ptr->z = p.Z();
        ptr->z += ptr->z > 0 ? ZOffset : -ZOffset;
        if (fabsf(ptr->z) > mMaxClusterZ) {
          mMaxClusterZ = fabsf(ptr->z);
        }
        ptr->x *= GL_SCALE_FACTOR;
        ptr->y *= GL_SCALE_FACTOR;
        ptr->z *= GL_SCALE_FACTOR;
        ptr->w = tITSCLUSTER;
        i++;
      }
    }
  }
  if (timer.IsRunning()) {
    GPUInfo("Display Time: Load ITS:\t\t%6.0f us", timer.GetCurrentElapsedTime(true) * 1e6);
  }
}
