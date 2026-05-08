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

/// \file GPUChainTrackingTransformation.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUTrackingInputProvider.h"
#include "GPUReconstructionConvert.h"
#include "GPUMemorySizeScalers.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterData.h"

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "utils/strtag.h"

using namespace o2::gpu;
using namespace o2::tpc;

bool GPUChainTracking::NeedTPCClustersOnGPU()
{
  return (mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCConversion) || (mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCSectorTracking) || (mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCMerging) || (mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCCompression);
}

int32_t GPUChainTracking::ConvertNativeToClusterData()
{
  mRec->PushNonPersistentMemory(qStr2Tag("TPCTRANS"));
  const auto& threadContext = GetThreadContext();

  bool transferClusters = false;
  if (mRec->IsGPU() && !(mRec->GetRecoStepsGPU() & gpudatatypes::RecoStep::TPCClusterFinding) && NeedTPCClustersOnGPU()) {
    mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mIOPtrs.clustersNative->nClustersTotal;
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
    processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
    WriteToConstantMemory(RecoStep::TPCConversion, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
    mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
    mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
    GPUMemCpy(RecoStep::TPCConversion, mInputsShadow->mPclusterNativeBuffer, mIOPtrs.clustersNative->clustersLinear, sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * mIOPtrs.clustersNative->nClustersTotal, 0, true);
    TransferMemoryResourceLinkToGPU(RecoStep::TPCConversion, mInputsHost->mResourceClusterNativeAccess, 0);
    transferClusters = true;
  }
  if (GetProcessingSettings().debugLevel >= 3) {
    GPUInfo("Early transform inactive, skipping TPC Early transformation kernel, transformed on the fly during sector data creation / refit");
  }
  if (transferClusters) {
    SynchronizeStream(0); // TODO: Synchronize implicitly with next step
  }
  return 0;
}

void GPUChainTracking::ConvertNativeToClusterDataLegacy()
{
  ClusterNativeAccess* tmp = mIOMem.clusterNativeAccess.get();
  if (tmp != mIOPtrs.clustersNative) {
    *tmp = *mIOPtrs.clustersNative;
  }
  GPUReconstructionConvert::ConvertNativeToClusterData(mIOMem.clusterNativeAccess.get(), mIOMem.clusterData, mIOPtrs.nClusterData, processors()->calibObjects.fastTransform, param().continuousMaxTimeBin);
  for (uint32_t i = 0; i < NSECTORS; i++) {
    mIOPtrs.clusterData[i] = mIOMem.clusterData[i].get();
    if (GetProcessingSettings().registerStandaloneInputMemory) {
      if (mRec->registerMemoryForGPU(mIOMem.clusterData[i].get(), mIOPtrs.nClusterData[i] * sizeof(*mIOPtrs.clusterData[i]))) {
        throw std::runtime_error("Error registering memory for GPU");
      }
    }
  }
  mIOPtrs.clustersNative = nullptr;
  mIOMem.clustersNative.reset(nullptr);
}

void GPUChainTracking::ConvertRun2RawToNative()
{
  GPUReconstructionConvert::ConvertRun2RawToNative(*mIOMem.clusterNativeAccess, mIOMem.clustersNative, mIOPtrs.rawClusters, mIOPtrs.nRawClusters);
  for (uint32_t i = 0; i < NSECTORS; i++) {
    mIOPtrs.rawClusters[i] = nullptr;
    mIOPtrs.nRawClusters[i] = 0;
    mIOMem.rawClusters[i].reset(nullptr);
    mIOPtrs.clusterData[i] = nullptr;
    mIOPtrs.nClusterData[i] = 0;
    mIOMem.clusterData[i].reset(nullptr);
  }
  mIOPtrs.clustersNative = mIOMem.clusterNativeAccess.get();
  if (GetProcessingSettings().registerStandaloneInputMemory) {
    if (mRec->registerMemoryForGPU(mIOMem.clustersNative.get(), mIOMem.clusterNativeAccess->nClustersTotal * sizeof(*mIOMem.clusterNativeAccess->clustersLinear))) {
      throw std::runtime_error("Error registering memory for GPU");
    }
  }
}

void GPUChainTracking::ConvertZSEncoder(int32_t version)
{
  mIOMem.tpcZSmeta2.reset(new GPUTrackingInOutZS::GPUTrackingInOutZSMeta);
  mIOMem.tpcZSmeta.reset(new GPUTrackingInOutZS);
  o2::InteractionRecord ir{0, mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasTfStartOrbit ? mIOPtrs.settingsTF->tfStartOrbit : 0u};
  GPUReconstructionConvert::RunZSEncoder(*mIOPtrs.tpcPackedDigits, &mIOMem.tpcZSpages, &mIOMem.tpcZSmeta2->n[0][0], nullptr, &ir, param(), version, true);
  GPUReconstructionConvert::RunZSEncoderCreateMeta(mIOMem.tpcZSpages.get(), &mIOMem.tpcZSmeta2->n[0][0], &mIOMem.tpcZSmeta2->ptr[0][0], mIOMem.tpcZSmeta.get());
  mIOPtrs.tpcZS = mIOMem.tpcZSmeta.get();
  if (GetProcessingSettings().registerStandaloneInputMemory) {
    for (uint32_t i = 0; i < NSECTORS; i++) {
      for (uint32_t j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (uint32_t k = 0; k < mIOPtrs.tpcZS->sector[i].count[j]; k++) {
          if (mRec->registerMemoryForGPU(mIOPtrs.tpcZS->sector[i].zsPtr[j][k], mIOPtrs.tpcZS->sector[i].nZSPtr[j][k] * TPCZSHDR::TPC_ZS_PAGE_SIZE)) {
            throw std::runtime_error("Error registering memory for GPU");
          }
        }
      }
    }
  }
}

void GPUChainTracking::ConvertZSFilter(bool zs12bit)
{
  GPUReconstructionConvert::RunZSFilter(mIOMem.tpcDigits, mIOPtrs.tpcPackedDigits->tpcDigits, mIOMem.digitMap->nTPCDigits, mIOPtrs.tpcPackedDigits->nTPCDigits, param(), zs12bit, param().rec.tpc.zsThreshold);
}

int32_t GPUChainTracking::ForwardTPCDigits()
{
  if (GetRecoStepsGPU() & RecoStep::TPCClusterFinding) {
    throw std::runtime_error("Cannot forward TPC digits with Clusterizer on GPU");
  }
  std::vector<ClusterNative> tmp[NSECTORS][GPUTPCGeometry::NROWS];
  uint32_t nTotal = 0;
  const float zsThreshold = param().rec.tpc.zsThreshold;
  for (int32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < mIOPtrs.tpcPackedDigits->nTPCDigits[i]; j++) {
      const auto& d = mIOPtrs.tpcPackedDigits->tpcDigits[i][j];
      if (d.getChargeFloat() >= zsThreshold) {
        ClusterNative c;
        c.setTimeFlags(d.getTimeStamp(), 0);
        c.setPad(d.getPad());
        c.setSigmaTime(1);
        c.setSigmaPad(1);
        c.qTot = c.qMax = d.getChargeFloat();
        tmp[i][d.getRow()].emplace_back(c);
        nTotal++;
      }
    }
  }
  mIOMem.clustersNative.reset(new ClusterNative[nTotal]);
  nTotal = 0;
  mClusterNativeAccess->clustersLinear = mIOMem.clustersNative.get();
  for (uint32_t i = 0; i < NSECTORS; i++) {
    for (uint32_t j = 0; j < GPUTPCGeometry::NROWS; j++) {
      mClusterNativeAccess->nClusters[i][j] = tmp[i][j].size();
      memcpy(&mIOMem.clustersNative[nTotal], tmp[i][j].data(), tmp[i][j].size() * sizeof(*mClusterNativeAccess->clustersLinear));
      nTotal += tmp[i][j].size();
    }
  }
  mClusterNativeAccess->setOffsetPtrs();
  mIOPtrs.tpcPackedDigits = nullptr;
  mIOPtrs.clustersNative = mClusterNativeAccess.get();
  GPUInfo("Forwarded %u TPC clusters", nTotal);
  mRec->MemoryScalers()->nTPCHits = nTotal;
  return 0;
}
