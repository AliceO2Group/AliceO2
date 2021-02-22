// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "GPUTPCClusterData.h"
#include "GPUReconstructionConvert.h"
#include "GPUMemorySizeScalers.h"
#include "AliHLTTPCRawCluster.h"

#ifdef HAVE_O2HEADERS
#include "DataFormatsTPC/ClusterNative.h"
#else
#include "GPUO2FakeClasses.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

int GPUChainTracking::ConvertNativeToClusterData()
{
#ifdef HAVE_O2HEADERS
  mRec->PushNonPersistentMemory();
  const auto& threadContext = GetThreadContext();
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCConversion;
  GPUTPCConvert& convert = processors()->tpcConverter;
  GPUTPCConvert& convertShadow = doGPU ? processorsShadow()->tpcConverter : convert;

  SetupGPUProcessor(&convert, true);
  if (doGPU) {
    if (!(mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding)) {
      mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mIOPtrs.clustersNative->nClustersTotal;
      AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
      processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
      WriteToConstantMemory(RecoStep::TPCConversion, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
      *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
      mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
      mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
      GPUMemCpy(RecoStep::TPCConversion, mInputsShadow->mPclusterNativeBuffer, mIOPtrs.clustersNative->clustersLinear, sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * mIOPtrs.clustersNative->nClustersTotal, 0, true);
      TransferMemoryResourceLinkToGPU(RecoStep::TPCConversion, mInputsHost->mResourceClusterNativeAccess, 0);
    }
  }
  if (!param().par.earlyTpcTransform) {
    if (GetProcessingSettings().debugLevel >= 3) {
      GPUInfo("Early transform inactive, skipping TPC Early transformation kernel, transformed on the fly during slice data creation / refit");
    }
    return 0;
  }
  for (unsigned int i = 0; i < NSLICES; i++) {
    convert.mMemory->clusters[i] = convertShadow.mClusters + mIOPtrs.clustersNative->clusterOffset[i][0];
  }

  WriteToConstantMemory(RecoStep::TPCConversion, (char*)&processors()->tpcConverter - (char*)processors(), &convertShadow, sizeof(convertShadow), 0);
  TransferMemoryResourcesToGPU(RecoStep::TPCConversion, &convert, 0);
  runKernel<GPUTPCConvertKernel>(GetGridBlk(NSLICES * GPUCA_ROW_COUNT, 0), krnlRunRangeNone, krnlEventNone);
  TransferMemoryResourcesToHost(RecoStep::TPCConversion, &convert, 0);
  SynchronizeStream(0);

  for (unsigned int i = 0; i < NSLICES; i++) {
    mIOPtrs.nClusterData[i] = (i == NSLICES - 1 ? mIOPtrs.clustersNative->nClustersTotal : mIOPtrs.clustersNative->clusterOffset[i + 1][0]) - mIOPtrs.clustersNative->clusterOffset[i][0];
    mIOPtrs.clusterData[i] = convert.mClusters + mIOPtrs.clustersNative->clusterOffset[i][0];
  }
  mRec->PopNonPersistentMemory(RecoStep::TPCConversion);
#endif
  return 0;
}

void GPUChainTracking::ConvertNativeToClusterDataLegacy()
{
  ClusterNativeAccess* tmp = mIOMem.clusterNativeAccess.get();
  if (tmp != mIOPtrs.clustersNative) {
    *tmp = *mIOPtrs.clustersNative;
  }
  GPUReconstructionConvert::ConvertNativeToClusterData(mIOMem.clusterNativeAccess.get(), mIOMem.clusterData, mIOPtrs.nClusterData, processors()->calibObjects.fastTransform, param().par.continuousMaxTimeBin);
  for (unsigned int i = 0; i < NSLICES; i++) {
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
  for (unsigned int i = 0; i < NSLICES; i++) {
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

void GPUChainTracking::ConvertZSEncoder(bool zs12bit)
{
#ifdef HAVE_O2HEADERS
  mIOMem.tpcZSmeta2.reset(new GPUTrackingInOutZS::GPUTrackingInOutZSMeta);
  mIOMem.tpcZSmeta.reset(new GPUTrackingInOutZS);
  GPUReconstructionConvert::RunZSEncoder<o2::tpc::Digit>(*mIOPtrs.tpcPackedDigits, &mIOMem.tpcZSpages, &mIOMem.tpcZSmeta2->n[0][0], nullptr, nullptr, param(), zs12bit, true);
  GPUReconstructionConvert::RunZSEncoderCreateMeta(mIOMem.tpcZSpages.get(), &mIOMem.tpcZSmeta2->n[0][0], &mIOMem.tpcZSmeta2->ptr[0][0], mIOMem.tpcZSmeta.get());
  mIOPtrs.tpcZS = mIOMem.tpcZSmeta.get();
  if (GetProcessingSettings().registerStandaloneInputMemory) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[i].count[j]; k++) {
          if (mRec->registerMemoryForGPU(mIOPtrs.tpcZS->slice[i].zsPtr[j][k], mIOPtrs.tpcZS->slice[i].nZSPtr[j][k] * TPCZSHDR::TPC_ZS_PAGE_SIZE)) {
            throw std::runtime_error("Error registering memory for GPU");
          }
        }
      }
    }
  }
#endif
}

void GPUChainTracking::ConvertZSFilter(bool zs12bit)
{
  GPUReconstructionConvert::RunZSFilter(mIOMem.tpcDigits, mIOPtrs.tpcPackedDigits->tpcDigits, mIOMem.digitMap->nTPCDigits, mIOPtrs.tpcPackedDigits->nTPCDigits, param(), zs12bit, param().rec.tpcZSthreshold);
}

int GPUChainTracking::ForwardTPCDigits()
{
#ifdef HAVE_O2HEADERS
  if (GetRecoStepsGPU() & RecoStep::TPCClusterFinding) {
    throw std::runtime_error("Cannot forward TPC digits with Clusterizer on GPU");
  }
  std::vector<ClusterNative> tmp[NSLICES][GPUCA_ROW_COUNT];
  unsigned int nTotal = 0;
  const float zsThreshold = param().rec.tpcZSthreshold;
  for (int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < mIOPtrs.tpcPackedDigits->nTPCDigits[i]; j++) {
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
  for (int i = 0; i < NSLICES; i++) {
    for (int j = 0; j < GPUCA_ROW_COUNT; j++) {
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
#endif
  return 0;
}
