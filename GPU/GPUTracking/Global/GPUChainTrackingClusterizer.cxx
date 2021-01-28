// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTrackingClusterizer.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUChainTrackingDefs.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"
#include <fstream>

#ifdef GPUCA_O2_LIB
#include "CommonDataFormat/InteractionRecord.h"
#endif
#ifdef HAVE_O2HEADERS
#include "GPUHostDataTypes.h"
#include "GPUTPCCFChainContext.h"
#include "GPURawData.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/Constants.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#ifndef GPUCA_NO_VC
#include <Vc/Vc>
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::tpc::constants;
using namespace o2::dataformats;

#ifdef GPUCA_TPC_GEOMETRY_O2
std::pair<unsigned int, unsigned int> GPUChainTracking::TPCClusterizerDecodeZSCountUpdate(unsigned int iSlice, const CfFragment& fragment)
{
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
  GPUTPCClusterFinder::ZSOffset* o = processors()->tpcClusterer[iSlice].mPzsOffsets;
  unsigned int digits = 0;
  unsigned short pages = 0;
  for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    unsigned short posInEndpoint = 0;
    unsigned short pagesEndpoint = 0;
    clusterer.mMinMaxCN[j] = mCFContext->fragmentData[fragment.index].minMaxCN[iSlice][j];
    if (doGPU) {
      for (unsigned int k = clusterer.mMinMaxCN[j].minC; k < clusterer.mMinMaxCN[j].maxC; k++) {
        const unsigned int minL = (k == clusterer.mMinMaxCN[j].minC) ? clusterer.mMinMaxCN[j].minN : 0;
        const unsigned int maxL = (k + 1 == clusterer.mMinMaxCN[j].maxC) ? clusterer.mMinMaxCN[j].maxN : mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
        for (unsigned int l = minL; l < maxL; l++) {
          unsigned short pageDigits = mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j][posInEndpoint++];
          if (pageDigits) {
            *(o++) = GPUTPCClusterFinder::ZSOffset{digits, j, pagesEndpoint};
            digits += pageDigits;
          }
          pagesEndpoint++;
        }
      }
      pages += pagesEndpoint;
    } else {
      clusterer.mPzsOffsets[j] = GPUTPCClusterFinder::ZSOffset{digits, j, 0};
      digits += mCFContext->fragmentData[fragment.index].nDigits[iSlice][j];
      pages += mCFContext->fragmentData[fragment.index].nPages[iSlice][j];
    }
  }

  return {digits, pages};
}

std::pair<unsigned int, unsigned int> GPUChainTracking::TPCClusterizerDecodeZSCount(unsigned int iSlice, const CfFragment& fragment)
{
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Start();
  unsigned int nDigits = 0;
  unsigned int nPages = 0;
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  int firstHBF = o2::raw::RDHUtils::getHeartBeatOrbit(*(const RAWDataHeaderGPU*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0]);

  for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    for (unsigned int k = 0; k < mCFContext->nFragments; k++) {
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxC = mIOPtrs.tpcZS->slice[iSlice].count[j];
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].minC = mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxC;
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxN = mIOPtrs.tpcZS->slice[iSlice].count[j] ? mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][mIOPtrs.tpcZS->slice[iSlice].count[j] - 1] : 0;
      mCFContext->fragmentData[k].minMaxCN[iSlice][j].minN = mCFContext->fragmentData[k].minMaxCN[iSlice][j].maxN;
    }

#ifndef GPUCA_NO_VC
    if (GetProcessingSettings().prefetchTPCpageScan >= 3 && j < GPUTrackingInOutZS::NENDPOINTS - 1) {
      for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j + 1]; k++) {
        for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j + 1][k]; l++) {
          Vc::Common::prefetchMid(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchMid(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(RAWDataHeaderGPU));
        }
      }
    }
#endif

    bool firstNextFound = false;
    CfFragment f = fragment;
    CfFragment fNext = f.next();
    bool firstSegment = true;

    for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
      for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
        if (f.isEnd()) {
          GPUError("Time bin passed last fragment");
          return {0, 0};
        }
#ifndef GPUCA_NO_VC
        if (GetProcessingSettings().prefetchTPCpageScan >= 2 && l + 1 < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]) {
          Vc::Common::prefetchForOneRead(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchForOneRead(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(RAWDataHeaderGPU));
        }
#endif
        const unsigned char* const page = ((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        const RAWDataHeaderGPU* rdh = (const RAWDataHeaderGPU*)page;
        if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(RAWDataHeaderGPU)) {
          nPages++;
          if (mCFContext->fragmentData[f.index].nPages[iSlice][j]) {
            mCFContext->fragmentData[f.index].nPages[iSlice][j]++;
            mCFContext->fragmentData[f.index].pageDigits[iSlice][j].emplace_back(0);
          }
          continue;
        }
        const TPCZSHDR* const hdr = (const TPCZSHDR*)(page + sizeof(RAWDataHeaderGPU));
        unsigned int timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
        for (unsigned int m = 0; m < 2; m++) {
          CfFragment& fm = m ? fNext : f;
          if (timeBin + hdr->nTimeBins >= (unsigned int)fm.first() && timeBin < (unsigned int)fm.last() && !fm.isEnd()) {
            mCFContext->fragmentData[fm.index].nPages[iSlice][j]++;
            mCFContext->fragmentData[fm.index].nDigits[iSlice][j] += hdr->nADCsamples;
            if (doGPU) {
              mCFContext->fragmentData[fm.index].pageDigits[iSlice][j].emplace_back(hdr->nADCsamples);
            }
          }
        }
        if (firstSegment && timeBin + hdr->nTimeBins >= (unsigned int)f.first()) {
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].minC = k;
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].minN = l;
          firstSegment = false;
        }
        if (!firstNextFound && timeBin + hdr->nTimeBins >= (unsigned int)fNext.first() && !fNext.isEnd()) {
          mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minC = k;
          mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minN = l;
          firstNextFound = true;
        }
        while (!f.isEnd() && timeBin >= (unsigned int)f.last()) {
          if (!firstNextFound && !fNext.isEnd()) {
            mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minC = k;
            mCFContext->fragmentData[fNext.index].minMaxCN[iSlice][j].minN = l;
          }
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].maxC = k + 1;
          mCFContext->fragmentData[f.index].minMaxCN[iSlice][j].maxN = l;
          f = fNext;
          fNext = f.next();
          firstNextFound = false;
        }
        if (timeBin + hdr->nTimeBins > mCFContext->tpcMaxTimeBin) {
          mCFContext->tpcMaxTimeBin = timeBin + hdr->nTimeBins;
        }

        nPages++;
        nDigits += hdr->nADCsamples;
      }
    }
  }
  mCFContext->nPagesTotal += nPages;
  mCFContext->nPagesSector[iSlice] = nPages;
  mCFContext->nPagesSectorMax = std::max(mCFContext->nPagesSectorMax, nPages);

  unsigned int digitsFragment = 0;
  for (unsigned int i = 0; i < mCFContext->nFragments; i++) {
    unsigned int pages = 0;
    unsigned int digits = 0;
    for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      pages += mCFContext->fragmentData[i].nPages[iSlice][j];
      digits += mCFContext->fragmentData[i].nDigits[iSlice][j];
    }
    mCFContext->nPagesFragmentMax = std::max(mCFContext->nPagesSectorMax, pages);
    digitsFragment = std::max(digitsFragment, digits);
  }
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Stop();
  return {nDigits, digitsFragment};
}

void GPUChainTracking::RunTPCClusterizer_compactPeaks(GPUTPCClusterFinder& clusterer, GPUTPCClusterFinder& clustererShadow, int stage, bool doGPU, int lane)
{
  auto& in = stage ? clustererShadow.mPpeakPositions : clustererShadow.mPpositions;
  auto& out = stage ? clustererShadow.mPfilteredPeakPositions : clustererShadow.mPpeakPositions;
  if (doGPU) {
    const unsigned int iSlice = clusterer.mISlice;
    auto& count = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;

    std::vector<size_t> counts;

    unsigned int nSteps = clusterer.getNSteps(count);
    if (nSteps > clusterer.mNBufs) {
      GPUError("Clusterer buffers exceeded (%u > %u)", nSteps, (int)clusterer.mNBufs);
      exit(1);
    }

    size_t tmpCount = count;
    if (nSteps > 1) {
      for (unsigned int i = 1; i < nSteps; i++) {
        counts.push_back(tmpCount);
        if (i == 1) {
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanStart>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, stage);
        } else {
          runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanUp>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, tmpCount);
        }
        tmpCount = (tmpCount + clusterer.mScanWorkGroupSize - 1) / clusterer.mScanWorkGroupSize;
      }

      runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanTop>(GetGrid(tmpCount, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, nSteps, tmpCount);

      for (unsigned int i = nSteps - 1; i > 1; i--) {
        tmpCount = counts[i - 1];
        runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::scanDown>(GetGrid(tmpCount - clusterer.mScanWorkGroupSize, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, i, clusterer.mScanWorkGroupSize, tmpCount);
      }
    }

    runKernel<GPUTPCCFStreamCompaction, GPUTPCCFStreamCompaction::compactDigits>(GetGrid(count, clusterer.mScanWorkGroupSize, lane), {iSlice}, {}, 1, stage, in, out);
  } else {
    auto& nOut = stage ? clusterer.mPmemory->counters.nClusters : clusterer.mPmemory->counters.nPeaks;
    auto& nIn = stage ? clusterer.mPmemory->counters.nPeaks : clusterer.mPmemory->counters.nPositions;
    size_t count = 0;
    for (size_t i = 0; i < nIn; i++) {
      if (clusterer.mPisPeak[i]) {
        out[count++] = in[i];
      }
    }
    nOut = count;
  }
}

std::pair<unsigned int, unsigned int> GPUChainTracking::RunTPCClusterizer_transferZS(int iSlice, const CfFragment& fragment, int lane)
{
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  const auto& retVal = TPCClusterizerDecodeZSCountUpdate(iSlice, fragment);
  if (doGPU) {
    GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
    GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
    unsigned int nPagesSector = 0;
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      unsigned int nPages = 0;
      mInputsHost->mPzsMeta->slice[iSlice].zsPtr[j] = &mInputsShadow->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j];
      mInputsHost->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j] = clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE;
      for (unsigned int k = clusterer.mMinMaxCN[j].minC; k < clusterer.mMinMaxCN[j].maxC; k++) {
        const unsigned int min = (k == clusterer.mMinMaxCN[j].minC) ? clusterer.mMinMaxCN[j].minN : 0;
        const unsigned int max = (k + 1 == clusterer.mMinMaxCN[j].maxC) ? clusterer.mMinMaxCN[j].maxN : mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
        if (max > min) {
          char* src = (char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k] + min * TPCZSHDR::TPC_ZS_PAGE_SIZE;
          size_t size = o2::raw::RDHUtils::getMemorySize(*(const RAWDataHeaderGPU*)src);
          size = (max - min - 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE + (size ? TPCZSHDR::TPC_ZS_PAGE_SIZE : size);
          GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE, src, size, lane, true);
        }
        nPages += max - min;
      }
      mInputsHost->mPzsMeta->slice[iSlice].nZSPtr[j] = &mInputsShadow->mPzsSizes[iSlice * GPUTrackingInOutZS::NENDPOINTS + j];
      mInputsHost->mPzsSizes[iSlice * GPUTrackingInOutZS::NENDPOINTS + j] = nPages;
      mInputsHost->mPzsMeta->slice[iSlice].count[j] = 1;
      nPagesSector += nPages;
    }
    GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPzsOffsets, clusterer.mPzsOffsets, clusterer.mNMaxPages * sizeof(*clusterer.mPzsOffsets), lane, true);
  }
  return retVal;
}

int GPUChainTracking::RunTPCClusterizer_prepare(bool restorePointers)
{
  if (restorePointers) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcClusterer[iSlice].mPzsOffsets = mCFContext->ptrSave[iSlice].zsOffsetHost;
      processorsShadow()->tpcClusterer[iSlice].mPzsOffsets = mCFContext->ptrSave[iSlice].zsOffsetDevice;
      processorsShadow()->tpcClusterer[iSlice].mPzs = mCFContext->ptrSave[iSlice].zsDevice;
    }
    processorsShadow()->ioPtrs.clustersNative = mCFContext->ptrClusterNativeSave;
    return 0;
  }
  const auto& threadContext = GetThreadContext();
  mRec->MemoryScalers()->nTPCdigits = 0;
  if (mCFContext == nullptr) {
    mCFContext.reset(new GPUTPCCFChainContext);
  }
  mCFContext->tpcMaxTimeBin = param().par.ContinuousTracking ? std::max<int>(param().par.continuousMaxTimeBin, TPC_MAX_FRAGMENT_LEN) : TPC_MAX_TIME_BIN_TRIGGERED;
  const CfFragment fragmentMax{(tpccf::TPCTime)mCFContext->tpcMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN};
  mCFContext->prepare(mIOPtrs.tpcZS, fragmentMax);
  if (mIOPtrs.tpcZS) {
    unsigned int nDigitsFragment[NSLICES];
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (mIOPtrs.tpcZS->slice[iSlice].count[0] == 0) {
        GPUError("No ZS data present, must contain at least empty HBF");
        return 1;
      }
      const void* rdh = mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0];
      if (o2::raw::RDHUtils::getVersion<o2::gpu::RAWDataHeaderGPU>() != o2::raw::RDHUtils::getVersion(rdh)) {
        GPUError("Data has invalid RDH version %d, %d required\n", o2::raw::RDHUtils::getVersion(rdh), o2::raw::RDHUtils::getVersion<o2::gpu::RAWDataHeaderGPU>());
        return 1;
      }
#ifndef GPUCA_NO_VC
      if (GetProcessingSettings().prefetchTPCpageScan >= 1 && iSlice < NSLICES - 1) {
        for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
            for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
              Vc::Common::prefetchFar(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
              Vc::Common::prefetchFar(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(RAWDataHeaderGPU));
            }
          }
        }
      }
#endif
      const auto& x = TPCClusterizerDecodeZSCount(iSlice, fragmentMax);
      nDigitsFragment[iSlice] = x.second;
      processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits = x.first;
      mRec->MemoryScalers()->nTPCdigits += x.first;
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      processors()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsFragment[iSlice]);
      if (mRec->IsGPU()) {
        processorsShadow()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsFragment[iSlice]);
      }
      if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
        mPipelineNotifyCtx->rec->AllocateRegisteredForeignMemory(processors()->tpcClusterer[iSlice].mZSOffsetId, mRec);
        mPipelineNotifyCtx->rec->AllocateRegisteredForeignMemory(processors()->tpcClusterer[iSlice].mZSId, mRec);
      } else {
        AllocateRegisteredMemory(processors()->tpcClusterer[iSlice].mZSOffsetId);
        AllocateRegisteredMemory(processors()->tpcClusterer[iSlice].mZSId);
      }
    }
  } else {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      unsigned int nDigits = mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice];
      mRec->MemoryScalers()->nTPCdigits += nDigits;
      processors()->tpcClusterer[iSlice].SetNMaxDigits(nDigits, mCFContext->nPagesFragmentMax, nDigits);
    }
  }
  if (mIOPtrs.tpcZS) {
    GPUInfo("Event has %u 8kb TPC ZS pages, %lld digits", mCFContext->nPagesTotal, (long long int)mRec->MemoryScalers()->nTPCdigits);
  } else {
    GPUInfo("Event has %lld TPC Digits", (long long int)mRec->MemoryScalers()->nTPCdigits);
  }
  mCFContext->fragmentFirst = CfFragment{std::max<int>(mCFContext->tpcMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN), TPC_MAX_FRAGMENT_LEN};
  for (int iSlice = 0; iSlice < GetProcessingSettings().nTPCClustererLanes && iSlice < NSLICES; iSlice++) {
    if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
      mCFContext->nextPos[iSlice] = RunTPCClusterizer_transferZS(iSlice, mCFContext->fragmentFirst, GetProcessingSettings().nTPCClustererLanes + iSlice);
    }
  }

  if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      mCFContext->ptrSave[iSlice].zsOffsetHost = processors()->tpcClusterer[iSlice].mPzsOffsets;
      mCFContext->ptrSave[iSlice].zsOffsetDevice = processorsShadow()->tpcClusterer[iSlice].mPzsOffsets;
      mCFContext->ptrSave[iSlice].zsDevice = processorsShadow()->tpcClusterer[iSlice].mPzs;
    }
  }
  return 0;
}
#endif

int GPUChainTracking::RunTPCClusterizer(bool synchronizeOutput)
{
  if (param().rec.fwdTPCDigitsAsClusters) {
    return ForwardTPCDigits();
  }
#ifdef GPUCA_TPC_GEOMETRY_O2
  mRec->PushNonPersistentMemory();
  const auto& threadContext = GetThreadContext();
  bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  if (RunTPCClusterizer_prepare(mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer)) {
    return 1;
  }
  if (GetProcessingSettings().ompAutoNThreads && !mRec->IsGPU()) {
    mRec->SetNOMPThreads(mRec->MemoryScalers()->nTPCdigits / 20000);
  }

  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    processors()->tpcClusterer[iSlice].SetMaxData(mIOPtrs); // First iteration to set data sizes
  }
  mRec->ComputeReuseMax(nullptr); // Resolve maximums for shared buffers
  for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
    SetupGPUProcessor(&processors()->tpcClusterer[iSlice], true); // Now we allocate
  }
  if (mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer) {
    RunTPCClusterizer_prepare(true); // Restore some pointers, allocated by the other pipeline, and set to 0 by SetupGPUProcessor (since not allocated in this pipeline)
  }

  if (doGPU && mIOPtrs.tpcZS) {
    processorsShadow()->ioPtrs.tpcZS = mInputsShadow->mPzsMeta;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), mRec->NStreams() - 1);
  }
  if (doGPU) {
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)processors()->tpcClusterer - (char*)processors(), processorsShadow()->tpcClusterer, sizeof(GPUTPCClusterFinder) * NSLICES, mRec->NStreams() - 1, &mEvents->init);
  }

  size_t nClsTotal = 0;
  ClusterNativeAccess* tmpNative = mClusterNativeAccess.get();

  // setup MC Labels
  bool propagateMCLabels = GetProcessingSettings().runMC && processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC != nullptr;

  auto* digitsMC = propagateMCLabels ? processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC : nullptr;

  if (param().par.continuousMaxTimeBin > 0 && mCFContext->tpcMaxTimeBin >= (unsigned int)std::max(param().par.continuousMaxTimeBin + 1, TPC_MAX_FRAGMENT_LEN)) {
    GPUError("Input data has invalid time bin\n");
    return 1;
  }
  bool buildNativeGPU = (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression);
  bool buildNativeHost = mRec->GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters; // TODO: Should do this also when clusters are needed for later steps on the host but not requested as output
  mRec->MemoryScalers()->nTPCHits = mRec->MemoryScalers()->NTPCClusters(mRec->MemoryScalers()->nTPCdigits);
  mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mRec->MemoryScalers()->nTPCHits;
  if (buildNativeGPU) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
  }
  if (buildNativeHost && !(buildNativeGPU && GetProcessingSettings().delayedOutput)) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
  }

  GPUTPCLinearLabels mcLinearLabels;
  if (propagateMCLabels) {
    // No need to overallocate here, nTPCHits is anyway an upper bound used for the GPU cluster buffer, and we can always enlarge the buffer anyway
    mcLinearLabels.header.reserve(mRec->MemoryScalers()->nTPCHits / 2);
    mcLinearLabels.data.reserve(mRec->MemoryScalers()->nTPCHits);
  }

  char transferRunning[NSLICES] = {0};
  unsigned int outputQueueStart = mOutputQueue.size();

  for (unsigned int iSliceBase = 0; iSliceBase < NSLICES; iSliceBase += GetProcessingSettings().nTPCClustererLanes) {
    std::vector<bool> laneHasData(GetProcessingSettings().nTPCClustererLanes, false);
    for (CfFragment fragment = mCFContext->fragmentFirst; !fragment.isEnd(); fragment = fragment.next()) {
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Processing time bins [%d, %d) for sectors %d to %d", fragment.start, fragment.last(), iSliceBase, iSliceBase + GetProcessingSettings().nTPCClustererLanes - 1);
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        if (fragment.index != 0) {
          SynchronizeStream(lane); // Don't overwrite charge map from previous iteration until cluster computation is finished
        }

        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        clusterer.mPmemory->counters.nPeaks = clusterer.mPmemory->counters.nClusters = 0;
        clusterer.mPmemory->fragment = fragment;

        if (propagateMCLabels && fragment.index == 0) {
          clusterer.PrepareMC();
          clusterer.mPinputLabels = digitsMC->v[iSlice];
          // TODO: Why is the number of header entries in truth container
          // sometimes larger than the number of digits?
          assert(clusterer.mPinputLabels->getIndexedSize() >= mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice]);
        }

        if (mIOPtrs.tpcPackedDigits) {
          bool setDigitsOnGPU = doGPU && not mIOPtrs.tpcZS;
          bool setDigitsOnHost = (not doGPU && not mIOPtrs.tpcZS) || propagateMCLabels;
          auto* inDigits = mIOPtrs.tpcPackedDigits;
          size_t numDigits = inDigits->nTPCDigits[iSlice];
          if (setDigitsOnGPU) {
            GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPdigits, inDigits->tpcDigits[iSlice], sizeof(clustererShadow.mPdigits[0]) * numDigits, lane, true);
            clusterer.mPmemory->counters.nDigits = numDigits;
          } else if (setDigitsOnHost) {
            clusterer.mPdigits = const_cast<o2::tpc::Digit*>(inDigits->tpcDigits[iSlice]); // TODO: Needs fixing (invalid const cast)
            clusterer.mPmemory->counters.nDigits = numDigits;
          }
        }

        if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
          clusterer.mPmemory->counters.nPositions = mCFContext->nextPos[iSlice].first;
          clusterer.mPmemory->counters.nPagesSubslice = mCFContext->nextPos[iSlice].second;
        }
        TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);

        using ChargeMapType = decltype(*clustererShadow.mPchargeMap);
        using PeakMapType = decltype(*clustererShadow.mPpeakMap);
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPchargeMap, TPCMapMemoryLayout<ChargeMapType>::items() * sizeof(ChargeMapType));
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpeakMap, TPCMapMemoryLayout<PeakMapType>::items() * sizeof(PeakMapType));
        if (fragment.index == 0) {
          using HasLostBaselineType = decltype(*clustererShadow.mPpadHasLostBaseline);
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpadHasLostBaseline, TPC_PADS_IN_SECTOR * sizeof(HasLostBaselineType));
        }
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Zeroed Charges");

        if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceZS, lane);
          SynchronizeStream(GetProcessingSettings().nTPCClustererLanes + lane);
        }

        SynchronizeStream(mRec->NStreams() - 1); // Wait for copying to constant memory

        if (mIOPtrs.tpcZS && !mCFContext->nPagesSector[iSlice]) {
          continue;
        }

        if (not mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        } else if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }

        if (mIOPtrs.tpcZS) {
          int firstHBF = o2::raw::RDHUtils::getHeartBeatOrbit(*(const RAWDataHeaderGPU*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0]);
          runKernel<GPUTPCCFDecodeZS, GPUTPCCFDecodeZS::decodeZS>(GetGridBlk(doGPU ? clusterer.mPmemory->counters.nPagesSubslice : GPUTrackingInOutZS::NENDPOINTS, lane), {iSlice}, {}, firstHBF);
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        SynchronizeStream(lane);
        if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
          CfFragment f = fragment.next();
          int nextSlice = iSlice;
          if (f.isEnd()) {
            nextSlice += GetProcessingSettings().nTPCClustererLanes;
            f = mCFContext->fragmentFirst;
          }
          if (nextSlice < NSLICES && mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice]) {
            mCFContext->nextPos[nextSlice] = RunTPCClusterizer_transferZS(nextSlice, f, GetProcessingSettings().nTPCClustererLanes + lane);
          }
        }
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        if (propagateMCLabels || not mIOPtrs.tpcZS) {
          if (clusterer.mPmemory->counters.nPositions == 0) {
            continue;
          }
        }
        if (!mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillFromDigits>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        }
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpDigits, *mDebugFile)) {
          clusterer.DumpChargeMap(*mDebugFile, "Charges");
        }

        if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillIndexMap>(GetGrid(clusterer.mPmemory->counters.nDigitsInFragment, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {});
        }

        bool checkForNoisyPads = (rec()->GetParam().rec.maxTimeBinAboveThresholdIn1000Bin > 0) || (rec()->GetParam().rec.maxConsecTimeBinAboveThreshold > 0);
        checkForNoisyPads &= (rec()->GetParam().rec.noisyPadsQuickCheck ? fragment.index == 0 : true);

        if (checkForNoisyPads) {
          int nBlocks = TPC_PADS_IN_SECTOR / GPUTPCCFCheckPadBaseline::PadsPerCacheline;
          runKernel<GPUTPCCFCheckPadBaseline>(GetGridBlk(nBlocks, lane), {iSlice}, {});
        }

        runKernel<GPUTPCCFPeakFinder>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpPeaks, *mDebugFile);

        RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 0, doGPU, lane);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpPeaksCompacted, *mDebugFile);
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        SynchronizeStream(lane);
        if (clusterer.mPmemory->counters.nPeaks == 0) {
          continue;
        }
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::noiseSuppression>(GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSlice}, {});
        runKernel<GPUTPCCFNoiseSuppression, GPUTPCCFNoiseSuppression::updatePeaks>(GetGrid(clusterer.mPmemory->counters.nPeaks, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaks, *mDebugFile);

        RunTPCClusterizer_compactPeaks(clusterer, clustererShadow, 1, doGPU, lane);
        TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpSuppressedPeaksCompacted, *mDebugFile);
      }
      for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        SynchronizeStream(lane);

        if (fragment.index == 0) {
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {nullptr, transferRunning[lane] == 1 ? &mEvents->stream[lane] : nullptr}, clustererShadow.mPclusterInRow, GPUCA_ROW_COUNT * sizeof(*clustererShadow.mPclusterInRow));
          transferRunning[lane] = 2;
        }

        if (clusterer.mPmemory->counters.nClusters == 0) {
          continue;
        }

        runKernel<GPUTPCCFDeconvolution>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Split Charges");

        runKernel<GPUTPCCFClusterizer>(GetGrid(clusterer.mPmemory->counters.nClusters, lane), {iSlice}, {}, 0);
        if (doGPU && propagateMCLabels) {
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mScratchId, lane);
          SynchronizeStream(lane);
          runKernel<GPUTPCCFClusterizer>(GetGrid(clusterer.mPmemory->counters.nClusters, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, 1);
        }
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Lane %d: Found clusters: digits %u peaks %u clusters %u", lane, (int)clusterer.mPmemory->counters.nPositions, (int)clusterer.mPmemory->counters.nPeaks, (int)clusterer.mPmemory->counters.nClusters);
        }

        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, lane);
        laneHasData[lane] = true;
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpCountedPeaks, *mDebugFile)) {
          clusterer.DumpClusters(*mDebugFile);
        }
      }
    }
    size_t nClsFirst = nClsTotal;
    bool anyLaneHasData = false;
    for (int lane = 0; lane < GetProcessingSettings().nTPCClustererLanes && iSliceBase + lane < NSLICES; lane++) {
      unsigned int iSlice = iSliceBase + lane;
      std::fill(&tmpNative->nClusters[iSlice][0], &tmpNative->nClusters[iSlice][0] + MAXGLOBALPADROW, 0);
      SynchronizeStream(lane);
      GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
      GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
      if (laneHasData[lane]) {
        anyLaneHasData = true;
        if (buildNativeGPU && GetProcessingSettings().tpccfGatherKernel) {
          runKernel<GPUTPCCFGather>(GetGridBlk(GPUCA_ROW_COUNT, mRec->NStreams() - 1), {iSlice}, {}, &mInputsShadow->mPclusterNativeBuffer[nClsTotal]);
        }
        for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
          if (buildNativeGPU) {
            if (!GetProcessingSettings().tpccfGatherKernel) {
              GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&mInputsShadow->mPclusterNativeBuffer[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, -2);
            }
          } else if (buildNativeHost) {
            GPUMemCpyAlways(RecoStep::TPCClusterFinding, (void*)&mInputsHost->mPclusterNativeOutput[nClsTotal], (const void*)&clustererShadow.mPclusterByRow[j * clusterer.mNMaxClusterPerRow], sizeof(mIOPtrs.clustersNative->clustersLinear[0]) * clusterer.mPclusterInRow[j], mRec->NStreams() - 1, false);
          }
          tmpNative->nClusters[iSlice][j] += clusterer.mPclusterInRow[j];
          nClsTotal += clusterer.mPclusterInRow[j];
        }
        if (transferRunning[lane]) {
          ReleaseEvent(&mEvents->stream[lane]);
        }
        RecordMarker(&mEvents->stream[lane], mRec->NStreams() - 1);
        transferRunning[lane] = 1;
      }

      if (not propagateMCLabels || clusterer.mPmemory->counters.nClusters == 0) {
        continue;
      }

      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::setRowOffsets>(GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {});
      GPUTPCCFMCLabelFlattener::setGlobalOffsetsAndAllocate(clusterer, mcLinearLabels);
      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::flatten>(GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, &mcLinearLabels);
      clusterer.clearMCMemory();
    }
    if (buildNativeHost && buildNativeGPU && anyLaneHasData) {
      if (GetProcessingSettings().delayedOutput) {
        mOutputQueue.emplace_back(outputQueueEntry{(void*)((char*)&mInputsHost->mPclusterNativeOutput[nClsFirst] - (char*)&mInputsHost->mPclusterNativeOutput[0]), &mInputsShadow->mPclusterNativeBuffer[nClsFirst], (nClsTotal - nClsFirst) * sizeof(mInputsHost->mPclusterNativeOutput[nClsFirst]), RecoStep::TPCClusterFinding});
      } else {
        GPUMemCpy(RecoStep::TPCClusterFinding, (void*)&mInputsHost->mPclusterNativeOutput[nClsFirst], (void*)&mInputsShadow->mPclusterNativeBuffer[nClsFirst], (nClsTotal - nClsFirst) * sizeof(mInputsHost->mPclusterNativeOutput[nClsFirst]), mRec->NStreams() - 1, false);
      }
    }
  }
  for (int i = 0; i < GetProcessingSettings().nTPCClustererLanes; i++) {
    if (transferRunning[i]) {
      ReleaseEvent(&mEvents->stream[i]);
    }
  }

  ClusterNativeAccess::ConstMCLabelContainerView* mcLabelsConstView = nullptr;
  if (propagateMCLabels) {
    // TODO: write to buffer directly
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcLabels;
    std::pair<ConstMCLabelContainer*, ConstMCLabelContainerView*> buffer;
    if (mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)] && mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)]->useExternal()) {
      if (!mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)]->allocator) {
        throw std::runtime_error("Cluster MC Label buffer missing");
      }
      ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer* container = reinterpret_cast<ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer*>(mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clusterLabels)]->allocator(0));
      buffer = {&container->first, &container->second};
    } else {
      mIOMem.clusterNativeMCView = std::make_unique<ConstMCLabelContainerView>();
      mIOMem.clusterNativeMCBuffer = std::make_unique<ConstMCLabelContainer>();
      buffer.first = mIOMem.clusterNativeMCBuffer.get();
      buffer.second = mIOMem.clusterNativeMCView.get();
    }

    assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);
    assert(propagateMCLabels ? mcLinearLabels.data.size() >= nClsTotal : true);

    mcLabels.setFrom(mcLinearLabels.header, mcLinearLabels.data);
    mcLabels.flatten_to(*buffer.first);
    *buffer.second = *buffer.first;
    mcLabelsConstView = buffer.second;
  }

  if (buildNativeHost && buildNativeGPU && GetProcessingSettings().delayedOutput) {
    mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = nClsTotal;
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeOutput, mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::clustersNative)]);
    for (unsigned int i = outputQueueStart; i < mOutputQueue.size(); i++) {
      mOutputQueue[i].dst = (char*)mInputsHost->mPclusterNativeOutput + (size_t)mOutputQueue[i].dst;
    }
  }

  if (buildNativeHost) {
    tmpNative->clustersLinear = mInputsHost->mPclusterNativeOutput;
    tmpNative->clustersMCTruth = mcLabelsConstView;
    tmpNative->setOffsetPtrs();
    mIOPtrs.clustersNative = tmpNative;
  }

  if (mPipelineNotifyCtx) {
    SynchronizeStream(mRec->NStreams() - 2); // Must finish before updating ioPtrs in (global) constant memory
    std::lock_guard<std::mutex> lock(mPipelineNotifyCtx->mutex);
    mPipelineNotifyCtx->ready = true;
    mPipelineNotifyCtx->cond.notify_one();
  }

  if (buildNativeGPU) {
    processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
    mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
    mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
    TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceClusterNativeAccess, 0);
  }
  if (synchronizeOutput) {
    SynchronizeStream(mRec->NStreams() - 1);
  }
  if (buildNativeHost && GetProcessingSettings().debugLevel >= 4) {
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        std::sort(&mInputsHost->mPclusterNativeOutput[tmpNative->clusterOffset[i][j]], &mInputsHost->mPclusterNativeOutput[tmpNative->clusterOffset[i][j] + tmpNative->nClusters[i][j]]);
      }
    }
  }
  mRec->MemoryScalers()->nTPCHits = nClsTotal;
  mRec->PopNonPersistentMemory(RecoStep::TPCClusterFinding);
  if (mPipelineNotifyCtx) {
    mRec->UnblockStackedMemory();
    mPipelineNotifyCtx = nullptr;
  }

#endif
  return 0;
}
