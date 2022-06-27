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
#ifdef GPUCA_HAVE_O2HEADERS
#include "GPUHostDataTypes.h"
#include "GPUTPCCFChainContext.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/Constants.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#include "utils/strtag.h"

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
  unsigned int pages = 0;
  for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
    clusterer.mMinMaxCN[j] = mCFContext->fragmentData[fragment.index].minMaxCN[iSlice][j];
    if (doGPU) {
      unsigned short posInEndpoint = 0;
      unsigned short pagesEndpoint = 0;
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
      if (pagesEndpoint != mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j].size()) {
        GPUFatal("TPC raw page count mismatch in TPCClusterizerDecodeZSCountUpdate: expected %d / buffered %lu", pagesEndpoint, mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j].size());
      }
    } else {
      clusterer.mPzsOffsets[j] = GPUTPCClusterFinder::ZSOffset{digits, j, 0};
      digits += mCFContext->fragmentData[fragment.index].nDigits[iSlice][j];
      pages += mCFContext->fragmentData[fragment.index].nPages[iSlice][j];
    }
  }
  if (doGPU) {
    pages = o - processors()->tpcClusterer[iSlice].mPzsOffsets;
  }
  return {digits, pages};
}

std::pair<unsigned int, unsigned int> GPUChainTracking::TPCClusterizerDecodeZSCount(unsigned int iSlice, const CfFragment& fragment)
{
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Start();
  unsigned int nDigits = 0;
  unsigned int nPages = 0;
  bool doGPU = mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCClusterFinding;
  int firstHBF = (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasTfStartOrbit) ? mIOPtrs.settingsTF->tfStartOrbit : (mIOPtrs.tpcZS->slice[iSlice].count[0] && mIOPtrs.tpcZS->slice[iSlice].nZSPtr[0][0]) ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0]) : 0;

  for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
#ifndef GPUCA_NO_VC
    if (GetProcessingSettings().prefetchTPCpageScan >= 3 && j < GPUTrackingInOutZS::NENDPOINTS - 1) {
      for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j + 1]; k++) {
        for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j + 1][k]; l++) {
          Vc::Common::prefetchMid(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchMid(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j + 1][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
        }
      }
    }
#endif

    std::vector<std::pair<CfFragment, std::array<int, 5>>> fragments;
    fragments.reserve(mCFContext->nFragments);
    fragments.emplace_back(std::pair<CfFragment, std::array<int, 5>>{fragment, {0, 0, 0, 0, 0}});
    for (unsigned int i = 1; i < mCFContext->nFragments; i++) {
      fragments.emplace_back(std::pair<CfFragment, std::array<int, 5>>{fragments.back().first.next(), {0, 0, 0, 0, 0}});
    }

    unsigned int emptyPages = 0;
    unsigned int firstPossibleFragment = 0;
    for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
      nPages += mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
      for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
#ifndef GPUCA_NO_VC
        if (GetProcessingSettings().prefetchTPCpageScan >= 2 && l + 1 < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]) {
          Vc::Common::prefetchForOneRead(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          Vc::Common::prefetchForOneRead(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + (l + 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
        }
#endif
        const unsigned char* const page = ((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)page;
        if (o2::raw::RDHUtils::getMemorySize(*rdh) == sizeof(o2::header::RAWDataHeader)) {
          emptyPages++;
          continue;
        }
        const TPCZSHDR* const hdr = (const TPCZSHDR*)(page + sizeof(o2::header::RAWDataHeader));
        if (mCFContext->zsVersion == -1) {
          mCFContext->zsVersion = hdr->version;
        } else if (mCFContext->zsVersion != (int)hdr->version) {
          GPUFatal("Received TPC ZS data of mixed versions");
        }
        nDigits += hdr->nADCsamples;
        unsigned int timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
        if (timeBin + hdr->nTimeBins > mCFContext->tpcMaxTimeBin) {
          mCFContext->tpcMaxTimeBin = timeBin + hdr->nTimeBins;
        }
        for (unsigned int f = firstPossibleFragment; f < mCFContext->nFragments; f++) {
          if (timeBin < (unsigned int)fragments[f].first.last()) {
            if ((unsigned int)fragments[f].first.first() <= timeBin + hdr->nTimeBins) {
              fragments[f].second[2] = k + 1;
              fragments[f].second[3] = l + 1;
              if (!fragments[f].second[4]) {
                fragments[f].second[4] = 1;
                fragments[f].second[0] = k;
                fragments[f].second[1] = l;
              } else if (emptyPages) {
                mCFContext->fragmentData[f].nPages[iSlice][j] += emptyPages;
                for (unsigned int m = 0; m < emptyPages; m++) {
                  mCFContext->fragmentData[f].pageDigits[iSlice][j].emplace_back(0);
                }
              }
              mCFContext->fragmentData[f].nPages[iSlice][j]++;
              mCFContext->fragmentData[f].nDigits[iSlice][j] += hdr->nADCsamples;
              if (doGPU) {
                mCFContext->fragmentData[f].pageDigits[iSlice][j].emplace_back(hdr->nADCsamples);
              }
            }
          } else {
            firstPossibleFragment = f + 1;
          }
        }
        emptyPages = 0;
      }
    }
    for (unsigned int f = 0; f < mCFContext->nFragments; f++) {
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].maxC = fragments[f].second[2];
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].minC = fragments[f].second[0];
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].maxN = fragments[f].second[3];
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].minN = fragments[f].second[1];
    }
  }
  mCFContext->nPagesTotal += nPages;
  mCFContext->nPagesSector[iSlice] = nPages;
  mCFContext->nPagesSectorMax = std::max(mCFContext->nPagesSectorMax, nPages);

  unsigned int nDigitsFragmentMax = 0;
  for (unsigned int i = 0; i < mCFContext->nFragments; i++) {
    unsigned int pages = 0;
    unsigned int digits = 0;
    for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      pages += mCFContext->fragmentData[i].nPages[iSlice][j];
      digits += mCFContext->fragmentData[i].nDigits[iSlice][j];
    }
    mCFContext->nPagesFragmentMax = std::max(mCFContext->nPagesSectorMax, pages);
    nDigitsFragmentMax = std::max(nDigitsFragmentMax, digits);
  }
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Stop();
  return {nDigits, nDigitsFragmentMax};
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
          char* ptrLast = (char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k] + (max - 1) * TPCZSHDR::TPC_ZS_PAGE_SIZE;
          size_t size = (ptrLast - src) + o2::raw::RDHUtils::getMemorySize(*(const o2::header::RAWDataHeader*)ptrLast);
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
  const bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  const short maxFragmentLen = doGPU ? TPC_MAX_FRAGMENT_LEN_GPU : TPC_MAX_FRAGMENT_LEN_HOST;
  mCFContext->tpcMaxTimeBin = param().par.continuousTracking ? std::max<int>(param().par.continuousMaxTimeBin, maxFragmentLen) : TPC_MAX_TIME_BIN_TRIGGERED;
  const CfFragment fragmentMax{(tpccf::TPCTime)mCFContext->tpcMaxTimeBin + 1, maxFragmentLen};
  mCFContext->prepare(mIOPtrs.tpcZS, fragmentMax);
  if (mIOPtrs.tpcZS) {
    unsigned int nDigitsFragmentMax[NSLICES];
    mCFContext->zsVersion = -1;
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (mIOPtrs.tpcZS->slice[iSlice].count[0]) {
        const void* rdh = mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0];
        if (rdh && o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>() != o2::raw::RDHUtils::getVersion(rdh)) {
          GPUError("Data has invalid RDH version %d, %d required\n", o2::raw::RDHUtils::getVersion(rdh), o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>());
          return 1;
        }
      }
#ifndef GPUCA_NO_VC
      if (GetProcessingSettings().prefetchTPCpageScan >= 1 && iSlice < NSLICES - 1) {
        for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
          for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
            for (unsigned int l = 0; l < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k]; l++) {
              Vc::Common::prefetchFar(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE);
              Vc::Common::prefetchFar(((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice + 1].zsPtr[j][k]) + l * TPCZSHDR::TPC_ZS_PAGE_SIZE + sizeof(o2::header::RAWDataHeader));
            }
          }
        }
      }
#endif
      const auto& x = TPCClusterizerDecodeZSCount(iSlice, fragmentMax);
      nDigitsFragmentMax[iSlice] = x.first;
      processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits = x.first;
      mRec->MemoryScalers()->nTPCdigits += x.first;
    }
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      unsigned int nDigitsBase = nDigitsFragmentMax[iSlice];
      unsigned int threshold = 40000000;
      unsigned int nDigitsScaled = nDigitsBase > threshold ? nDigitsBase : std::min((threshold + nDigitsBase) / 2, 2 * nDigitsBase);
      processors()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsScaled);
      if (mRec->IsGPU()) {
        processorsShadow()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsScaled);
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
    GPUInfo("Event has %u 8kb TPC ZS pages (version %d), %lld digits", mCFContext->nPagesTotal, mCFContext->zsVersion, (long long int)mRec->MemoryScalers()->nTPCdigits);
  } else {
    GPUInfo("Event has %lld TPC Digits", (long long int)mRec->MemoryScalers()->nTPCdigits);
  }
  mCFContext->fragmentFirst = CfFragment{std::max<int>(mCFContext->tpcMaxTimeBin + 1, maxFragmentLen), maxFragmentLen};
  for (int iSlice = 0; iSlice < GetProcessingSettings().nTPCClustererLanes && iSlice < NSLICES; iSlice++) {
    if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice] && mCFContext->zsVersion != -1) {
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

// TODO: Clusterizer not working with OCL1 (Clusterizer on CPU, Tracking on GPU)
int GPUChainTracking::RunTPCClusterizer(bool synchronizeOutput)
{
  if (param().rec.fwdTPCDigitsAsClusters) {
    return ForwardTPCDigits();
  }
#ifdef GPUCA_TPC_GEOMETRY_O2
  mRec->PushNonPersistentMemory(qStr2Tag("TPCCLUST"));
  const auto& threadContext = GetThreadContext();
  const bool doGPU = GetRecoStepsGPU() & RecoStep::TPCClusterFinding;
  const short maxFragmentLen = TPC_MAX_FRAGMENT_LEN_PADDED(doGPU ? TPC_MAX_FRAGMENT_LEN_GPU : TPC_MAX_FRAGMENT_LEN_HOST);
  if (RunTPCClusterizer_prepare(mPipelineNotifyCtx && GetProcessingSettings().doublePipelineClusterizer)) {
    return 1;
  }
  if (GetProcessingSettings().ompAutoNThreads && !mRec->IsGPU()) {
    mRec->SetNOMPThreads(mRec->MemoryScalers()->nTPCdigits / 20000);
  }

  mRec->MemoryScalers()->nTPCHits = mRec->MemoryScalers()->NTPCClusters(mRec->MemoryScalers()->nTPCdigits);
  float tpcHitLowOccupancyScalingFactor = 1.f;
  if (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasNHBFPerTF) {
    unsigned int nHitsBase = mRec->MemoryScalers()->nTPCHits;
    unsigned int threshold = 30000000 / 256 * mIOPtrs.settingsTF->nHBFPerTF;
    mRec->MemoryScalers()->nTPCHits = std::max<unsigned int>(nHitsBase, std::min<unsigned int>(threshold, nHitsBase * 3.5)); // Increase the buffer size for low occupancy data to compensate for noisy pads creating exceiive clusters
    if (nHitsBase < threshold) {
      float maxFactor = mRec->MemoryScalers()->nTPCHits < threshold * 2 / 3 ? 3 : (mRec->MemoryScalers()->nTPCHits < threshold ? 2.25 : 1.75);
      mRec->MemoryScalers()->temporaryFactor *= std::min(maxFactor, (float)threshold / nHitsBase);
      tpcHitLowOccupancyScalingFactor = std::min(3.5f, (float)threshold / nHitsBase);
    }
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
  bool propagateMCLabels = GetProcessingSettings().runMC && processors()->ioPtrs.tpcPackedDigits && processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC;

  auto* digitsMC = propagateMCLabels ? processors()->ioPtrs.tpcPackedDigits->tpcDigitsMC : nullptr;

  if (param().par.continuousMaxTimeBin > 0 && mCFContext->tpcMaxTimeBin >= std::max<unsigned int>(param().par.continuousMaxTimeBin + 1, maxFragmentLen)) {
    GPUError("Input data has invalid time bin %u > %d\n", mCFContext->tpcMaxTimeBin, std::max<unsigned int>(param().par.continuousMaxTimeBin + 1, (int)maxFragmentLen));
    return 1;
  }
  bool buildNativeGPU = (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression);
  bool buildNativeHost = mRec->GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters; // TODO: Should do this also when clusters are needed for later steps on the host but not requested as output

  mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mRec->MemoryScalers()->nTPCHits * tpcHitLowOccupancyScalingFactor;
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
    const int maxLane = std::min<int>(GetProcessingSettings().nTPCClustererLanes, NSLICES - iSliceBase);
    for (CfFragment fragment = mCFContext->fragmentFirst; !fragment.isEnd(); fragment = fragment.next()) {
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Processing time bins [%d, %d) for sectors %d to %d", fragment.start, fragment.last(), iSliceBase, iSliceBase + GetProcessingSettings().nTPCClustererLanes - 1);
      }
      GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, GetProcessingSettings().nTPCClustererLanes)))
      for (int lane = 0; lane < maxLane; lane++) {
        if (fragment.index != 0) {
          SynchronizeStream(lane); // Don't overwrite charge map from previous iteration until cluster computation is finished
        }

        unsigned int iSlice = iSliceBase + lane;
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        clusterer.mPmemory->counters.nPeaks = clusterer.mPmemory->counters.nClusters = 0;
        clusterer.mPmemory->fragment = fragment;

        if (mIOPtrs.tpcPackedDigits) {
          bool setDigitsOnGPU = doGPU && not mIOPtrs.tpcZS;
          bool setDigitsOnHost = (not doGPU && not mIOPtrs.tpcZS) || propagateMCLabels;
          auto* inDigits = mIOPtrs.tpcPackedDigits;
          size_t numDigits = inDigits->nTPCDigits[iSlice];
          if (setDigitsOnGPU) {
            GPUMemCpy(RecoStep::TPCClusterFinding, clustererShadow.mPdigits, inDigits->tpcDigits[iSlice], sizeof(clustererShadow.mPdigits[0]) * numDigits, lane, true);
          }
          if (setDigitsOnHost) {
            clusterer.mPdigits = const_cast<o2::tpc::Digit*>(inDigits->tpcDigits[iSlice]); // TODO: Needs fixing (invalid const cast)
          }
          clusterer.mPmemory->counters.nDigits = numDigits;
        }

        if (mIOPtrs.tpcZS) {
          if (mCFContext->nPagesSector[iSlice] && mCFContext->zsVersion != -1) {
            clusterer.mPmemory->counters.nPositions = mCFContext->nextPos[iSlice].first;
            clusterer.mPmemory->counters.nPagesSubslice = mCFContext->nextPos[iSlice].second;
          } else {
            clusterer.mPmemory->counters.nPositions = clusterer.mPmemory->counters.nPagesSubslice = 0;
          }
        }
        TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);

        using ChargeMapType = decltype(*clustererShadow.mPchargeMap);
        using PeakMapType = decltype(*clustererShadow.mPpeakMap);
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPchargeMap, TPCMapMemoryLayout<ChargeMapType>::items(doGPU) * sizeof(ChargeMapType));
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpeakMap, TPCMapMemoryLayout<PeakMapType>::items(doGPU) * sizeof(PeakMapType));
        if (fragment.index == 0) {
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpadIsNoisy, TPC_PADS_IN_SECTOR * sizeof(*clustererShadow.mPpadIsNoisy));
        }
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Zeroed Charges", doGPU);

        if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice] && mCFContext->zsVersion != -1) {
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceZS, lane);
          SynchronizeStream(GetProcessingSettings().nTPCClustererLanes + lane);
        }

        SynchronizeStream(mRec->NStreams() - 1); // Wait for copying to constant memory

        if (mIOPtrs.tpcZS && (!mCFContext->nPagesSector[iSlice] || mCFContext->zsVersion == -1)) {
          continue;
        }
        if (!mIOPtrs.tpcZS && mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice] == 0) {
          clusterer.mPmemory->counters.nPositions = 0;
          continue;
        }

        if (propagateMCLabels && fragment.index == 0) {
          clusterer.PrepareMC();
          clusterer.mPinputLabels = digitsMC->v[iSlice];
          if (clusterer.mPinputLabels == nullptr) {
            GPUFatal("MC label container missing, sector %d", iSlice);
          }
          if (clusterer.mPinputLabels->getIndexedSize() != mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice]) {
            GPUFatal("MC label container has incorrect number of entries: %d expected, has %d\n", (int)mIOPtrs.tpcPackedDigits->nTPCDigits[iSlice], (int)clusterer.mPinputLabels->getIndexedSize());
          }
        }

        if (not mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        } else if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
          TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }

        if (mIOPtrs.tpcZS) {
          int firstHBF = (mIOPtrs.settingsTF && mIOPtrs.settingsTF->hasTfStartOrbit) ? mIOPtrs.settingsTF->tfStartOrbit : (mIOPtrs.tpcZS->slice[iSlice].count[0] && mIOPtrs.tpcZS->slice[iSlice].nZSPtr[0][0]) ? o2::raw::RDHUtils::getHeartBeatOrbit(*(const o2::header::RAWDataHeader*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0])
                                                                                                                                                                                                               : 0;
          unsigned int nBlocks = doGPU ? clusterer.mPmemory->counters.nPagesSubslice : GPUTrackingInOutZS::NENDPOINTS;

          switch (mCFContext->zsVersion) {
            default:
              GPUFatal("Data with invalid TPC ZS mode (%d) received", mCFContext->zsVersion);
              break;
            case ZSVersionRowBased10BitADC:
            case ZSVersionRowBased12BitADC:
              runKernel<GPUTPCCFDecodeZS>(GetGridBlk(nBlocks, lane), {iSlice}, {}, firstHBF);
              break;
            case ZSVersionLinkBasedWithMeta:
              runKernel<GPUTPCCFDecodeZSLink>(GetGridBlk(nBlocks, lane), {iSlice}, {}, firstHBF);
              break;
          }
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }
      }
      GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, GetProcessingSettings().nTPCClustererLanes)))
      for (int lane = 0; lane < maxLane; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        SynchronizeStream(lane);
        if (mIOPtrs.tpcZS) {
          CfFragment f = fragment.next();
          int nextSlice = iSlice;
          if (f.isEnd()) {
            nextSlice += GetProcessingSettings().nTPCClustererLanes;
            f = mCFContext->fragmentFirst;
          }
          if (nextSlice < NSLICES && mIOPtrs.tpcZS && mCFContext->nPagesSector[nextSlice] && mCFContext->zsVersion != -1) {
            mCFContext->nextPos[nextSlice] = RunTPCClusterizer_transferZS(nextSlice, f, GetProcessingSettings().nTPCClustererLanes + lane);
          }
        }
        GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
        GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
        if (clusterer.mPmemory->counters.nPositions == 0) {
          continue;
        }
        if (!mIOPtrs.tpcZS) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillFromDigits>(GetGrid(clusterer.mPmemory->counters.nPositions, lane), {iSlice}, {});
        }
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpDigits, *mDebugFile)) {
          clusterer.DumpChargeMap(*mDebugFile, "Charges", doGPU);
        }

        if (propagateMCLabels) {
          runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::fillIndexMap>(GetGrid(clusterer.mPmemory->counters.nDigitsInFragment, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {});
        }

        bool checkForNoisyPads = (rec()->GetParam().rec.tpc.maxTimeBinAboveThresholdIn1000Bin > 0) || (rec()->GetParam().rec.tpc.maxConsecTimeBinAboveThreshold > 0);
        checkForNoisyPads &= (rec()->GetParam().rec.tpc.noisyPadsQuickCheck ? fragment.index == 0 : true);
        checkForNoisyPads &= !GetProcessingSettings().disableTPCNoisyPadFilter;

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
      GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, GetProcessingSettings().nTPCClustererLanes)))
      for (int lane = 0; lane < maxLane; lane++) {
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
      GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, GetProcessingSettings().nTPCClustererLanes)))
      for (int lane = 0; lane < maxLane; lane++) {
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
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Split Charges", doGPU);

        runKernel<GPUTPCCFClusterizer>(GetGrid(clusterer.mPmemory->counters.nClusters, lane), {iSlice}, {}, 0);
        if (doGPU && propagateMCLabels) {
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mScratchId, lane);
          SynchronizeStream(lane);
          runKernel<GPUTPCCFClusterizer>(GetGrid(clusterer.mPmemory->counters.nClusters, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, 1);
        }
        if (GetProcessingSettings().debugLevel >= 3) {
          GPUInfo("Sector %02d Fragment %02d Lane %d: Found clusters: digits %u peaks %u clusters %u", iSlice, fragment.index, lane, (int)clusterer.mPmemory->counters.nPositions, (int)clusterer.mPmemory->counters.nPeaks, (int)clusterer.mPmemory->counters.nClusters);
        }

        TransferMemoryResourcesToHost(RecoStep::TPCClusterFinding, &clusterer, lane);
        laneHasData[lane] = true;
        if (DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpCountedPeaks, *mDebugFile)) {
          clusterer.DumpClusters(*mDebugFile);
        }
      }
      mRec->SetNestedLoopOmpFactor(1);
    }

    size_t nClsFirst = nClsTotal;
    bool anyLaneHasData = false;
    for (int lane = 0; lane < maxLane; lane++) {
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
          if (nClsTotal + clusterer.mPclusterInRow[j] > mInputsHost->mNClusterNative) {
            clusterer.raiseError(GPUErrors::ERROR_CF_GLOBAL_CLUSTER_OVERFLOW, iSlice * 1000 + j, nClsTotal + clusterer.mPclusterInRow[j], mInputsHost->mNClusterNative);
            continue;
          }
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
          ReleaseEvent(&mEvents->stream[lane], doGPU);
        }
        RecordMarker(&mEvents->stream[lane], mRec->NStreams() - 1);
        transferRunning[lane] = 1;
      }

      if (not propagateMCLabels || not laneHasData[lane]) {
        assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);
        continue;
      }

      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::setRowOffsets>(GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {});
      GPUTPCCFMCLabelFlattener::setGlobalOffsetsAndAllocate(clusterer, mcLinearLabels);
      runKernel<GPUTPCCFMCLabelFlattener, GPUTPCCFMCLabelFlattener::flatten>(GetGrid(GPUCA_ROW_COUNT, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, &mcLinearLabels);
      clusterer.clearMCMemory();
      assert(propagateMCLabels ? mcLinearLabels.header.size() == nClsTotal : true);
    }
    if (propagateMCLabels) {
      for (int lane = 0; lane < maxLane; lane++) {
        processors()->tpcClusterer[iSliceBase + lane].clearMCMemory();
      }
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
      ReleaseEvent(&mEvents->stream[i], doGPU);
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
  mRec->PopNonPersistentMemory(RecoStep::TPCClusterFinding, qStr2Tag("TPCCLUST"));
  if (mPipelineNotifyCtx) {
    mRec->UnblockStackedMemory();
    mPipelineNotifyCtx = nullptr;
  }

#endif
  return 0;
}
