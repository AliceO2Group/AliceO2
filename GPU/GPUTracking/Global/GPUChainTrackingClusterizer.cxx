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
#include "GPUTriggerOutputs.h"
#include "GPUHostDataTypes.h"
#include "GPUTPCCFChainContext.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCBase/RDHUtils.h"
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
      for (unsigned int k = clusterer.mMinMaxCN[j].zsPtrFirst; k < clusterer.mMinMaxCN[j].zsPtrLast; k++) {
        const unsigned int pageFirst = (k == clusterer.mMinMaxCN[j].zsPtrFirst) ? clusterer.mMinMaxCN[j].zsPageFirst : 0;
        const unsigned int pageLast = (k + 1 == clusterer.mMinMaxCN[j].zsPtrLast) ? clusterer.mMinMaxCN[j].zsPageLast : mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
        for (unsigned int l = pageFirst; l < pageLast; l++) {
          unsigned short pageDigits = mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j][posInEndpoint++];
          if (pageDigits) {
            *(o++) = GPUTPCClusterFinder::ZSOffset{digits, j, pagesEndpoint};
            digits += pageDigits;
          }
          pagesEndpoint++;
        }
      }
      if (pagesEndpoint != mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j].size()) {
        if (GetProcessingSettings().ignoreNonFatalGPUErrors) {
          GPUError("TPC raw page count mismatch in TPCClusterizerDecodeZSCountUpdate: expected %d / buffered %lu", pagesEndpoint, mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j].size());
          return {0, 0};
        } else {
          GPUFatal("TPC raw page count mismatch in TPCClusterizerDecodeZSCountUpdate: expected %d / buffered %lu", pagesEndpoint, mCFContext->fragmentData[fragment.index].pageDigits[iSlice][j].size());
        }
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
  if (!doGPU && GetProcessingSettings().debugLevel >= 4 && mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
    TPCClusterizerEnsureZSOffsets(iSlice, fragment);
  }
  return {digits, pages};
}

void GPUChainTracking::TPCClusterizerEnsureZSOffsets(unsigned int iSlice, const CfFragment& fragment)
{
  GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
  unsigned int nAdcs = 0;
  for (unsigned short endpoint = 0; endpoint < GPUTrackingInOutZS::NENDPOINTS; endpoint++) {
    const auto& data = mCFContext->fragmentData[fragment.index];
    unsigned int pagesEndpoint = 0;
    const unsigned int nAdcsExpected = data.nDigits[iSlice][endpoint];
    const unsigned int nPagesExpected = data.nPages[iSlice][endpoint];

    unsigned int nAdcDecoded = 0;
    const auto& zs = mIOPtrs.tpcZS->slice[iSlice];
    for (unsigned int i = data.minMaxCN[iSlice][endpoint].zsPtrFirst; i < data.minMaxCN[iSlice][endpoint].zsPtrLast; i++) {
      const unsigned int pageFirst = (i == data.minMaxCN[iSlice][endpoint].zsPtrFirst) ? data.minMaxCN[iSlice][endpoint].zsPageFirst : 0;
      const unsigned int pageLast = (i + 1 == data.minMaxCN[iSlice][endpoint].zsPtrLast) ? data.minMaxCN[iSlice][endpoint].zsPageLast : zs.nZSPtr[endpoint][i];
      for (unsigned int j = pageFirst; j < pageLast; j++) {
        const unsigned char* page = static_cast<const unsigned char*>(zs.zsPtr[endpoint][i]) + j * TPCZSHDR::TPC_ZS_PAGE_SIZE;
        const header::RAWDataHeader* rawDataHeader = reinterpret_cast<const header::RAWDataHeader*>(page);
        const TPCZSHDRV2* decHdr = reinterpret_cast<const TPCZSHDRV2*>(page + raw::RDHUtils::getMemorySize(*rawDataHeader) - sizeof(TPCZSHDRV2));
        const unsigned short nSamplesInPage = decHdr->nADCsamples;

        nAdcDecoded += nSamplesInPage;
        pagesEndpoint++;
      }
    }

    if (pagesEndpoint != nPagesExpected) {
      GPUFatal("Sector %d, Endpoint %d, Fragment %d: TPC raw page count mismatch: expected %d / buffered %lu", iSlice, endpoint, fragment.index, pagesEndpoint, nPagesExpected);
    }

    if (nAdcDecoded != nAdcsExpected) {
      GPUFatal("Sector %d, Endpoint %d, Fragment %d: TPC ADC count mismatch: expected %u, buffered %u", iSlice, endpoint, fragment.index, nAdcsExpected, nAdcDecoded);
    }

    if (nAdcs != clusterer.mPzsOffsets[endpoint].offset) {
      GPUFatal("Sector %d, Endpoint %d, Fragment %d: TPC ADC offset mismatch: expected %u, buffered %u", iSlice, endpoint, fragment.index, nAdcs, clusterer.mPzsOffsets[endpoint].offset);
    }

    nAdcs += nAdcsExpected;
  }
}

namespace
{
struct TPCCFDecodeScanTmp {
  int zsPtrFirst, zsPageFirst, zsPtrLast, zsPageLast, hasData, pageCounter;
};
} // namespace

std::pair<unsigned int, unsigned int> GPUChainTracking::TPCClusterizerDecodeZSCount(unsigned int iSlice, const CfFragment& fragment)
{
  mRec->getGeneralStepTimer(GeneralStep::Prepare).Start();
  unsigned int nDigits = 0;
  unsigned int nPages = 0;
  unsigned int endpointAdcSamples[GPUTrackingInOutZS::NENDPOINTS];
  memset(endpointAdcSamples, 0, sizeof(endpointAdcSamples));
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

    std::vector<std::pair<CfFragment, TPCCFDecodeScanTmp>> fragments;
    fragments.reserve(mCFContext->nFragments);
    fragments.emplace_back(std::pair<CfFragment, TPCCFDecodeScanTmp>{fragment, {0, 0, 0, 0, 0, -1}});
    for (unsigned int i = 1; i < mCFContext->nFragments; i++) {
      fragments.emplace_back(std::pair<CfFragment, TPCCFDecodeScanTmp>{fragments.back().first.next(), {0, 0, 0, 0, 0, -1}});
    }
    std::vector<bool> fragmentExtends(mCFContext->nFragments, false);

    unsigned int firstPossibleFragment = 0;
    unsigned int pageCounter = 0;
    unsigned int emptyPages = 0;
    for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[iSlice].count[j]; k++) {
      if (GetProcessingSettings().tpcSingleSector != -1 && GetProcessingSettings().tpcSingleSector != (int)iSlice) {
        break;
      }
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
        pageCounter++;
        const TPCZSHDR* const hdr = (const TPCZSHDR*)(rdh_utils::getLink(o2::raw::RDHUtils::getFEEID(*rdh)) == rdh_utils::DLBZSLinkID ? (page + o2::raw::RDHUtils::getMemorySize(*rdh) - sizeof(TPCZSHDRV2)) : (page + sizeof(o2::header::RAWDataHeader)));
        if (mCFContext->zsVersion == -1) {
          mCFContext->zsVersion = hdr->version;
          if (GetProcessingSettings().param.tpcTriggerHandling && mCFContext->zsVersion < ZSVersion::ZSVersionDenseLinkBased) { // TODO: Move tpcTriggerHandling to recoSteps bitmask
            static bool errorShown = false;
            if (errorShown == false) {
              GPUAlarm("Trigger handling only possible with TPC Dense Link Based data, received version %d, disabling", mCFContext->zsVersion);
            }
            errorShown = true;
          }
        } else if (mCFContext->zsVersion != (int)hdr->version) {
          GPUError("Received TPC ZS 8kb page of mixed versions, expected %d, received %d (linkid %d, feeCRU %d, feeEndpoint %d, feelinkid %d)", mCFContext->zsVersion, (int)hdr->version, (int)o2::raw::RDHUtils::getLinkID(*rdh), (int)rdh_utils::getCRU(*rdh), (int)rdh_utils::getEndPoint(*rdh), (int)rdh_utils::getLink(*rdh));
          constexpr size_t bufferSize = 3 * std::max(sizeof(*rdh), sizeof(*hdr)) + 1;
          char dumpBuffer[bufferSize];
          for (size_t i = 0; i < sizeof(*rdh); i++) {
            // "%02X " guaranteed to be 3 chars + ending 0.
            snprintf(dumpBuffer + 3 * i, 4, "%02X ", (int)((unsigned char*)rdh)[i]);
          }
          GPUAlarm("RDH of page: %s", dumpBuffer);
          for (size_t i = 0; i < sizeof(*hdr); i++) {
            // "%02X " guaranteed to be 3 chars + ending 0.
            snprintf(dumpBuffer + 3 * i, 4, "%02X ", (int)((unsigned char*)hdr)[i]);
          }
          GPUAlarm("Metainfo of page: %s", dumpBuffer);
          if (GetProcessingSettings().ignoreNonFatalGPUErrors) {
            mCFContext->abandonTimeframe = true;
            return {0, 0};
          } else {
            GPUFatal("Cannot process with invalid TPC ZS data, exiting");
          }
        }
        if (GetProcessingSettings().param.tpcTriggerHandling) {
          const TPCZSHDRV2* const hdr2 = (const TPCZSHDRV2*)hdr;
          if (hdr2->flags & TPCZSHDRV2::ZSFlags::TriggerWordPresent) {
            const char* triggerWord = (const char*)hdr - TPCZSHDRV2::TRIGGER_WORD_SIZE;
            o2::tpc::TriggerInfoDLBZS tmp;
            memcpy((void*)&tmp.triggerWord, triggerWord, TPCZSHDRV2::TRIGGER_WORD_SIZE);
            tmp.orbit = o2::raw::RDHUtils::getHeartBeatOrbit(*rdh);
            if (tmp.triggerWord.isValid(0)) {
              mTriggerBuffer->triggers.emplace(tmp);
            }
          }
        }
        nDigits += hdr->nADCsamples;
        endpointAdcSamples[j] += hdr->nADCsamples;
        unsigned int timeBin = (hdr->timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;
        unsigned int maxTimeBin = timeBin + hdr->nTimeBinSpan;
        if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
          const TPCZSHDRV2* const hdr2 = (const TPCZSHDRV2*)hdr;
          if (hdr2->flags & TPCZSHDRV2::ZSFlags::nTimeBinSpanBit8) {
            maxTimeBin += 256;
          }
        }
        if (maxTimeBin > mCFContext->tpcMaxTimeBin) {
          mCFContext->tpcMaxTimeBin = maxTimeBin;
        }
        bool extendsInNextPage = false;
        if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
          if (l + 1 < mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k] && o2::raw::RDHUtils::getMemorySize(*rdh) == TPCZSHDR::TPC_ZS_PAGE_SIZE) {
            const o2::header::RAWDataHeader* nextrdh = (const o2::header::RAWDataHeader*)(page + TPCZSHDR::TPC_ZS_PAGE_SIZE);
            extendsInNextPage = o2::raw::RDHUtils::getHeartBeatOrbit(*nextrdh) == o2::raw::RDHUtils::getHeartBeatOrbit(*rdh) && o2::raw::RDHUtils::getMemorySize(*nextrdh) > sizeof(o2::header::RAWDataHeader);
          }
        }
        while (firstPossibleFragment && (unsigned int)fragments[firstPossibleFragment - 1].first.last() > timeBin) {
          firstPossibleFragment--;
        }
        auto handleExtends = [&](unsigned int ff) {
          if (fragmentExtends[ff]) {
            if (doGPU) {
              // Only add extended page on GPU. On CPU the pages are in consecutive memory anyway.
              // Not adding the page prevents an issue where a page is decoded twice on CPU, when only the extend should be decoded.
              fragments[ff].second.zsPageLast++;
              mCFContext->fragmentData[ff].nPages[iSlice][j]++;
              mCFContext->fragmentData[ff].pageDigits[iSlice][j].emplace_back(0);
            }
            fragmentExtends[ff] = false;
          }
        };
        if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
          for (unsigned int ff = 0; ff < firstPossibleFragment; ff++) {
            handleExtends(ff);
          }
        }
        for (unsigned int f = firstPossibleFragment; f < mCFContext->nFragments; f++) {
          if (timeBin < (unsigned int)fragments[f].first.last() && (unsigned int)fragments[f].first.first() <= maxTimeBin) {
            if (!fragments[f].second.hasData) {
              fragments[f].second.hasData = 1;
              fragments[f].second.zsPtrFirst = k;
              fragments[f].second.zsPageFirst = l;
            } else {
              if (pageCounter > (unsigned int)fragments[f].second.pageCounter + 1) {
                mCFContext->fragmentData[f].nPages[iSlice][j] += emptyPages + pageCounter - fragments[f].second.pageCounter - 1;
                for (unsigned int k2 = fragments[f].second.zsPtrLast - 1; k2 <= k; k2++) {
                  for (unsigned int l2 = ((int)k2 == fragments[f].second.zsPtrLast - 1) ? fragments[f].second.zsPageLast : 0; l2 < (k2 < k ? mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k2] : l); l2++) {
                    if (doGPU) {
                      mCFContext->fragmentData[f].pageDigits[iSlice][j].emplace_back(0);
                    } else {
                      // CPU cannot skip unneeded pages, so we must keep space to store the invalid dummy clusters
                      const unsigned char* const pageTmp = ((const unsigned char*)mIOPtrs.tpcZS->slice[iSlice].zsPtr[j][k2]) + l2 * TPCZSHDR::TPC_ZS_PAGE_SIZE;
                      const o2::header::RAWDataHeader* rdhTmp = (const o2::header::RAWDataHeader*)pageTmp;
                      if (o2::raw::RDHUtils::getMemorySize(*rdhTmp) != sizeof(o2::header::RAWDataHeader)) {
                        const TPCZSHDR* const hdrTmp = (const TPCZSHDR*)(rdh_utils::getLink(o2::raw::RDHUtils::getFEEID(*rdhTmp)) == rdh_utils::DLBZSLinkID ? (pageTmp + o2::raw::RDHUtils::getMemorySize(*rdhTmp) - sizeof(TPCZSHDRV2)) : (pageTmp + sizeof(o2::header::RAWDataHeader)));
                        mCFContext->fragmentData[f].nDigits[iSlice][j] += hdrTmp->nADCsamples;
                      }
                    }
                  }
                }
              } else if (emptyPages) {
                mCFContext->fragmentData[f].nPages[iSlice][j] += emptyPages;
                if (doGPU) {
                  for (unsigned int m = 0; m < emptyPages; m++) {
                    mCFContext->fragmentData[f].pageDigits[iSlice][j].emplace_back(0);
                  }
                }
              }
            }
            fragments[f].second.zsPtrLast = k + 1;
            fragments[f].second.zsPageLast = l + 1;
            fragments[f].second.pageCounter = pageCounter;
            mCFContext->fragmentData[f].nPages[iSlice][j]++;
            mCFContext->fragmentData[f].nDigits[iSlice][j] += hdr->nADCsamples;
            if (doGPU) {
              mCFContext->fragmentData[f].pageDigits[iSlice][j].emplace_back(hdr->nADCsamples);
            }
            fragmentExtends[f] = extendsInNextPage;
          } else {
            handleExtends(f);
            if (timeBin < (unsigned int)fragments[f].first.last()) {
              if (mCFContext->zsVersion >= ZSVersion::ZSVersionDenseLinkBased) {
                for (unsigned int ff = f + 1; ff < mCFContext->nFragments; ff++) {
                  handleExtends(ff);
                }
              }
              break;
            } else {
              firstPossibleFragment = f + 1;
            }
          }
        }
        emptyPages = 0;
      }
    }
    for (unsigned int f = 0; f < mCFContext->nFragments; f++) {
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].zsPtrLast = fragments[f].second.zsPtrLast;
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].zsPtrFirst = fragments[f].second.zsPtrFirst;
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].zsPageLast = fragments[f].second.zsPageLast;
      mCFContext->fragmentData[f].minMaxCN[iSlice][j].zsPageFirst = fragments[f].second.zsPageFirst;
    }
  }
  mCFContext->nPagesTotal += nPages;
  mCFContext->nPagesSector[iSlice] = nPages;

  mCFContext->nDigitsEndpointMax[iSlice] = 0;
  for (unsigned int i = 0; i < GPUTrackingInOutZS::NENDPOINTS; i++) {
    if (endpointAdcSamples[i] > mCFContext->nDigitsEndpointMax[iSlice]) {
      mCFContext->nDigitsEndpointMax[iSlice] = endpointAdcSamples[i];
    }
  }
  unsigned int nDigitsFragmentMax = 0;
  for (unsigned int i = 0; i < mCFContext->nFragments; i++) {
    unsigned int pagesInFragment = 0;
    unsigned int digitsInFragment = 0;
    for (unsigned short j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      pagesInFragment += mCFContext->fragmentData[i].nPages[iSlice][j];
      digitsInFragment += mCFContext->fragmentData[i].nDigits[iSlice][j];
    }
    mCFContext->nPagesFragmentMax = std::max(mCFContext->nPagesFragmentMax, pagesInFragment);
    nDigitsFragmentMax = std::max(nDigitsFragmentMax, digitsInFragment);
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
  if (mCFContext->abandonTimeframe) {
    return {0, 0};
  }
  const auto& retVal = TPCClusterizerDecodeZSCountUpdate(iSlice, fragment);
  if (doGPU) {
    GPUTPCClusterFinder& clusterer = processors()->tpcClusterer[iSlice];
    GPUTPCClusterFinder& clustererShadow = doGPU ? processorsShadow()->tpcClusterer[iSlice] : clusterer;
    unsigned int nPagesSector = 0;
    for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
      unsigned int nPages = 0;
      mInputsHost->mPzsMeta->slice[iSlice].zsPtr[j] = &mInputsShadow->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j];
      mInputsHost->mPzsPtrs[iSlice * GPUTrackingInOutZS::NENDPOINTS + j] = clustererShadow.mPzs + (nPagesSector + nPages) * TPCZSHDR::TPC_ZS_PAGE_SIZE;
      for (unsigned int k = clusterer.mMinMaxCN[j].zsPtrFirst; k < clusterer.mMinMaxCN[j].zsPtrLast; k++) {
        const unsigned int min = (k == clusterer.mMinMaxCN[j].zsPtrFirst) ? clusterer.mMinMaxCN[j].zsPageFirst : 0;
        const unsigned int max = (k + 1 == clusterer.mMinMaxCN[j].zsPtrLast) ? clusterer.mMinMaxCN[j].zsPageLast : mIOPtrs.tpcZS->slice[iSlice].nZSPtr[j][k];
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
  const short maxFragmentLen = GetProcessingSettings().overrideClusterizerFragmentLen;
  const unsigned int maxAllowedTimebin = param().par.continuousTracking ? std::max<int>(param().par.continuousMaxTimeBin, maxFragmentLen) : TPC_MAX_TIME_BIN_TRIGGERED;
  mCFContext->tpcMaxTimeBin = maxAllowedTimebin;
  const CfFragment fragmentMax{(tpccf::TPCTime)mCFContext->tpcMaxTimeBin + 1, maxFragmentLen};
  mCFContext->prepare(mIOPtrs.tpcZS, fragmentMax);
  if (GetProcessingSettings().param.tpcTriggerHandling) {
    mTriggerBuffer->triggers.clear();
  }
  if (mIOPtrs.tpcZS) {
    unsigned int nDigitsFragmentMax[NSLICES];
    mCFContext->zsVersion = -1;
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      if (mIOPtrs.tpcZS->slice[iSlice].count[0]) {
        const void* rdh = mIOPtrs.tpcZS->slice[iSlice].zsPtr[0][0];
        if (rdh && o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeaderV6>() > o2::raw::RDHUtils::getVersion(rdh)) {
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
      processors()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsScaled, mCFContext->nDigitsEndpointMax[iSlice]);
      if (mRec->IsGPU()) {
        processorsShadow()->tpcClusterer[iSlice].SetNMaxDigits(processors()->tpcClusterer[iSlice].mPmemory->counters.nDigits, mCFContext->nPagesFragmentMax, nDigitsScaled, mCFContext->nDigitsEndpointMax[iSlice]);
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
      processors()->tpcClusterer[iSlice].SetNMaxDigits(nDigits, mCFContext->nPagesFragmentMax, nDigits, 0);
    }
  }

  if (mIOPtrs.tpcZS) {
    GPUInfo("Event has %u 8kb TPC ZS pages (version %d), %lld digits", mCFContext->nPagesTotal, mCFContext->zsVersion, (long long int)mRec->MemoryScalers()->nTPCdigits);
  } else {
    GPUInfo("Event has %lld TPC Digits", (long long int)mRec->MemoryScalers()->nTPCdigits);
  }

  if (mCFContext->tpcMaxTimeBin > maxAllowedTimebin) {
    GPUError("Input data has invalid time bin %u > %d", mCFContext->tpcMaxTimeBin, maxAllowedTimebin);
    if (GetProcessingSettings().ignoreNonFatalGPUErrors) {
      mCFContext->abandonTimeframe = true;
      mCFContext->tpcMaxTimeBin = maxAllowedTimebin;
    } else {
      return 1;
    }
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
    if (mIOPtrs.settingsTF->nHBFPerTF < 64) {
      threshold *= 2;
    }
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

  bool buildNativeGPU = (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCConversion) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCSliceTracking) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCMerging) || (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TPCCompression);
  bool buildNativeHost = mRec->GetRecoStepsOutputs() & GPUDataTypes::InOutType::TPCClusters; // TODO: Should do this also when clusters are needed for later steps on the host but not requested as output

  mInputsHost->mNClusterNative = mInputsShadow->mNClusterNative = mRec->MemoryScalers()->nTPCHits * tpcHitLowOccupancyScalingFactor;
  if (buildNativeGPU) {
    AllocateRegisteredMemory(mInputsHost->mResourceClusterNativeBuffer);
  }
  if (buildNativeHost && !(buildNativeGPU && GetProcessingSettings().delayedOutput)) {
    if (mWaitForFinalInputs) {
      GPUFatal("Cannot use waitForFinalInput callback without delayed output");
    }
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

  auto notifyForeignChainFinished = [this]() {
    if (mPipelineNotifyCtx) {
      SynchronizeStream(mRec->NStreams() - 2); // Must finish before updating ioPtrs in (global) constant memory
      {
        std::lock_guard<std::mutex> lock(mPipelineNotifyCtx->mutex);
        mPipelineNotifyCtx->ready = true;
      }
      mPipelineNotifyCtx->cond.notify_one();
    }
  };
  bool synchronizeCalibUpdate = false;

  for (unsigned int iSliceBase = 0; iSliceBase < NSLICES; iSliceBase += GetProcessingSettings().nTPCClustererLanes) {
    std::vector<bool> laneHasData(GetProcessingSettings().nTPCClustererLanes, false);
    static_assert(NSLICES <= GPUCA_MAX_STREAMS, "Stream events must be able to hold all slices");
    const int maxLane = std::min<int>(GetProcessingSettings().nTPCClustererLanes, NSLICES - iSliceBase);
    for (CfFragment fragment = mCFContext->fragmentFirst; !fragment.isEnd(); fragment = fragment.next()) {
      if (GetProcessingSettings().debugLevel >= 3) {
        GPUInfo("Processing time bins [%d, %d) for sectors %d to %d", fragment.start, fragment.last(), iSliceBase, iSliceBase + GetProcessingSettings().nTPCClustererLanes - 1);
      }
      GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, GetProcessingSettings().nTPCClustererLanes)))
      for (int lane = 0; lane < maxLane; lane++) {
        if (doGPU && fragment.index != 0) {
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
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPchargeMap, TPCMapMemoryLayout<ChargeMapType>::items(GetProcessingSettings().overrideClusterizerFragmentLen) * sizeof(ChargeMapType));
        runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpeakMap, TPCMapMemoryLayout<PeakMapType>::items(GetProcessingSettings().overrideClusterizerFragmentLen) * sizeof(PeakMapType));
        if (fragment.index == 0) {
          runKernel<GPUMemClean16>(GetGridAutoStep(lane, RecoStep::TPCClusterFinding), krnlRunRangeNone, {}, clustererShadow.mPpadIsNoisy, TPC_PADS_IN_SECTOR * sizeof(*clustererShadow.mPpadIsNoisy));
        }
        DoDebugAndDump(RecoStep::TPCClusterFinding, 0, clusterer, &GPUTPCClusterFinder::DumpChargeMap, *mDebugFile, "Zeroed Charges", doGPU);

        if (doGPU) {
          if (mIOPtrs.tpcZS && mCFContext->nPagesSector[iSlice] && mCFContext->zsVersion != -1) {
            TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceZS, lane);
            SynchronizeStream(GetProcessingSettings().nTPCClustererLanes + lane);
          }
          SynchronizeStream(mRec->NStreams() - 1); // Wait for copying to constant memory
        }

        if (mIOPtrs.tpcZS && (mCFContext->abandonTimeframe || !mCFContext->nPagesSector[iSlice] || mCFContext->zsVersion == -1)) {
          clusterer.mPmemory->counters.nPositions = 0;
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

        if (GetProcessingSettings().tpcSingleSector == -1 || GetProcessingSettings().tpcSingleSector == (int)iSlice) {
          if (not mIOPtrs.tpcZS) {
            runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
            TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
          } else if (propagateMCLabels) {
            runKernel<GPUTPCCFChargeMapFiller, GPUTPCCFChargeMapFiller::findFragmentStart>(GetGrid(1, lane, GPUReconstruction::krnlDeviceType::CPU), {iSlice}, {}, mIOPtrs.tpcZS == nullptr);
            TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
          }
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
            case ZSVersionDenseLinkBased:
              runKernel<GPUTPCCFDecodeZSDenseLink>(GetGridBlk(nBlocks, lane), {iSlice}, {}, firstHBF);
              break;
          }
          TransferMemoryResourceLinkToHost(RecoStep::TPCClusterFinding, clusterer.mMemoryId, lane);
        }
      }
      GPUCA_OPENMP(parallel for if(!doGPU && GetProcessingSettings().ompKernels != 1) num_threads(mRec->SetAndGetNestedLoopOmpFactor(!doGPU, GetProcessingSettings().nTPCClustererLanes)))
      for (int lane = 0; lane < maxLane; lane++) {
        unsigned int iSlice = iSliceBase + lane;
        if (doGPU) {
          SynchronizeStream(lane);
        }
        if (mIOPtrs.tpcZS) {
          CfFragment f = fragment.next();
          int nextSlice = iSlice;
          if (f.isEnd()) {
            nextSlice += GetProcessingSettings().nTPCClustererLanes;
            f = mCFContext->fragmentFirst;
          }
          if (nextSlice < NSLICES && mIOPtrs.tpcZS && mCFContext->nPagesSector[nextSlice] && mCFContext->zsVersion != -1 && !mCFContext->abandonTimeframe) {
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
        if (doGPU) {
          SynchronizeStream(lane);
        }
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
        if (doGPU) {
          SynchronizeStream(lane);
        }

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
          if (doGPU) {
            SynchronizeStream(lane);
          }
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
      if (doGPU) {
        SynchronizeStream(lane);
      }
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

    if (mWaitForFinalInputs && iSliceBase >= 21 && (int)iSliceBase < 21 + GetProcessingSettings().nTPCClustererLanes) {
      notifyForeignChainFinished();
    }
    if (mWaitForFinalInputs && iSliceBase >= 30 && (int)iSliceBase < 30 + GetProcessingSettings().nTPCClustererLanes) {
      mWaitForFinalInputs();
      synchronizeCalibUpdate = DoQueuedUpdates(0, false);
    }
  }
  for (int i = 0; i < GetProcessingSettings().nTPCClustererLanes; i++) {
    if (transferRunning[i]) {
      ReleaseEvent(&mEvents->stream[i], doGPU);
    }
  }

  if (GetProcessingSettings().param.tpcTriggerHandling) {
    GPUOutputControl* triggerOutput = mSubOutputControls[GPUTrackingOutputs::getIndex(&GPUTrackingOutputs::tpcTriggerWords)];
    if (triggerOutput && triggerOutput->allocator) {
      // GPUInfo("Storing %lu trigger words", mTriggerBuffer->triggers.size());
      auto* outputBuffer = (decltype(mTriggerBuffer->triggers)::value_type*)triggerOutput->allocator(mTriggerBuffer->triggers.size() * sizeof(decltype(mTriggerBuffer->triggers)::value_type));
      std::copy(mTriggerBuffer->triggers.begin(), mTriggerBuffer->triggers.end(), outputBuffer);
    }
    mTriggerBuffer->triggers.clear();
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

  if (!mWaitForFinalInputs) {
    notifyForeignChainFinished();
  }

  if (buildNativeGPU) {
    processorsShadow()->ioPtrs.clustersNative = mInputsShadow->mPclusterNativeAccess;
    WriteToConstantMemory(RecoStep::TPCClusterFinding, (char*)&processors()->ioPtrs - (char*)processors(), &processorsShadow()->ioPtrs, sizeof(processorsShadow()->ioPtrs), 0);
    *mInputsHost->mPclusterNativeAccess = *mIOPtrs.clustersNative;
    mInputsHost->mPclusterNativeAccess->clustersLinear = mInputsShadow->mPclusterNativeBuffer;
    mInputsHost->mPclusterNativeAccess->setOffsetPtrs();
    TransferMemoryResourceLinkToGPU(RecoStep::TPCClusterFinding, mInputsHost->mResourceClusterNativeAccess, 0);
  }
  if (doGPU && synchronizeOutput) {
    SynchronizeStream(mRec->NStreams() - 1);
  }
  if (doGPU && synchronizeCalibUpdate) {
    SynchronizeStream(0);
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
