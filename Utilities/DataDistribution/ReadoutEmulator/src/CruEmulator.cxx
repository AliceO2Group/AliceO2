// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReadoutEmulator/CruEmulator.h"
#include "Common/ConcurrentQueue.h"

#include <fairmq/FairMQDevice.h> /* NewUnmanagedRegionFor */
#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>

#include <chrono>
#include <thread>

namespace o2 {
namespace DataDistribution {

unsigned CruLinkEmulator::sCruUsedLinks = 0U;

void CruLinkEmulator::linkReadoutThread()
{
  static const size_t cHBFrameFreq = 11223;
  static const size_t cStfPerS = 45; /* Parametrize this? */

  const auto cSuperpageSize = mMemHandler->getSuperpageSize();
  const auto cSuperpagesPerS = std::max(uint64_t(1), mLinkBitsPerS / (cSuperpageSize << 3));

  const auto cHBFrameSize = (mLinkBitsPerS / cHBFrameFreq) >> 3;

  const auto cStfLinkSize = cHBFrameSize * cHBFrameFreq / cStfPerS;

  // this must take into accunt that a HFrame cannot span 2 superpages
  // const auto cNumDmaChunkPerSuperpage = (cSuperpageSize / cHBFrameSize) * cHBFrameSize / mDmaChunkSize;
  const auto cNumDmaChunkPerSuperpage = cSuperpageSize / mDmaChunkSize;

  const auto cSleepTimeUs = std::chrono::microseconds(1000000 / cSuperpagesPerS);

  LOG(DEBUG) << "Superpage size: " << cSuperpageSize;
  LOG(DEBUG) << "mDmaChunkSize size: " << mDmaChunkSize;
  LOG(DEBUG) << "HBFrameSize size: " << cHBFrameSize;
  LOG(DEBUG) << "StfLinkSize size: " << cStfLinkSize;
  LOG(DEBUG) << "cNumDmaChunkPerSuperpage: " << cNumDmaChunkPerSuperpage;
  LOG(DEBUG) << "Sleep time us: " << 1000000 / cSuperpagesPerS;

  mRunning = true;

  // os might sleep much longer than requested
  // keep count of transmitted pages and adjust when needed
  uint64_t lFilledPages = 0;
  auto start = std::chrono::high_resolution_clock::now();

  while (mRunning) {

    auto lPagesToSend =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start) /
      cSleepTimeUs;
    lPagesToSend -= lFilledPages;

    for (auto p = 0; p < lPagesToSend; p++, lFilledPages++) {
      CRUSuperpage sp;
      if (mMemHandler->getSuperpage(sp)) {

        RawDmaChunkDesc* const desc = reinterpret_cast<RawDmaChunkDesc*>(sp.mDescVirtualAddress);
        // Real-world scenario: CRU marks some of the DMA packet slots as invalid.
        // Simulate this by making ~1% of them invalid.
        for (unsigned d = 0; d < cNumDmaChunkPerSuperpage; d++) {
          desc[d].mValidHBF = (rand() % 100 > 1) ? true : false;
        }

        // Enumerate valid data and create work-item for STFBuilder
        // Each channel is reported separately to the O2
        ReadoutLinkO2Data linkO2Data;

        // this is only a minimum of O2 DataHeader information for the STF builder
        linkO2Data.mLinkDataHeader.headerSize = sizeof(DataHeader);
        linkO2Data.mLinkDataHeader.flags = 0;
        linkO2Data.mLinkDataHeader.dataDescription = o2::Header::gDataDescriptionRawData;
        linkO2Data.mLinkDataHeader.dataOrigin = o2::Header::gDataOriginTPC;
        linkO2Data.mLinkDataHeader.payloadSerializationMethod = o2::Header::gSerializationMethodNone;
        linkO2Data.mLinkDataHeader.subSpecification = mLinkID;

        for (unsigned d = 0; d < cNumDmaChunkPerSuperpage; d++) {
          if (!desc[d].mValidHBF)
            continue;

          linkO2Data.mLinkRawData.emplace_back(
            CruDmaPacket{ mMemHandler->getDataRegion(),
                          sp.mDataVirtualAddress +
                            (d * mDmaChunkSize), // Valid data DMA Chunk <superpage offset + length>
                          mDmaChunkSize,         // This should be taken from desc->mRawDataSize (filled by the CRU)
                          mMemHandler->getDescRegion(),
                          reinterpret_cast<char* const>(&desc[d]), sizeof(RawDmaChunkDesc) });
        }

        // record how many chunks are there in a superpage
        linkO2Data.mLinkDataHeader.payloadSize = linkO2Data.mLinkRawData.size();
        // Put the link info data into the send queue
        mMemHandler->putLinkData(std::move(linkO2Data));

      } else {
        // signal lost data (no free superpages)
        ReadoutLinkO2Data linkO2Data;
        linkO2Data.mLinkDataHeader.subSpecification = -1;

        mMemHandler->putLinkData(std::move(linkO2Data));
      }
    }

    std::this_thread::sleep_for(cSleepTimeUs);
  }
}

/// Start "data taking" thread
void CruLinkEmulator::start()
{
  mCRULinkThread = std::thread(&CruLinkEmulator::linkReadoutThread, this);
}

/// Stop "data taking" thread
void CruLinkEmulator::stop()
{
  mRunning = false;
  mCRULinkThread.join();
}
}
} /* namespace o2::DataDistribution */
