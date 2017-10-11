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

#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>

#include <chrono>
#include <thread>

namespace o2
{
namespace DataDistribution
{

void CruLinkEmulator::linkReadoutThread()
{
  static const size_t cHBFrameFreq = 11223;
  static const size_t cStfPerS = 43; /* Parametrize this? */

  const auto cSuperpageSize = mMemHandler->getSuperpageSize();
  const auto cSuperpagesPerS = std::max(uint64_t(1), mLinkBitsPerS / (cSuperpageSize << 3));
  const auto cHBFrameSize = (mLinkBitsPerS / cHBFrameFreq) >> 3;
  const auto cStfLinkSize = cHBFrameSize * cHBFrameFreq / cStfPerS;
  const auto cNumDmaChunkPerSuperpage = std::min(size_t(256), size_t(cSuperpageSize / mDmaChunkSize));
  constexpr auto cStfTimeUs = std::chrono::microseconds(1000000 / cStfPerS);

  LOG(DEBUG) << "Superpage size: " << cSuperpageSize;
  LOG(DEBUG) << "mDmaChunkSize size: " << mDmaChunkSize;
  LOG(DEBUG) << "HBFrameSize size: " << cHBFrameSize;
  LOG(DEBUG) << "StfLinkSize size: " << cStfLinkSize;
  LOG(DEBUG) << "cNumDmaChunkPerSuperpage: " << cNumDmaChunkPerSuperpage;
  LOG(DEBUG) << "Sleep time us: " << cStfTimeUs.count();

  mRunning = true;

  // os might sleep much longer than requested
  // keep count of transmitted pages and adjust when needed
  uint64_t lFilledPages = 0;
  uint64_t lSentStf = 0;
  const auto start = std::chrono::high_resolution_clock::now();

  std::deque<CRUSuperpage> lSuperpages;

  while (mRunning) {

    const auto lStfToSend = (std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::high_resolution_clock::now() - start) /
                             cStfTimeUs) -
                            lSentStf;

    if (lStfToSend <= 0) {
      std::this_thread::sleep_for(cStfTimeUs);
      continue;
    }

    const std::int64_t lPagesToSend = std::max(lStfToSend, lStfToSend * (cStfLinkSize + cSuperpageSize - 1) / cSuperpageSize);

    // request enough superpages (can be less!)
    auto lPagesAvail = lSuperpages.size();
    if (lPagesAvail < lPagesToSend)
      lPagesAvail = mMemHandler->getSuperpages(std::max(lPagesToSend, std::int64_t(32)), std::back_inserter(lSuperpages));

    for (auto stf = 0; stf < lStfToSend; stf++, lSentStf++) {
      auto lHbfToSend = 256;

      while (lHbfToSend > 0) {
        if (!lSuperpages.empty()) {
          CRUSuperpage sp{ std::move(lSuperpages.front()) };
          lSuperpages.pop_front();

          // Enumerate valid data and create work-item for STFBuilder
          // Each channel is reported separately to the O2
          ReadoutLinkO2Data linkO2Data;

          // this is only a minimum of O2 DataHeader information for the STF builder
          // TODO: for now, only mLinkID is taken into account
          linkO2Data.mLinkDataHeader.dataOrigin = (rand() % 100 < 70) ? o2::header::gDataOriginTPC : o2::header::gDataOriginITS;
          linkO2Data.mLinkDataHeader.dataDescription = o2::header::gDataDescriptionRawData;
          linkO2Data.mLinkDataHeader.payloadSerializationMethod = o2::header::gSerializationMethodNone;
          linkO2Data.mLinkDataHeader.subSpecification = mLinkID;

          for (unsigned d = 0; d < cNumDmaChunkPerSuperpage; d++, lHbfToSend--) {

            if (lHbfToSend == 0)
              break; // start a new superpage

            linkO2Data.mLinkRawData.emplace_back(CruDmaPacket{
              mMemHandler->getDataRegion(),
              sp.mDataVirtualAddress + (d * mDmaChunkSize),   // Valid data DMA Chunk <superpage offset + length>
              mDmaChunkSize - (rand() % (mDmaChunkSize / 10)) // This should be taken from desc->mRawDataSize (filled by the CRU)
            });
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
          break;
        }
      }
    }
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
