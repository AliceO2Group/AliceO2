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

/// \file RawPixelReader.h
/// \brief Definition of the Alpide pixel reader for raw data processing
#ifndef ALICEO2_ITSMFT_RAWPIXELREADER_H_
#define ALICEO2_ITSMFT_RAWPIXELREADER_H_

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/ChipMappingITS.h" // this will become template parameter
#include "ITSMFTReconstruction/AlpideCoder.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "CommonConstants/Triggers.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/PayLoadSG.h"

#include "ITSMFTReconstruction/GBTLink.h"
#include "ITSMFTReconstruction/RUDecodeData.h"
#include "DetectorsRaw/RDHUtils.h"

#include <TTree.h>
#include <TStopwatch.h>
#include <fairlogger/Logger.h>
#include <vector>
#include <limits>
#include <climits>
#include <memory>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string_view>
#include <array>
#include <bitset>
#include <iomanip>

#define _RAW_READER_ERROR_CHECKS_

#define OUTHEX(v, l) "0x" << std::hex << std::setfill('0') << std::setw(l) << v << std::dec

namespace o2
{
namespace itsmft
{

constexpr int MaxGBTPacketBytes = 8 * 1024;                                   // Max size of GBT packet in bytes (8KB)
constexpr int NCRUPagesPerSuperpage = 256;                                    // Expected max number of CRU pages per superpage
using RDHUtils = o2::raw::RDHUtils;

struct RawDecodingStat {
  enum DecErrors : int {
    ErrInvalidFEEId, // RDH provided invalid FEEId
    NErrorsDefined
  };

  using ULL = unsigned long long;
  uint64_t nTriggersProcessed = 0;                  // total number of triggers processed
  uint64_t nPagesProcessed = 0;                     // total number of pages processed
  uint64_t nRUsProcessed = 0;                       // total number of RUs processed (1 RU may take a few pages)
  uint64_t nBytesProcessed = 0;                     // total number of bytes (rdh->memorySize) processed
  uint64_t nNonEmptyChips = 0;                      // number of non-empty chips found
  uint64_t nHitsDecoded = 0;                        // number of hits found
  std::array<int, NErrorsDefined> errorCounts = {}; // error counters

  RawDecodingStat() = default;

  void clear()
  {
    nTriggersProcessed = 0;
    nPagesProcessed = 0;
    nRUsProcessed = 0;
    nBytesProcessed = 0;
    nNonEmptyChips = 0;
    nHitsDecoded = 0;
    errorCounts.fill(0);
  }

  void print(bool skipNoErr = true) const
  {
    printf("\nDecoding statistics\n");
    printf("%llu bytes for %llu RUs processed in %llu pages for %llu triggers\n", (ULL)nBytesProcessed, (ULL)nRUsProcessed,
           (ULL)nPagesProcessed, (ULL)nTriggersProcessed);
    printf("%llu hits found in %llu non-empty chips\n", (ULL)nHitsDecoded, (ULL)nNonEmptyChips);
    int nErr = 0;
    for (int i = NErrorsDefined; i--;) {
      nErr += errorCounts[i];
    }
    printf("Decoding errors: %d\n", nErr);
    for (int i = 0; i < NErrorsDefined; i++) {
      if (!skipNoErr || errorCounts[i]) {
        printf("%-70s: %d\n", ErrNames[i].data(), errorCounts[i]);
      }
    }
  }

  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "RDH cointains invalid FEEID" // ErrInvalidFEEId
  };

  ClassDefNV(RawDecodingStat, 2);
};

/// Used both for encoding to and decoding from the alpide raw data format
/// Requires as a template parameter a helper class for detector-specific
/// mapping between the software global chip ID and HW module ID and chip ID
/// within the module, see for example ChipMappingITS class.
/// Similar helper class must be provided for the MFT

template <class Mapping = o2::itsmft::ChipMappingITS>
class RawPixelReader : public PixelReader
{
  using Coder = o2::itsmft::AlpideCoder;

 public:
  RawPixelReader()
  {
    mRUEntry.fill(-1); // no known links in the beginning
  }

  ~RawPixelReader() override
  {
    mSWIO.Stop();
    printf("RawPixelReader IO time: ");
    mSWIO.Print();

    printf("Cache filling time: ");
    mSWCache.Print();
  }

  /// do we interpred GBT words as padded to 128 bits?
  bool isPadding128() const { return mPadding128; }

  /// do we treat CRU pages as having max size?
  bool isMaxPageImposed() const { return mImposeMaxPage; }

  /// assumed GBT word size (accounting for eventual padding)
  int getGBTWordSize() const { return mGBTWordSize; }

  /// impose padding model for GBT words
  void setPadding128(bool v)
  {
    mPadding128 = v;
    mGBTWordSize = mPadding128 ? o2::itsmft::GBTPaddedWordLength : o2::itsmft::GBTWordLength;
  }

  /// set min number of triggers to cache per frame
  void setMinTriggersToCache(int n) { mMinTriggersToCache = n > NCRUPagesPerSuperpage ? n : NCRUPagesPerSuperpage + 1; }

  int getMinTriggersToCache() const { return mMinTriggersToCache; }

  /// CRU pages are of max size of 8KB
  void imposeMaxPage(bool v) { mImposeMaxPage = v; }

  ///______________________________________________________________________
  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) override
  {
    // decode new RU if no cached non-empty chips

    if (mCurRUDecodeID >= 0) { // make sure current RU has fired chips to extract
      for (; mCurRUDecodeID < mNRUs; mCurRUDecodeID++) {
        auto& ru = mRUDecodeVec[mCurRUDecodeID];
        if (ru.lastChipChecked < ru.nChipsFired) {
          auto& chipData = ru.chipsData[ru.lastChipChecked++];
          int id = chipData.getChipID();
          chipDataVec[id].swap(chipData);
          return &chipDataVec[id];
        }
      }
      mCurRUDecodeID = 0; // no more decoded data if reached this place,
    }
    // will need to decode new trigger
    if (!mDecodeNextAuto) { // no more data in the current ROF and no automatic decoding of next one was requested
      return nullptr;
    }
    if (mMinTriggersCached < 2) { // last trigger might be incomplete, need to cache more data
      cacheLinksData(mRawBuffer);
    }
    if (mMinTriggersCached < 1 || !decodeNextTrigger()) {
      mCurRUDecodeID = -1;
      return nullptr; // nothing left
    }
    return getNextChipData(chipDataVec); // is it ok to use recursion here?
  }

  ///______________________________________________________________________
  void init() override{};

  ///______________________________________________________________________
  void clear(bool resetStat = true)
  {
    LOG(info) << "Cleaning decoder, reset_statistics_flag " << resetStat;
    if (resetStat) {
      mDecodingStat.clear();
    }
    for (auto& rudec : mRUDecodeVec) {
      rudec.clear();
    }
    for (auto& lnk : mGBTLinks) {
      lnk.clear(resetStat);
    }
    mMinTriggersCached = 0;
    mCurRUDecodeID = -1;
    mIOFile.close();
    mRawBuffer.clear();
  }

  ///================================== Encoding methods ========================

  ///______________________________________________________________________
  int digits2raw(const std::vector<o2::itsmft::Digit>& digiVec, int from, int ndig, const o2::InteractionRecord& bcData,
                 uint8_t ruSWMin = 0, uint8_t ruSWMax = 0xff)
  {
    // Convert ndig digits belonging to the same trigger to raw data
    // The digits in the vector must be in increasing chipID order
    // Return the number of pages in the link with smallest amount of pages

    int nDigTot = digiVec.size();
    assert(from < nDigTot);
    int last = (from + ndig <= nDigTot) ? from + ndig : nDigTot;
    RUDecodeData* curRUDecode = nullptr;
    ChipPixelData* curChipData = nullptr;
    ChipInfo chInfo;
    UShort_t curChipID = 0xffff; // currently processed SW chip id
    mInteractionRecord = bcData;
    ruSWMax = (ruSWMax < uint8_t(mMAP.getNRUs())) ? ruSWMax : mMAP.getNRUs() - 1;

    if (mNRUs < int(ruSWMax) - ruSWMin) { // book containers if needed
      for (uint8_t ru = ruSWMin; ru <= ruSWMax; ru++) {
        auto& ruData = getCreateRUDecode(ru);
        int nLinks = 0;
        for (int il = 0; il < RUDecodeData::MaxLinksPerRU; il++) {
          nLinks += ruData.links[il] < 0 ? 0 : 1;
        }
        mNLinks += nLinks;
        if (!nLinks) {
          LOG(info) << "Imposing single link readout for RU " << int(ru);
          ruData.links[0] = addGBTLink();
          getGBTLink(ruData.links[0])->lanes = mMAP.getCablesOnRUType(ruData.ruInfo->ruType);
          mNLinks++;
        }
      }
    }

    // place digits into corresponding chip buffers
    for (int id = from; id < last; id++) {
      const auto& dig = digiVec[id];
      if (curChipID != dig.getChipIndex()) {
        mMAP.getChipInfoSW(dig.getChipIndex(), chInfo);
        if (chInfo.ru < ruSWMin || chInfo.ru > ruSWMax) { // ignore this chip?
          continue;
        }
        curChipID = dig.getChipIndex();
        mCurRUDecodeID = chInfo.ru;
        curRUDecode = &mRUDecodeVec[mCurRUDecodeID];
        curChipData = &curRUDecode->chipsData[curRUDecode->nChipsFired++];
        curChipData->setChipID(chInfo.chOnRU->id); // set ID within the RU
      }
      curChipData->getData().emplace_back(&dig); // add new digit to the container
    }
    // convert digits to alpide data in the per-cable buffers
    int minPages = 0xffffff;
    for (mCurRUDecodeID = ruSWMin; mCurRUDecodeID <= int(ruSWMax); mCurRUDecodeID++) {
      curRUDecode = &mRUDecodeVec[mCurRUDecodeID];
      uint16_t next2Proc = 0, nchTot = mMAP.getNChipsOnRUType(curRUDecode->ruInfo->ruType);
      for (int ich = 0; ich < curRUDecode->nChipsFired; ich++) {
        auto& chipData = curRUDecode->chipsData[ich];
        convertEmptyChips(next2Proc, chipData.getChipID()); // if needed store EmptyChip flags
        next2Proc = chipData.getChipID() + 1;
        convertChip(chipData);
        chipData.clear();
      }
      convertEmptyChips(next2Proc, nchTot); // if needed store EmptyChip flags
      int minPageRU = fillGBTLinks();       // flush per-lane buffers to link buffers
      if (minPageRU < minPages) {
        minPages = minPageRU;
      }
    }

    return minPages;
  }

  //___________________________________________________________________________________
  void convertChip(o2::itsmft::ChipPixelData& chipData)
  {
    ///< convert digits of single chip to Alpide format.

    auto& ruData = mRUDecodeVec[mCurRUDecodeID]; // current RU container
    // fetch info of the chip with chipData->getChipID() ID within the RU
    const auto& chip = *mMAP.getChipOnRUInfo(ruData.ruInfo->ruType, chipData.getChipID());
    ruData.cableHWID[chip.cableHWPos] = chip.cableHW; // register the cable HW ID

    auto& pixels = chipData.getData();
    std::sort(pixels.begin(), pixels.end(),
              [](auto lhs, auto rhs) {
                if (lhs.getRow() < rhs.getRow()) {
                  return true;
                }
                if (lhs.getRow() > rhs.getRow()) {
                  return false;
                }
                return lhs.getCol() < rhs.getCol();
              });
    ruData.cableData[chip.cableHWPos].ensureFreeCapacity(40 * (2 + pixels.size())); // make sure buffer has enough capacity
    mCoder.encodeChip(ruData.cableData[chip.cableHWPos], chipData, chip.chipOnModuleHW, mInteractionRecord.bc);
  }

  //______________________________________________________
  void convertEmptyChips(int fromChip, int uptoChip)
  {
    // add empty chip words to respective cable's buffers for all chips of the current RU container
    auto& ruData = mRUDecodeVec[mCurRUDecodeID];                     // current RU container
    for (int chipIDSW = fromChip; chipIDSW < uptoChip; chipIDSW++) { // flag chips w/o data
      const auto& chip = *mMAP.getChipOnRUInfo(ruData.ruInfo->ruType, chipIDSW);
      ruData.cableHWID[chip.cableHWPos] = chip.cableHW; // register the cable HW ID
      ruData.cableData[chip.cableHWPos].ensureFreeCapacity(100);
      mCoder.addEmptyChip(ruData.cableData[chip.cableHWPos], chip.chipOnModuleHW, mInteractionRecord.bc);
    }
  }

  //___________________________________________________________________________________
  int fillGBTLinks()
  {
    // fill data of the RU to links buffer, return the number of pages in the link with smallest amount of pages
    constexpr uint8_t zero16[o2::itsmft::GBTPaddedWordLength] = {0}; // to speedup padding
    const int dummyNPages = 0xffffff;                                // any large number
    int minPages = dummyNPages;
    auto& ruData = mRUDecodeVec[mCurRUDecodeID];
    ruData.nCables = ruData.ruInfo->nCables;
    o2::header::RAWDataHeader rdh;

    RDHUtils::setTriggerOrbit(rdh, mInteractionRecord.orbit);
    RDHUtils::setHeartBeatOrbit(rdh, mInteractionRecord.orbit);
    RDHUtils::setTriggerBC(rdh, mInteractionRecord.orbit);
    RDHUtils::setHeartBeatBC(rdh, mInteractionRecord.orbit);
    RDHUtils::setTriggerType(rdh, o2::trigger::PhT); // ??
    RDHUtils::setDetectorField(rdh, mMAP.getRUDetectorField());

    int maxGBTWordsPerPacket = (MaxGBTPacketBytes - RDHUtils::getHeaderSize(rdh)) / mGBTWordSize - 2;

    int nGBTW[RUDecodeData::MaxLinksPerRU] = {0};
    for (int il = 0; il < RUDecodeData::MaxLinksPerRU; il++) {

      auto* link = getGBTLink(ruData.links[il]);
      if (!link) {
        continue;
      }
      int nGBTWordsNeeded = 0;
      for (int icab = ruData.nCables; icab--;) { // calculate number of GBT words per link
        if ((link->lanes & (0x1 << icab))) {
          int nb = ruData.cableData[icab].getSize();
          nGBTWordsNeeded += nb ? 1 + (nb - 1) / 9 : 0;
        }
      }
      // move data in padded GBT words from cable buffers to link buffers
      RDHUtils::setFEEID(rdh, mMAP.RUSW2FEEId(ruData.ruInfo->idSW, il)); // write on link 0 always
      RDHUtils::setLinkID(rdh, il);
      RDHUtils::setPageCounter(rdh, 0);
      RDHUtils::setStop(rdh, 0);
      int loadsize = RDHUtils::getHeaderSize(rdh) + (nGBTWordsNeeded + 2) * mGBTWordSize; // total data to dump
      RDHUtils::setMemorySize(rdh, loadsize < MaxGBTPacketBytes ? loadsize : MaxGBTPacketBytes);
      RDHUtils::setOffsetToNext(rdh, mImposeMaxPage ? MaxGBTPacketBytes : RDHUtils::getMemorySize(rdh));

      link->data.ensureFreeCapacity(MaxGBTPacketBytes);
      link->data.addFast(reinterpret_cast<uint8_t*>(&rdh), RDHUtils::getHeaderSize(rdh)); // write RDH for current packet
      link->nTriggers++;                                                                  // acknowledge the page, note: here we count pages, not triggers
      o2::itsmft::GBTDataHeaderL gbtHeader(0, link->lanes);
      o2::itsmft::GBTDataTrailer gbtTrailer; // lanes will be set on closing the last page

      gbtHeader.packetIdx = RDHUtils::getPageCounter(rdh);
      link->data.addFast(gbtHeader.getW8(), mGBTWordSize); // write GBT header for current packet
      if (mVerbose) {
        LOG(info) << "Filling RU data";
        RDHUtils::printRDH(rdh);
        gbtHeader.printX(mPadding128);
      }

      // now loop over the lanes served by this link, writing each time at most 9 bytes, untill all lanes are copied
      int nGBTWordsInPacket = 0;
      do {
        for (int icab = 0; icab < ruData.nCables; icab++) {
          if ((link->lanes & (0x1 << icab))) {
            auto& cableData = ruData.cableData[icab];
            int nb = cableData.getUnusedSize();
            if (!nb) {
              continue; // write 80b word only if there is something to write
            }
            if (nb > 9) {
              nb = 9;
            }
            int gbtWordStart = link->data.getSize();                                                               // beginning of the current GBT word in the link
            link->data.addFast(cableData.getPtr(), nb);                                                            // fill payload of cable
            link->data.addFast(zero16, mGBTWordSize - nb);                                                         // fill the rest of the GBT word by 0
            link->data[gbtWordStart + 9] = mMAP.getGBTHeaderRUType(ruData.ruInfo->ruType, ruData.cableHWID[icab]); // set cable flag
            cableData.setPtr(cableData.getPtr() + nb);
            nGBTWordsNeeded--;
            if (mVerbose > 1) {
              ((GBTData*)(&link->data[gbtWordStart]))->printX(mPadding128);
            }
            if (++nGBTWordsInPacket == maxGBTWordsPerPacket) { // check if new GBT packet must be created
              break;
            }
          } // storing data of single cable
        }   // loop over cables of this link

        if (nGBTWordsNeeded && nGBTWordsInPacket >= maxGBTWordsPerPacket) {
          // more data to write, write trailer and add new GBT packet
          link->data.add(gbtTrailer.getW8(), mGBTWordSize); // write empty GBT trailer for current packet
          if (mVerbose) {
            gbtTrailer.printX(mPadding128);
          }
          RDHUtils::setPageCounter(rdh, RDHUtils::getPageCounter(rdh) + 1); // flag new page
          RDHUtils::setStop(rdh, nGBTWordsNeeded < maxGBTWordsPerPacket);   // flag if this is the last packet of multi-packet
          // update remaining size, using padded GBT words (as CRU writes)
          loadsize = RDHUtils::getHeaderSize(rdh) + (nGBTWordsNeeded + 2) * mGBTWordSize; // update remaining size
          RDHUtils::setMemorySize(rdh, loadsize < MaxGBTPacketBytes ? loadsize : MaxGBTPacketBytes);
          RDHUtils::setOffsetToNext(rdh, mImposeMaxPage ? MaxGBTPacketBytes : RDHUtils::getMemorySize(rdh));
          link->data.ensureFreeCapacity(MaxGBTPacketBytes);
          link->data.addFast(reinterpret_cast<uint8_t*>(&rdh), RDHUtils::getHeaderSize(rdh)); // write RDH for current packet
          link->nTriggers++;                                                                  // acknowledge the page, note: here we count pages, not triggers
          if (mVerbose) {
            RDHUtils::printRDH(rdh);
          }
          gbtHeader.packetIdx = RDHUtils::getPageCounter(rdh);
          link->data.addFast(gbtHeader.getW8(), mGBTWordSize); // write GBT header for current packet
          if (mVerbose) {
            gbtHeader.printX(mPadding128);
          }
          nGBTWordsInPacket = 0; // reset counter of words in the packet
        }
      } while (nGBTWordsNeeded);

      gbtTrailer.lanesStops = link->lanes;
      gbtTrailer.packetDone = true;
      link->data.addFast(gbtTrailer.getW8(), mGBTWordSize); // write GBT trailer for the last packet
      if (mVerbose) {
        gbtTrailer.printX(mPadding128);
      }
      // NOTE: here we don't pad the page to 8KB, will do this when flushing everything to the sink

      if (minPages > link->nTriggers) {
        minPages = link->nTriggers;
      }

    } // loop over links of RU
    ruData.clear();
    return minPages == dummyNPages ? 0 : minPages;
  }

  //___________________________________________________________________________________
  int flushSuperPages(int maxPages, PayLoadCont& sink, bool unusedToHead = true)
  {
    // flush superpage (at most maxPages) of each link to the output,
    // return total number of pages flushed

    int totPages = 0;
    for (int ru = 0; ru < mMAP.getNRUs(); ru++) {
      auto* ruData = getRUDecode(ru);
      if (!ruData) {
        continue;
      }
      for (int il = 0; il < RUDecodeData::MaxLinksPerRU; il++) {
        auto link = getGBTLink(ruData->links[il]);
        if (!link || link->data.isEmpty()) {
          continue;
        }
        int nPages = 0;
        sink.ensureFreeCapacity(maxPages * MaxGBTPacketBytes);
        const auto* ptrIni = link->data.getPtr();
        while (nPages < maxPages && !link->data.isEmpty()) {
          const auto ptr = link->data.getPtr();
          o2::header::RAWDataHeader* rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);
          sink.addFast(ptr, RDHUtils::getMemorySize(rdh));                    // copy header + payload
          sink.fillFast(0, MaxGBTPacketBytes - RDHUtils::getMemorySize(rdh)); // complete with 0's till the end of the page
          link->data.setPtr(ptr + RDHUtils::getMemorySize(rdh));
          link->nTriggers--; // here we count pages, not triggers
          nPages++;
        }
        totPages += nPages;
        if (unusedToHead) {
          link->data.moveUnusedToHead();
        }
      } // loop over links
    }   // loop over RUs
    return totPages;
  }

  ///================================== Decoding methods ========================

  //_____________________________________
  size_t cacheLinksData(PayLoadCont& buffer)
  {
    // distribute data from the single buffer among the links caches

    LOG(info) << "Caching links data, currently in cache: " << mMinTriggersCached << " triggers";
    auto nRead = loadInput(buffer);
    if (buffer.isEmpty()) {
      return nRead;
    }
    mSWCache.Start(false);
    enum LinkFlag : int8_t { NotUpdated,
                             Updated,
                             HasEnoughTriggers };
    LinkFlag linkFlags[Mapping::getNRUs()][3] = {NotUpdated}; // flag that enough triggeres were loaded for this link
    int nLEnoughTriggers = 0;                                 // number of links for we which enough number of triggers were loaded
    auto ptr = buffer.getPtr();
    o2::header::RAWDataHeader* rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);

    do {
      if (!RDHUtils::checkRDH(rdh)) { // does it look like RDH?
        if (!findNextRDH(buffer)) {   // try to recover the pointer
          break;                      // no data to continue
        }
        ptr = buffer.getPtr();
        rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);
      }
      if (mVerbose) {
        RDHUtils::printRDH(rdh);
      }

      int ruIDSW = mMAP.FEEId2RUSW(RDHUtils::getFEEID(rdh));
#ifdef _RAW_READER_ERROR_CHECKS_
      if (ruIDSW >= mMAP.getNRUs()) {
        mDecodingStat.errorCounts[RawDecodingStat::ErrInvalidFEEId]++;
        LOG(error) << mDecodingStat.ErrNames[RawDecodingStat::ErrInvalidFEEId]
                   << " : FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << ", skipping CRU page";
        RDHUtils::printRDH(rdh);
        ptr += RDHUtils::getOffsetToNext(rdh);
        buffer.setPtr(ptr);
        if (buffer.getUnusedSize() < MaxGBTPacketBytes) {
          nRead += loadInput(buffer); // update
          ptr = buffer.getPtr();      // pointer might have been changed
        }
        continue;
      }
#endif
      auto& ruDecode = getCreateRUDecode(ruIDSW);

      bool newTrigger = true; // check if we see new trigger
      uint16_t lr, ruOnLr, linkIDinRU;
      mMAP.expandFEEId(RDHUtils::getFEEID(rdh), lr, ruOnLr, linkIDinRU);
      auto link = getGBTLink(ruDecode.links[linkIDinRU]);
      if (link) {                                                                                                    // was there any data seen on this link before?
        const auto rdhPrev = reinterpret_cast<o2::header::RAWDataHeader*>(link->data.getEnd() - link->lastPageSize); // last stored RDH
        if (isSameRUandTrigger(rdhPrev, rdh)) {
          newTrigger = false;
        }
      } else { // a new link was added
        ruDecode.links[linkIDinRU] = addGBTLink();
        link = getGBTLink(ruDecode.links[linkIDinRU]);
        link->statistics.feeID = RDHUtils::getFEEID(rdh);
        LOG(info) << "Adding new GBT LINK FEEId:" << OUTHEX(link->statistics.feeID, 4);
        mNLinks++;
      }
      if (linkFlags[ruIDSW][linkIDinRU] == NotUpdated) {
        link->data.moveUnusedToHead(); // reuse space of already processed data
        linkFlags[ruIDSW][linkIDinRU] = Updated;
      }
      // copy data to the buffer of the link and memorize its RDH pointer
      link->data.add(ptr, RDHUtils::getMemorySize(rdh));
      link->lastPageSize = RDHUtils::getMemorySize(rdh); // account new added size
      auto rdhC = reinterpret_cast<o2::header::RAWDataHeader*>(link->data.getEnd() - link->lastPageSize);
      RDHUtils::setOffsetToNext(rdhC, RDHUtils::getMemorySize(rdh)); // since we skip 0-s, we have to modify the offset

      if (newTrigger) {
        link->nTriggers++; // acknowledge 1st trigger
        if (link->nTriggers >= mMinTriggersToCache && linkFlags[ruIDSW][linkIDinRU] != HasEnoughTriggers) {
          nLEnoughTriggers++;
          linkFlags[ruIDSW][linkIDinRU] = HasEnoughTriggers;
        }
      }

      ptr += RDHUtils::getOffsetToNext(rdh);
      buffer.setPtr(ptr);
      if (buffer.getUnusedSize() < MaxGBTPacketBytes) {
        nRead += loadInput(buffer); // update
        ptr = buffer.getPtr();      // pointer might have been changed
      }

      rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);

      if (mNLinks == nLEnoughTriggers) {
        break;
      }

    } while (!buffer.isEmpty());

    if (mNLinks == nLEnoughTriggers) {
      mMinTriggersCached = mMinTriggersToCache; // wanted number of triggers acquired
    } else {                                    // there were no enough triggers to fulfill mMinTriggersToCache requirement
      mMinTriggersCached = INT_MAX;
      for (int ir = 0; ir < mNRUs; ir++) {
        const auto& ruDecData = mRUDecodeVec[ir];
        for (auto linkID : ruDecData.links) {
          const auto* link = getGBTLink(linkID);
          if (link && link->nTriggers < mMinTriggersCached) {
            mMinTriggersCached = link->nTriggers;
          }
        }
      }
    }
    mSWCache.Stop();
    LOG(info) << "Cached at least " << mMinTriggersCached << " triggers on " << mNLinks << " links of " << mNRUs << " RUs";

    return nRead;
  }

  //_____________________________________
  int decodeNextTrigger() final
  {
    // Decode next trigger from the cached links data and decrease cached triggers counter, return N links decoded
    if (mMinTriggersCached < 1) {
      cacheLinksData(mRawBuffer);
      if (mMinTriggersCached < 1) {
        return 0;
      }
    }
    int nlinks = 0;
    for (int ir = mNRUs; ir--;) {
      auto& ruDecode = mRUDecodeVec[ir];
      if (!nlinks) {                         // on 1st occasion extract trigger data
        for (auto linkID : ruDecode.links) { // loop over links to fill cable buffers
          auto* link = getGBTLink(linkID);
          if (link && !link->data.isEmpty()) {
            const auto rdh = reinterpret_cast<const o2::header::RAWDataHeader*>(link->data.getPtr());
            mInteractionRecord = RDHUtils::getTriggerIR(rdh);
            mTrigger = RDHUtils::getTriggerType(rdh);
            mInteractionRecordHB = RDHUtils::getHeartBeatIR(rdh);
            break;
          }
        }
      }

      nlinks += decodeNextRUData(ruDecode);
      mDecodingStat.nRUsProcessed++;
    }
    if (nlinks) {
      mDecodingStat.nTriggersProcessed++;
    }
    mCurRUDecodeID = 0;
    mMinTriggersCached--;
    return nlinks;
  }

  //_____________________________________
  int decodeNextRUData(RUDecodeData& ruDecData)
  {
    // process data of single RU trigger from its links buffers
    int minTriggers = INT_MAX;
    int res = 0;
    ruDecData.clear();
    bool aborted = false;
    for (auto linkID : ruDecData.links) { // loop over links to fill cable buffers
      auto* link = getGBTLink(linkID);
      if (link && !link->data.isEmpty()) {
        link->data.setPtr(decodeRUData(link->data.getPtr(), ruDecData, aborted));
        // we don't need to check the "abort" status since the checks for links data presence and synchronization
        // should have been done in advance
        if (--link->nTriggers < minTriggers) { // decrement counter of cached triggers
          minTriggers = link->nTriggers;
        }
        res++;
        if (link->data.isEmpty()) {
          link->data.clear();
        }
      }
    }
    if (ruDecData.nCables) {       // there are cables with data to decode
      decodeAlpideData(ruDecData); // decode Alpide data from the compressed RU Data
    }
    return res;
  }

  //_____________________________________
  bool findNextRDH(PayLoadCont& buffer)
  {
    // keep reading GRB words until RDH is found
    size_t nRead = 0;
    int scan = 0;
    bool goodRDH = false;
    auto ptr = buffer.getPtr();
    o2::header::RAWDataHeader* rdh = nullptr;
    do {
      if (buffer.isEmpty()) {
        auto nrl = loadInput(buffer);
        if (!nrl) {
          break;
        }
        nRead += nrl;
        ptr = buffer.getPtr();
      }
      scan++;
      ptr += mGBTWordSize;
      buffer.setPtr(ptr);
      if (!buffer.isEmpty()) {
        rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);
      } else {
        break;
      }
    } while (!(goodRDH = RDHUtils::checkRDH(rdh)));
    LOG(info) << "End of pointer recovery after skipping " << scan << " GBT words, RDH is"
              << (goodRDH ? "" : " not") << " found";
    return goodRDH;
  }

  //_____________________________________
  uint8_t* decodeRUData(uint8_t* raw, RUDecodeData& ruDecData, bool& aborted)
  {
    /// Decode raw data of single RU (possibly in a few GBT packets), collecting raw data
    /// for every cable in the corresponding slot of the provided ruDecData.
    /// No check is done if the necessary data are fully contained in the raw buffer.
    /// Return the pointer on the last raw data byte after decoding the RU
    /// In case of unrecoverable error set aborted to true

    aborted = false;

    // data must start by RDH
    auto rdh = reinterpret_cast<o2::header::RAWDataHeader*>(raw);

#ifdef _RAW_READER_ERROR_CHECKS_
    if (!RDHUtils::checkRDH(rdh)) {
      LOG(error) << "Page does not start with RDH";
      RDHUtils::printRDH(rdh);
      for (int i = 0; i < 4; i++) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw + i * 16);
        gbtD->printX(mPadding128);
      }
      raw += mGBTWordSize;
      aborted = true;
      return raw;
    }
#endif

    int ruIDSW = mMAP.FEEId2RUSW(RDHUtils::getFEEID(rdh));
#ifdef _RAW_READER_ERROR_CHECKS_
    if (ruIDSW >= mMAP.getNRUs()) {
      mDecodingStat.errorCounts[RawDecodingStat::ErrInvalidFEEId]++;
      LOG(error) << mDecodingStat.ErrNames[RawDecodingStat::ErrInvalidFEEId]
                 << " : FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << ", skipping CRU page";
      RDHUtils::printRDH(rdh);
      raw += RDHUtils::getOffsetToNext(rdh);
      return raw;
    }

    if (ruIDSW != ruDecData.ruInfo->idSW) { // should not happen with cached data
      LOG(error) << "RDG RU IDSW " << ruIDSW << " differs from expected " << ruDecData.ruInfo->idSW;
      RDHUtils::printRDH(rdh);
    }
#endif

    uint16_t lr, ruOnLr, linkIDinRU;
    mMAP.expandFEEId(RDHUtils::getFEEID(rdh), lr, ruOnLr, linkIDinRU);
    auto* ruLink = getGBTLink(ruDecData.links[linkIDinRU]);
    auto& ruLinkStat = ruLink->statistics;
    ruLink->lastRDH = reinterpret_cast<o2::header::RDHAny*>(rdh); // hack but this reader should be outphased anyway
    ruLinkStat.nPackets++;

#ifdef _RAW_READER_ERROR_CHECKS_
    if (RDHUtils::getPacketCounter(rdh) > ruLink->packetCounter + 1) {
      ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump]++;
      LOG(warn) << ruLinkStat.ErrNames[GBTLinkDecodingStat::ErrPacketCounterJump]
                << " : FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << ": jump from " << int(ruLink->packetCounter)
                << " to " << int(RDHUtils::getPacketCounter(rdh));
      RDHUtils::printRDH(rdh);
    }
#endif

    ruDecData.nCables = ruDecData.ruInfo->nCables;
    while (1) {
      ruLink->packetCounter = RDHUtils::getPacketCounter(rdh);

      mDecodingStat.nBytesProcessed += RDHUtils::getMemorySize(rdh);
      mDecodingStat.nPagesProcessed++;
      raw += RDHUtils::getHeaderSize(rdh);
      int nGBTWords = (RDHUtils::getMemorySize(rdh) - RDHUtils::getHeaderSize(rdh)) / mGBTWordSize - 2; // number of GBT words excluding header/trailer
      auto gbtH = reinterpret_cast<const o2::itsmft::GBTDataHeaderL*>(raw);                             // process GBT header

#ifdef _RAW_READER_ERROR_CHECKS_
      if (mVerbose) {
        RDHUtils::printRDH(rdh);
        gbtH->printX(mPadding128);
        LOG(info) << "Expect " << nGBTWords << " GBT words";
      }

      if (!gbtH->isDataHeader()) {
        gbtH->printX(mPadding128);
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " GBT payload header was expected, abort page decoding";
        RDHUtils::printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
        gbtH->printX(mPadding128);
        aborted = true;
        return raw;
      }

      if (gbtH->packetIdx != RDHUtils::getPageCounter(rdh)) {
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Different GBT header " << gbtH->packetIdx
                   << " and RDH page " << RDHUtils::getPageCounter(rdh) << " counters";
        RDHUtils::printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt]++;
      }

      if (ruLink->lanesActive == ruLink->lanesStop) { // all lanes received their stop, new page 0 expected
        if (RDHUtils::getPageCounter(rdh)) {          // flag lanes of this FEE
          LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Non-0 page counter (" << RDHUtils::getPageCounter(rdh) << ") while all lanes were stopped";
          RDHUtils::printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]++;
        }
      }

      ruLink->lanesActive = gbtH->activeLanes; // TODO do we need to update this for every page?

      if (~(mMAP.getCablesOnRUType(ruDecData.ruInfo->ruType)) & ruLink->lanesActive) { // are there wrong lanes?
        std::bitset<32> expectL(mMAP.getCablesOnRUType(ruDecData.ruInfo->ruType)), gotL(ruLink->lanesActive);
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Active lanes pattern " << gotL
                   << " conflicts with expected " << expectL << " for given RU type, skip page";
        RDHUtils::printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrInvalidActiveLanes]++;
        raw = ((uint8_t*)rdh) + RDHUtils::getOffsetToNext(rdh); // jump to the next packet
        return raw;
      }

      if (!RDHUtils::getPageCounter(rdh)) { // reset flags
        ruLink->lanesStop = 0;
        ruLink->lanesWithData = 0;
      }

#endif
      raw += mGBTWordSize;
      for (int iw = 0; iw < nGBTWords; iw++, raw += mGBTWordSize) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw);
        // TODO: need to clarify if the nGBTWords from the RDHUtils::getMemorySize(rdh) is reliable estimate of the real payload, at the moment this is not the case

        if (mVerbose > 1) {
          printf("W%4d |", iw);
          gbtD->printX(mPadding128);
        }
        if (gbtD->isDataTrailer()) {
          nGBTWords = iw;
          break; // this means that the nGBTWords estimate was wrong
        }

        int cableHW = gbtD->getCableID();
        int cableSW = mMAP.cableHW2SW(ruDecData.ruInfo->ruType, cableHW);
        ruDecData.cableData[cableSW].add(gbtD->getW8(), 9);
        ruDecData.cableHWID[cableSW] = cableHW;

#ifdef _RAW_READER_ERROR_CHECKS_
        int cableHWPos = mMAP.cableHW2Pos(ruDecData.ruInfo->ruType, cableHW);
        ruDecData.cableLinkID[cableSW] = linkIDinRU;
        ruLink->lanesWithData |= 0x1 << cableHWPos;    // flag that the data was seen on this lane
        if (ruLink->lanesStop & (0x1 << cableHWPos)) { // make sure stopped lanes do not transmit the data
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane]++;
          LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Data received for stopped lane " << cableHW << " (sw:" << cableSW << ")";
          RDHUtils::printRDH(rdh);
        }
#endif

      } // we are at the trailer, packet is over, check if there are more for the same ru

      auto gbtT = reinterpret_cast<const o2::itsmft::GBTDataTrailer*>(raw); // process GBT trailer
#ifdef _RAW_READER_ERROR_CHECKS_

      if (mVerbose) {
        gbtT->printX(mPadding128);
      }

      if (!gbtT->isDataTrailer()) {
        gbtT->printX(mPadding128);
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << std::dec
                   << " GBT payload trailer was expected, abort page decoding NW" << nGBTWords;
        RDHUtils::printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer]++;
        aborted = true;
        return raw;
      }

      ruLink->lanesTimeOut |= gbtT->lanesTimeout; // register timeouts
      ruLink->lanesStop |= gbtT->lanesStops;      // register stops
#endif
      raw += mGBTWordSize;
      // we finished the GBT page, see if there is a continuation and if it belongs to the same multipacket

      if (!RDHUtils::getOffsetToNext(rdh)) { // RS TODO: what the last page in memory will contain as offsetToNext, is it 0?
        break;
      }

      raw = ((uint8_t*)rdh) + RDHUtils::getOffsetToNext(rdh); // jump to the next packet:
      auto rdhN = reinterpret_cast<o2::header::RAWDataHeader*>(raw);
      // check if data of given RU are over, i.e. we the page counter was wrapped to 0 (should be enough!) or other RU/trigger started
      if (!isSameRUandTrigger(rdh, rdhN)) {

#ifdef _RAW_READER_ERROR_CHECKS_
        // make sure all lane stops for finished page are received
        if ((ruLink->lanesActive & ~ruLink->lanesStop) && nGBTWords) {
          if (RDHUtils::getTriggerType(rdh) != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
            std::bitset<32> active(ruLink->lanesActive), stopped(ruLink->lanesStop);
            LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " end of FEE data but not all lanes received stop"
                       << "| active: " << active << " stopped: " << stopped;
            RDHUtils::printRDH(rdh);
            ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes]++;
          }
        }

        // make sure all active lanes (except those in time-out) have sent some data
        if ((~ruLink->lanesWithData & ruLink->lanesActive) != ruLink->lanesTimeOut && nGBTWords) {
          std::bitset<32> withData(ruLink->lanesWithData), active(ruLink->lanesActive), timeOut(ruLink->lanesTimeOut);
          LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Lanes not in time-out but not sending data"
                     << "\n| with data: " << withData << " active: " << active << " timeOut: " << timeOut;
          RDHUtils::printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane]++;
        }
#endif
        // accumulate packet states
        ruLinkStat.packetStates[gbtT->getPacketState()]++;

        break;
      }
#ifdef _RAW_READER_ERROR_CHECKS_
      // check if the page counter increases
      if (RDHUtils::getPageCounter(rdhN) != RDHUtils::getPageCounter(rdh) + 1) {
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Discontinuity in the RDH page counter of the same RU trigger: old "
                   << RDHUtils::getPageCounter(rdh) << " new: " << RDHUtils::getPageCounter(rdhN);
        RDHUtils::printRDH(rdh);
        RDHUtils::printRDH(rdhN);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPageCounterDiscontinuity]++;
      }
#endif
      rdh = rdhN;
      ruLink->lastRDH = reinterpret_cast<o2::header::RDHAny*>(rdh);
    }

#ifdef _RAW_READER_ERROR_CHECKS_
//    if (RDHUtils::getPageCounter(rdh) && !RDHUtils::getStop(rdh)) {
//      LOG(warning) << "Last packet(" << RDHUtils::getPageCounter(rdh) << ") of GBT multi-packet is reached w/o STOP set in the RDH";
//    }
#endif

    return raw;
  }

  //_____________________________________
  int skimNextRUData(PayLoadCont& outBuffer)
  {
    if (mIOFile) {
      loadInput(mRawBuffer); // if needed, upload additional data to the buffer
    }

    int res = 0;
    if (!mRawBuffer.isEmpty()) {
      bool aborted = false;

      auto ptr = skimPaddedRUData(mRawBuffer.getPtr(), outBuffer, aborted);
      mDecodingStat.nRUsProcessed++;
      if (!aborted) {
        mRawBuffer.setPtr(ptr);
        res = 1; // success
        if (mRawBuffer.isEmpty()) {
          mRawBuffer.clear();
        }
      } else { // try to seek to the next RDH, can be done only for 128b padded GBT words
        if (findNextRDH(mRawBuffer)) {
          ptr = mRawBuffer.getPtr();
          res = 1;
        } else {
          mRawBuffer.clear(); // did not find new RDH
        }
      } // try to seek to the next ...
    }
    return res;
  }

  //_____________________________________
  uint8_t* skimPaddedRUData(uint8_t* raw, PayLoadCont& outBuffer, bool& aborted)
  {
    /// Skim CRU data with 128b-padded GBT words and fixed 8KB pages to 80b-GBT words and
    /// page size corresponding to real payload.

    aborted = false;

    // data must start by RDH
    auto rdh = reinterpret_cast<o2::header::RAWDataHeader*>(raw);
#ifdef _RAW_READER_ERROR_CHECKS_
    if (!RDHUtils::checkRDH(rdh)) {
      LOG(error) << "Page does not start with RDH";
      RDHUtils::printRDH(rdh);
      for (int i = 0; i < 4; i++) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw + i * 16);
        gbtD->printX(mPadding128);
      }
      aborted = true;
      return raw;
    }
    int ruIDSWD = mMAP.FEEId2RUSW(RDHUtils::getFEEID(rdh));
    if (ruIDSWD >= mMAP.getNRUs()) {
      mDecodingStat.errorCounts[RawDecodingStat::ErrInvalidFEEId]++;
      LOG(error) << mDecodingStat.ErrNames[RawDecodingStat::ErrInvalidFEEId]
                 << " : FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << ", skipping CRU page";
      RDHUtils::printRDH(rdh);
      raw += RDHUtils::getOffsetToNext(rdh);
      return raw;
    }
#endif
    uint16_t lr, ruOnLr, linkIDinRU;
    mMAP.expandFEEId(RDHUtils::getFEEID(rdh), lr, ruOnLr, linkIDinRU);
    int ruIDSW = mMAP.FEEId2RUSW(RDHUtils::getFEEID(rdh));
    auto& ruDecode = getCreateRUDecode(ruIDSW);
    auto ruInfo = mMAP.getRUInfoSW(ruIDSW);

    if (ruDecode.links[linkIDinRU] < 0) {
      ruDecode.links[linkIDinRU] = addGBTLink();
      getGBTLink(ruDecode.links[linkIDinRU])->statistics.feeID = RDHUtils::getFEEID(rdh);
      mNLinks++;
    }

    mInteractionRecord = RDHUtils::getTriggerIR(rdh);

    mTrigger = RDHUtils::getTriggerType(rdh);

    mInteractionRecordHB = RDHUtils::getHeartBeatIR(rdh);

    auto ruLink = getGBTLink(ruDecode.links[linkIDinRU]);
    auto& ruLinkStat = ruLink->statistics;
    ruLink->lastRDH = reinterpret_cast<o2::header::RDHAny*>(rdh); // hack but this reader should be outphased anyway
    ruLinkStat.nPackets++;

#ifdef _RAW_READER_ERROR_CHECKS_
    if (RDHUtils::getPacketCounter(rdh) > ruLink->packetCounter + 1) {
      ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump]++;
      LOG(warn) << ruLinkStat.ErrNames[GBTLinkDecodingStat::ErrPacketCounterJump]
                << " : FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << ": jump from " << int(ruLink->packetCounter)
                << " to " << int(RDHUtils::getPacketCounter(rdh));
      RDHUtils::printRDH(rdh);
    }
#endif
    ruLink->packetCounter = RDHUtils::getPacketCounter(rdh);

    int sizeAtEntry = outBuffer.getSize(); // save the size of outbuffer size at entry, in case of severe error we will need to rewind to it.

    while (1) {
      mDecodingStat.nPagesProcessed++;
      mDecodingStat.nBytesProcessed += RDHUtils::getMemorySize(rdh);
      raw += RDHUtils::getHeaderSize(rdh);
      // number of 128 b GBT words excluding header/trailer
      int nGBTWords = (RDHUtils::getMemorySize(rdh) - RDHUtils::getHeaderSize(rdh)) / o2::itsmft::GBTPaddedWordLength - 2;
      auto gbtH = reinterpret_cast<const o2::itsmft::GBTDataHeaderL*>(raw); // process GBT header

#ifdef _RAW_READER_ERROR_CHECKS_
      if (mVerbose) {
        RDHUtils::printRDH(rdh);
        gbtH->printX(true);
        LOG(info) << "Expect " << nGBTWords << " GBT words";
      }
      if (!gbtH->isDataHeader()) {
        gbtH->printX(true);
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " GBT payload header was expected, abort page decoding";
        RDHUtils::printRDH(rdh);
        gbtH->printX(true);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
        aborted = true;
        outBuffer.shrinkToSize(sizeAtEntry); // reset output buffer to initial state
        return raw;
      }
      if (gbtH->packetIdx != RDHUtils::getPageCounter(rdh)) {
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Different GBT header " << gbtH->packetIdx
                   << " and RDH page " << RDHUtils::getPageCounter(rdh) << " counters";
        RDHUtils::printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt]++;
      }

      if (ruLink->lanesActive == ruLink->lanesStop) { // all lanes received their stop, new page 0 expected
        if (RDHUtils::getPageCounter(rdh)) {          // flag lanes of this FEE
          LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Non-0 page counter (" << RDHUtils::getPageCounter(rdh) << ") while all lanes were stopped";
          RDHUtils::printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]++;
        }
      }

      ruLink->lanesActive = gbtH->activeLanes; // TODO do we need to update this for every page?

      if (!RDHUtils::getPageCounter(rdh)) { // reset flags
        ruLink->lanesStop = 0;
        ruLink->lanesWithData = 0;
      }

#endif
      // start writting skimmed data for this page, making sure the buffer has enough free slots
      outBuffer.ensureFreeCapacity(8 * 1024);
      auto rdhS = reinterpret_cast<o2::header::RAWDataHeader*>(outBuffer.getEnd()); // save RDH and make saved copy editable
      outBuffer.addFast(reinterpret_cast<const uint8_t*>(rdh), RDHUtils::getHeaderSize(rdh));

      outBuffer.addFast(reinterpret_cast<const uint8_t*>(gbtH), mGBTWordSize); // save gbt header w/o 128b padding

      raw += o2::itsmft::GBTPaddedWordLength;
      for (int iw = 0; iw < nGBTWords; iw++, raw += o2::itsmft::GBTPaddedWordLength) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw);
        // TODO: need to clarify if the nGBTWords from the RDHUtils::getMemorySize(rdh) is reliable estimate of the real payload, at the moment this is not the case

        if (mVerbose > 1) {
          printf("W%4d |", iw);
          gbtD->printX(mPadding128);
        }
        if (gbtD->isDataTrailer()) {
          nGBTWords = iw;
          break; // this means that the nGBTWords estimate was wrong
        }

        int cableHW = gbtD->getCableID();
        int cableSW = mMAP.cableHW2SW(ruInfo->ruType, cableHW);

        outBuffer.addFast(reinterpret_cast<const uint8_t*>(gbtD), mGBTWordSize); // save gbt word w/o 128b padding

#ifdef _RAW_READER_ERROR_CHECKS_
        int cableHWPos = mMAP.cableHW2Pos(ruInfo->ruType, cableHW);
        ruLink->lanesWithData |= 0x1 << cableHWPos;    // flag that the data was seen on this lane
        if (ruLink->lanesStop & (0x1 << cableHWPos)) { // make sure stopped lanes do not transmit the data
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane]++;
          LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Data received for stopped lane " << cableHW << " (sw:" << cableSW << ")";
          RDHUtils::printRDH(rdh);
        }
#endif

      } // we are at the trailer, packet is over, check if there are more for the same ru

      auto gbtT = reinterpret_cast<const o2::itsmft::GBTDataTrailer*>(raw); // process GBT trailer
#ifdef _RAW_READER_ERROR_CHECKS_

      if (mVerbose) {
        gbtT->printX(true);
      }

      if (!gbtT->isDataTrailer()) {
        gbtT->printX(true);
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " GBT payload trailer was expected, abort page decoding at NW" << nGBTWords;
        RDHUtils::printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer]++;
        aborted = true;
        outBuffer.shrinkToSize(sizeAtEntry); // reset output buffer to initial state
        return raw;
      }

      ruLink->lanesTimeOut |= gbtT->lanesTimeout; // register timeouts
      ruLink->lanesStop |= gbtT->lanesStops;      // register stops
#endif

      outBuffer.addFast(reinterpret_cast<const uint8_t*>(gbtT), mGBTWordSize); // save gbt trailer w/o 128b padding

      raw += o2::itsmft::GBTPaddedWordLength;

      // we finished the GBT page, register in the stored RDH the memory size and new offset
      RDHUtils::setMemorySize(rdhS, RDHUtils::getHeaderSize(rdhS) + (2 + nGBTWords) * mGBTWordSize);
      RDHUtils::setOffsetToNext(rdhS, RDHUtils::getMemorySize(rdhS));

      if (!RDHUtils::getOffsetToNext(rdh)) { // RS TODO: what the last page in memory will contain as offsetToNext, is it 0?
        break;
      }

      raw = ((uint8_t*)rdh) + RDHUtils::getOffsetToNext(rdh); // jump to the next packet:
      auto rdhN = reinterpret_cast<o2::header::RAWDataHeader*>(raw);
      // check if data of given RU are over, i.e. we the page counter was wrapped to 0 (should be enough!) or other RU/trigger started
      if (!isSameRUandTrigger(rdh, rdhN)) {

#ifdef _RAW_READER_ERROR_CHECKS_
        // make sure all lane stops for finished page are received
        if (ruLink->lanesActive != ruLink->lanesStop && nGBTWords) {
          if (RDHUtils::getTriggerType(rdh) != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
            std::bitset<32> active(ruLink->lanesActive), stopped(ruLink->lanesStop);
            LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " end of FEE data but not all lanes received stop"
                       << "| active: " << active << " stopped: " << stopped;
            RDHUtils::printRDH(rdh);
            ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes]++;
          }
        }

        // make sure all active lanes (except those in time-out) have sent some data
        if ((~ruLink->lanesWithData & ruLink->lanesActive) != ruLink->lanesTimeOut && nGBTWords) {
          std::bitset<32> withData(ruLink->lanesWithData), active(ruLink->lanesActive), timeOut(ruLink->lanesTimeOut);
          LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Lanes not in time-out but not sending data"
                     << "| with data: " << withData << " active: " << active << " timeOut: " << timeOut;
          RDHUtils::printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane]++;
        }

        // accumulate packet states
        ruLinkStat.packetStates[gbtT->getPacketState()]++;
#endif

        break;
      }
#ifdef _RAW_READER_ERROR_CHECKS_
      // check if the page counter increases
      if (RDHUtils::getPageCounter(rdhN) != RDHUtils::getPageCounter(rdh) + 1) {
        LOG(error) << "FEEId:" << OUTHEX(RDHUtils::getFEEID(rdh), 4) << " Discontinuity in the RDH page counter of the same RU trigger: old "
                   << RDHUtils::getPageCounter(rdh) << " new: " << RDHUtils::getPageCounter(rdhN);
        RDHUtils::printRDH(rdh);
        RDHUtils::printRDH(rdhN);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPageCounterDiscontinuity]++;
      }
#endif
      rdh = rdhN;
      ruLink->lastRDH = reinterpret_cast<o2::header::RDHAny*>(rdh); // hack but this reader should be outphased anyway
    }

#ifdef _RAW_READER_ERROR_CHECKS_
//    if (RDHUtils::getPageCounter(rdh) && !RDHUtils::getStop(rdh)) {
//      LOG(warning) << "Last packet(" << RDHUtils::getPageCounter(rdh) << ") of GBT multi-packet is reached w/o STOP set in the RDH";
//    }
#endif

    return raw;
  }

  //_____________________________________
  bool isSameRUandTrigger(const o2::header::RAWDataHeader* rdhOld, const o2::header::RAWDataHeader* rdhNew) const
  {
    /// check if the rdhNew is just a continuation of the data described by the rdhOld
    if (RDHUtils::getPageCounter(rdhNew) == 0 || RDHUtils::getFEEID(rdhNew) != RDHUtils::getFEEID(rdhOld) ||
        RDHUtils::getTriggerIR(rdhNew) != RDHUtils::getTriggerIR(rdhOld) ||
        RDHUtils::getHeartBeatIR(rdhNew) != RDHUtils::getHeartBeatIR(rdhOld) ||
        !(RDHUtils::getTriggerType(rdhNew) & RDHUtils::getTriggerType(rdhOld))) {
      return false;
    }
    return true;
  }

  //_____________________________________
  int decodeAlpideData(RUDecodeData& decData)
  {
    /// decode the ALPIDE data from the buffer of single lane

    auto* chipData = &decData.chipsData[0];

    decData.nChipsFired = decData.lastChipChecked = 0;
    int ntot = 0;
    for (int icab = 0; icab < decData.nCables; icab++) {
      auto& cableData = decData.cableData[icab];
      int res = 0;

#ifdef _RAW_READER_ERROR_CHECKS_
      auto& ruLinkStat = getGBTLink(decData.links[decData.cableLinkID[icab]])->statistics;

      // make sure the lane data starts with chip header or empty chip
      uint8_t h;
      if (cableData.current(h) && !mCoder.isChipHeaderOrEmpty(h)) {
        LOG(error) << "FEEId:" << OUTHEX(decData.ruInfo->idHW, 4) << " cable " << icab
                   << " data does not start with ChipHeader or ChipEmpty";
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrCableDataHeadWrong]++;
        RDHUtils::printRDH(reinterpret_cast<const o2::header::RAWDataHeader*>(getGBTLink(decData.links[decData.cableLinkID[icab]])->lastRDH));
      }
#endif
      auto cabHW = decData.cableHWID[icab];
      auto ri = decData.ruInfo;
      auto chIdGetter = [this, cabHW, ri](int cid) {
        return this->mMAP.getGlobalChipID(cid, cabHW, *ri);
      };
      while ((res = mCoder.decodeChip(*chipData, cableData, chIdGetter))) { // we register only chips with hits or errors flags set
        if (res > 0) {
#ifdef _RAW_READER_ERROR_CHECKS_
          // for the IB staves check if the cable ID is the same as the chip ID on the module
          if (mMAP.getName() == "ITS" && decData.ruInfo->ruType == 0) { // ATTENTION: this is a hack tailored for temporary check
            if (chipData->getChipID() != icab) {
              LOG(error) << "FEEId:" << OUTHEX(decData.ruInfo->idHW, 4) << " IB cable " << icab
                         << " shipped chip ID= " << chipData->getChipID();
              ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrIBChipLaneMismatch]++;
              RDHUtils::printRDH(reinterpret_cast<const o2::header::RAWDataHeader*>(getGBTLink(decData.links[decData.cableLinkID[icab]])->lastRDH));
            }
          }
#endif
          // convert HW chip id within the module to absolute chip id
          // chipData->setChipID(mMAP.getGlobalChipID(chipData->getChipID(), decData.cableHWID[icab], *decData.ruInfo));
          chipData->setInteractionRecord(mInteractionRecord);
          chipData->setTrigger(mTrigger);
          mDecodingStat.nNonEmptyChips++;
          mDecodingStat.nHitsDecoded += chipData->getData().size();
          ntot += res;
          // fetch next free chip
          if (++decData.nChipsFired < int(decData.chipsData.size())) {
            chipData = &decData.chipsData[decData.nChipsFired];
          } else {
            break; // last chip decoded
          }
        }
      }
    }
    return ntot;
  }

  //_____________________________________
  bool getNextChipData(ChipPixelData& chipData) override
  {
    /// read single chip data to the provided container

    if (mCurRUDecodeID >= 0) { // make sure current RU has fired chips to extract
      for (; mCurRUDecodeID < mNRUs; mCurRUDecodeID++) {
        auto& ru = mRUDecodeVec[mCurRUDecodeID];
        if (ru.lastChipChecked < ru.nChipsFired) {
          chipData.swap(ru.chipsData[ru.lastChipChecked++]);
          return true;
        }
      }
      mCurRUDecodeID = 0; // no more decoded data if reached this place,
    }

    // will need to decode new trigger
    if (!mDecodeNextAuto) { // no more data in the current ROF and no automatic decoding of next one was requested
      return false;
    }

    if (mMinTriggersCached < 2) { // last trigger might be incomplete, need to cache more data
      cacheLinksData(mRawBuffer);
    }
    if (mMinTriggersCached < 1 || !decodeNextTrigger()) {
      mCurRUDecodeID = -1;
      return false; // nothing left
    }
    return getNextChipData(chipData); // is it ok to use recursion here?
  }

  //_____________________________________
  void openInput(const std::string filename)
  {
    // open input for raw data decoding from file
    mSWIO.Stop();
    mSWIO.Start();
    clear(false); // do not reset statistics
    LOG(info) << "opening raw data input file " << filename;
    mIOFile.open(filename.c_str(), std::ifstream::binary);
    assert(mIOFile.good());
    mRawBuffer.clear();
    mRawBuffer.expand(RawBufferSize);
    mSWIO.Stop();
  }

  //_____________________________________
  size_t loadInput(PayLoadCont& buffer)
  {
    /// assure the buffers are large enough
    static_assert(RawBufferMargin > MaxGBTPacketBytes * 100 &&
                    RawBufferSize > 3 * RawBufferMargin,
                  "raw buffer size is too small");

    if (!mIOFile) {
      return 0;
    }
    if (buffer.getUnusedSize() > RawBufferMargin) { // bytes read but not used yet are enough
      return 0;
    }
    mSWIO.Start(false);
    auto readFromFile = [this](uint8_t* ptr, int n) {
      mIOFile.read(reinterpret_cast<char*>(ptr), n);
      return mIOFile.gcount(); // fread( ptr, sizeof(uint8_t), n, mIOFile);
    };
    auto nread = buffer.append(readFromFile);
    mSWIO.Stop();
    return nread;
  }

  // get statics of FEE with sequential idSW
  const GBTLinkDecodingStat* getGBTLinkDecodingStatSW(uint16_t idSW, int ruLink) const
  {
    if (mRUEntry[idSW] < 0 || ruLink >= RUDecodeData::MaxLinksPerRU || mRUDecodeVec[mRUEntry[idSW]].links[ruLink] < 0) {
      return nullptr;
    } else {
      return &getGBTLink(mRUDecodeVec[mRUEntry[idSW]].links[ruLink])->statistics;
    }
  }

  // get statics of FEE with given HW id
  const GBTLinkDecodingStat* getGBTLinkDecodingStatHW(uint16_t idHW, int ruLink) const
  {
    int idsw = mMAP.FEEId2RUSW(idHW);
    assert(idsw != 0xffff);
    return getGBTLinkDecodingStatSW(idsw, ruLink);
  }

  // aliases for BWD compatibility
  const GBTLinkDecodingStat* getRUDecodingStatSW(uint16_t idSW, int ruLink = 0) const { return getGBTLinkDecodingStatSW(idSW, ruLink); }
  const GBTLinkDecodingStat* getRUDecodingStatHW(uint16_t idHW, int ruLink = 0) const { return getGBTLinkDecodingStatHW(idHW, ruLink); }

  // get global decoding statistics
  const RawDecodingStat& getDecodingStat() const { return mDecodingStat; }

  void setVerbosity(int v) { mVerbose = v; }
  int getVerbosity() const { return mVerbose; }

  Mapping& getMapping() { return mMAP; }

  // get currently processed RU container
  const RUDecodeData* getCurrRUDecodeData() const { return mCurRUDecodeID < 0 ? nullptr : &mRUDecodeVec[mCurRUDecodeID]; }

  PayLoadCont& getRawBuffer() { return mRawBuffer; }

  // number of links seen in the data
  int getNLinks() const { return mNLinks; }

  // number of RUs seen in the data
  int getNRUs() const { return mNRUs; }

  // get vector of RU decode containers for RUs seen in the data
  const std::array<RUDecodeData, Mapping::getNRUs()>& getRUDecodeVec() const { return mRUDecodeVec; }

  const std::array<int, Mapping::getNRUs()>& getRUEntries() const { return mRUEntry; }

  // get RU decode container for RU with given SW ID
  const RUDecodeData* getRUDecode(int ruSW) const
  {
    return mRUEntry[ruSW] < 0 ? nullptr : &mRUDecodeVec[mRUEntry[ruSW]];
  }

  // get RU decode container for RU with given SW ID, if does not exist, create it
  RUDecodeData& getCreateRUDecode(int ruSW)
  {
    assert(ruSW < mMAP.getNRUs());
    if (mRUEntry[ruSW] < 0) {
      mRUEntry[ruSW] = mNRUs++;
      mRUDecodeVec[mRUEntry[ruSW]].ruInfo = mMAP.getRUInfoSW(ruSW); // info on the stave/RU
      mRUDecodeVec[mRUEntry[ruSW]].chipsData.resize(mMAP.getNChipsOnRUType(mMAP.getRUInfoSW(ruSW)->ruType));
      LOG(info) << "Defining container for RU " << ruSW << " at slot " << mRUEntry[ruSW];
    }
    return mRUDecodeVec[mRUEntry[ruSW]];
  }

  // create new gbt link
  int addGBTLink()
  {
    int sz = mGBTLinks.size();
    mGBTLinks.emplace_back();
    return sz;
  }

  // get the link
  GBTLink* getGBTLink(int i) { return i < 0 ? nullptr : &mGBTLinks[i]; }
  const GBTLink* getGBTLink(int i) const { return i < 0 ? nullptr : &mGBTLinks[i]; }

 private:
  std::ifstream mIOFile;
  Coder mCoder;
  Mapping mMAP;
  int mVerbose = 0;        //! verbosity level
  int mCurRUDecodeID = -1; //! index of currently processed RUDecode container

  PayLoadCont mRawBuffer; //! buffer for binary raw data file IO

  std::array<RUDecodeData, Mapping::getNRUs()> mRUDecodeVec; // decoding buffers for all active RUs
  std::array<int, Mapping::getNRUs()> mRUEntry;              //! entry of the RU with given SW ID in the mRUDecodeVec
  std::vector<GBTLink> mGBTLinks;
  int mNRUs = 0;   //! total number of RUs seen
  int mNLinks = 0; //! total number of GBT links seen

  //! min number of triggers to cache per link (keep this > N pages per CRU superpage)
  int mMinTriggersToCache = NCRUPagesPerSuperpage + 10;
  int mMinTriggersCached = 0; //! actual minimum (among different links) number of triggers to cache

  // statistics
  RawDecodingStat mDecodingStat; //! global decoding statistics

  TStopwatch mSWIO;    //! timer for IO operations
  TStopwatch mSWCache; //! timer for caching operations

  static constexpr int RawBufferMargin = 5000000;                      // keep uploaded at least this amount
  static constexpr int RawBufferSize = 10000000 + 2 * RawBufferMargin; // size in MB
  bool mPadding128 = true;                                             // is payload padded to 128 bits
  bool mImposeMaxPage = true;                                          // standard CRU data comes in 8KB pages
  // number of bytes the GBT word, including optional padding to 128 bits
  int mGBTWordSize = mPadding128 ? o2::itsmft::GBTPaddedWordLength : o2::itsmft::GBTWordLength;

  ClassDefOverride(RawPixelReader, 1);
};

template <class Mapping>
constexpr int RawPixelReader<Mapping>::RawBufferMargin;

template <class Mapping>
constexpr int RawPixelReader<Mapping>::RawBufferSize;

} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITS_RAWPIXELREADER_H */
