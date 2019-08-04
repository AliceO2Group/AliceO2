// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DetectorsBase/Triggers.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/PayLoadSG.h"
#include <TTree.h>
#include <TStopwatch.h>
#include <FairLogger.h>
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

constexpr int MaxLinksPerRU = 3;            // max number of GBT links per RU
constexpr int MaxCablesPerRU = 28;          // max number of cables RU can readout
constexpr int MaxChipsPerRU = 196;          // max number of chips the RU can readout
constexpr int MaxGBTPacketBytes = 8 * 1024; // Max size of GBT packet in bytes (8KB)
constexpr int NCRUPagesPerSuperpage = 256;  // Expected max number of CRU pages per superpage

struct GBTLinkDecodingStat {
  // Statisting for per-link decoding
  // counters for format checks
  enum DecErrors : int {
    ErrPageCounterDiscontinuity, // RDH page counters for the same RU/trigger are not continuous
    ErrRDHvsGBTHPageCnt,         // RDH ang GBT header page counters are not consistent
    ErrMissingGBTHeader,         // GBT payload header was expected but not foun
    ErrMissingGBTTrailer,        // GBT payload trailer was expected but not found
    ErrNonZeroPageAfterStop,     // all lanes were stopped but the page counter in not 0
    ErrUnstoppedLanes,           // end of FEE data reached while not all lanes received stop
    ErrDataForStoppedLane,       // data was received for stopped lane
    ErrNoDataForActiveLane,      // no data was seen for lane (which was not in timeout)
    ErrIBChipLaneMismatch,       // chipID (on module) was different from the lane ID on the IB stave
    ErrCableDataHeadWrong,       // cable data does not start with chip header or empty chip
    ErrInvalidActiveLanes,       // active lanes pattern conflicts with expected for given RU type
    ErrPacketCounterJump,        // jump in RDH.packetCounter
    NErrorsDefined
  };
  uint32_t ruLinkID = 0; // Link ID within RU

  // Note: packet here is meant as a group of CRU pages belonging to the same trigger
  uint32_t nPackets = 0;                                                   // total number of packets
  std::array<int, NErrorsDefined> errorCounts = {};                        // error counters
  std::array<int, GBTDataTrailer::MaxStateCombinations> packetStates = {}; // packet status from the trailer

  //_____________________________________________________
  void clear()
  {
    nPackets = 0;
    errorCounts.fill(0);
    packetStates.fill(0);
  }

  //_____________________________________________________
  void print(bool skipEmpty = true) const
  {
    int nErr = 0;
    for (int i = NErrorsDefined; i--;) {
      nErr += errorCounts[i];
    }
    printf("GBTLink#0x%d Packet States Statistics (total packets: %d)\n", ruLinkID, nPackets);
    for (int i = 0; i < GBTDataTrailer::MaxStateCombinations; i++) {
      if (packetStates[i]) {
        std::bitset<GBTDataTrailer::NStatesDefined> patt(i);
        printf("counts for triggers B[%s] : %d\n", patt.to_string().c_str(), packetStates[i]);
      }
    }
    printf("Decoding errors: %d\n", nErr);
    for (int i = 0; i < NErrorsDefined; i++) {
      if (!skipEmpty || errorCounts[i]) {
        printf("%-70s: %d\n", ErrNames[i].data(), errorCounts[i]);
      }
    }
  }

  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "RDH page counters for the same RU/trigger are not continuous",      // ErrPageCounterDiscontinuity
    "RDH ang GBT header page counters are not consistent",               // ErrRDHvsGBTHPageCnt
    "GBT payload header was expected but not found",                     // ErrMissingGBTHeader
    "GBT payload trailer was expected but not found",                    // ErrMissingGBTTrailer
    "All lanes were stopped but the page counter in not 0",              // ErrNonZeroPageAfterStop
    "End of FEE data reached while not all lanes received stop",         // ErrUnstoppedLanes
    "Data was received for stopped lane",                                // ErrDataForStoppedLane
    "No data was seen for lane (which was not in timeout)",              // ErrNoDataForActiveLane
    "ChipID (on module) was different from the lane ID on the IB stave", // ErrIBChipLaneMismatch
    "Cable data does not start with chip header or empty chip",          // ErrCableDataHeadWrong
    "Active lanes pattern conflicts with expected for given RU type",    // ErrInvalidActiveLanes
    "Jump in RDH_packetCounter"                                          // ErrPacketCounterJump
  };

  ClassDefNV(GBTLinkDecodingStat, 1);
};

constexpr std::array<std::string_view, GBTLinkDecodingStat::NErrorsDefined> GBTLinkDecodingStat::ErrNames;

struct RawDecodingStat {
  enum DecErrors : int {
    ErrInvalidFEEId, // RDH provided invalid FEEId
    NErrorsDefined
  };

  using ULL = unsigned long long;
  uint64_t nPagesProcessed = 0;                     // total number of pages processed
  uint64_t nRUsProcessed = 0;                       // total number of RUs processed (1 RU may take a few pages)
  uint64_t nBytesProcessed = 0;                     // total number of bytes (rdh->memorySize) processed
  uint64_t nNonEmptyChips = 0;                      // number of non-empty chips found
  uint64_t nHitsDecoded = 0;                        // number of hits found
  std::array<int, NErrorsDefined> errorCounts = {}; // error counters

  RawDecodingStat() = default;

  void clear()
  {
    nPagesProcessed = 0;
    nRUsProcessed = 0;
    nBytesProcessed = 0;
    nNonEmptyChips = 0;
    nHitsDecoded = 0;
    errorCounts.fill(0);
  }

  void print(bool skipEmpty = true) const
  {
    printf("\nDecoding statistics\n");
    printf("%llu bytes for %llu RUs processed in %llu pages\n", (ULL)nBytesProcessed, (ULL)nRUsProcessed, (ULL)nPagesProcessed);
    printf("%llu hits found in %llu non-empty chips\n", (ULL)nHitsDecoded, (ULL)nNonEmptyChips);
    int nErr = 0;
    for (int i = NErrorsDefined; i--;) {
      nErr += errorCounts[i];
    }
    printf("Decoding errors: %d\n", nErr);
    for (int i = 0; i < NErrorsDefined; i++) {
      if (!skipEmpty || errorCounts[i]) {
        printf("%-70s: %d\n", ErrNames[i].data(), errorCounts[i]);
      }
    }
  }

  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "RDH cointains invalid FEEID" // ErrInvalidFEEId
  };

  ClassDefNV(RawDecodingStat, 1);
};

// support for the GBT single link data
struct GBTLink {
  PayLoadCont data;     // data buffer per link
  int lastPageSize = 0; // size of last added page = offset from the end to get to the RDH
  int nTriggers = 0;    // number of triggers loaded (the last one might be incomplete)
  uint32_t lanes = 0;   // lanes served by this link
  // transient data filled from current RDH
  uint32_t lanesActive = 0;   // lanes declared by the payload header
  uint32_t lanesStop = 0;     // lanes received stop in the payload trailer
  uint32_t lanesTimeOut = 0;  // lanes received timeout
  uint32_t lanesWithData = 0; // lanes with data transmitted
  int32_t packetCounter = -1; // current packet counter from RDH (RDH.packetCounter)
  const o2::header::RAWDataHeader* lastRDH = nullptr;
  GBTLinkDecodingStat statistics; // decoding statistics

  void clear(bool resetStat = true)
  {
    data.clear();
    lastPageSize = 0;
    nTriggers = 0;
    lanes = 0;
    lanesActive = lanesStop = lanesTimeOut = lanesWithData = 0;
    lastRDH = nullptr;
    if (resetStat) {
      statistics.clear();
    }
  }
};

struct RUDecodeData {
  std::array<PayLoadCont, MaxCablesPerRU> cableData;              // cable data in compressed ALPIDE format
  std::array<o2::itsmft::ChipPixelData, MaxChipsPerRU> chipsData; // fully decoded data
  std::array<std::unique_ptr<GBTLink>, MaxLinksPerRU> links;      // data + counters for links of this RU
  std::array<uint8_t, MaxCablesPerRU> cableHWID;                  // HW ID of cable whose data is in the corresponding slot of cableData
  std::array<uint8_t, MaxCablesPerRU> cableLinkID;                // ID of the GBT link transmitting this cable data (for error stat. only)

  int nCables = 0;         // total number of cables decoded for single trigger
  int nChipsFired = 0;     // number of chips with data or with errors
  int lastChipChecked = 0; // last chips checked among nChipsFired
  const RUInfo* ruInfo = nullptr;

  RUDecodeData() = default;
  //  RUDecodeData(const RUDecodeData& src) {}; // dummy?

  void clear(bool resetStat = true)
  {
    clearTrigger();
    nChipsFired = 0;
    for (int i = 0; i < MaxLinksPerRU; i++) {
      auto* link = links[i].get();
      if (link) {
        link->clear(resetStat);
      }
    }
  }

  void clearTrigger()
  {
    for (int i = nCables; i--;) {
      cableData[i].clear();
    }
    nCables = 0;
  }
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
    LOG(INFO) << "Cleaning decoder, reset_statistics_flag " << resetStat;
    if (resetStat) {
      mDecodingStat.clear();
    }
    for (auto& rudec : mRUDecodeVec) {
      rudec.clear(resetStat);
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
        for (int il = 0; il < MaxLinksPerRU; il++) {
          nLinks += ruData.links[il] ? 1 : 0;
        }
        mNLinks += nLinks;
        if (!nLinks) {
          LOG(INFO) << "Imposing single link readout for RU " << int(ru);
          ruData.links[0] = std::make_unique<GBTLink>();
          ruData.links[0]->lanes = mMAP.getCablesOnRUType(ruData.ruInfo->ruType);
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
    ruData.cableHWID[chip.cableSW] = chip.cableHW; // register the cable HW ID

    auto& pixels = chipData.getData();
    std::sort(pixels.begin(), pixels.end(),
              [](auto lhs, auto rhs) {
                if (lhs.getRow() < rhs.getRow())
                  return true;
                if (lhs.getRow() > rhs.getRow())
                  return false;
                return lhs.getCol() < rhs.getCol();
              });
    ruData.cableData[chip.cableSW].ensureFreeCapacity(40 * (2 + pixels.size())); // make sure buffer has enough capacity
    mCoder.encodeChip(ruData.cableData[chip.cableSW], chipData, chip.chipOnModuleHW, mInteractionRecord.bc);
  }

  //______________________________________________________
  void convertEmptyChips(int fromChip, int uptoChip)
  {
    // add empty chip words to respective cable's buffers for all chips of the current RU container
    auto& ruData = mRUDecodeVec[mCurRUDecodeID];                     // current RU container
    for (int chipIDSW = fromChip; chipIDSW < uptoChip; chipIDSW++) { // flag chips w/o data
      const auto& chip = *mMAP.getChipOnRUInfo(ruData.ruInfo->ruType, chipIDSW);
      ruData.cableHWID[chip.cableSW] = chip.cableHW; // register the cable HW ID
      ruData.cableData[chip.cableSW].ensureFreeCapacity(100);
      mCoder.addEmptyChip(ruData.cableData[chip.cableSW], chip.chipOnModuleHW, mInteractionRecord.bc);
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
    rdh.triggerOrbit = rdh.heartbeatOrbit = mInteractionRecord.orbit;
    rdh.triggerBC = rdh.heartbeatBC = mInteractionRecord.bc;
    rdh.triggerType = o2::trigger::PhT; // ??
    rdh.detectorField = mMAP.getRUDetectorField();
    rdh.blockLength = 0xffff; // ITS keeps this dummy

    int maxGBTWordsPerPacket = (MaxGBTPacketBytes - rdh.headerSize) / o2::itsmft::GBTPaddedWordLength - 2;

    int nGBTW[MaxLinksPerRU] = {0};
    for (int il = 0; il < MaxLinksPerRU; il++) {
      auto link = ruData.links[il].get();
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
      rdh.feeId = mMAP.RUSW2FEEId(ruData.ruInfo->idSW, il); // write on link 0 always
      rdh.linkID = il;
      rdh.pageCnt = 0;
      rdh.stop = 0;
      rdh.memorySize = rdh.headerSize + (nGBTWordsNeeded + 2) * mGBTWordSize; // update remaining size
      if (rdh.memorySize > MaxGBTPacketBytes) {
        rdh.memorySize = MaxGBTPacketBytes;
      }
      rdh.offsetToNext = mImposeMaxPage ? MaxGBTPacketBytes : rdh.memorySize;

      link->data.ensureFreeCapacity(MaxGBTPacketBytes);
      link->data.addFast(reinterpret_cast<uint8_t*>(&rdh), rdh.headerSize); // write RDH for current packet
      link->nTriggers++;                                                    // acknowledge the page, note: here we count pages, not triggers
      o2::itsmft::GBTDataHeader gbtHeader(0, link->lanes);
      o2::itsmft::GBTDataTrailer gbtTrailer; // lanes will be set on closing the last page

      gbtHeader.setPacketID(rdh.pageCnt);
      link->data.addFast(gbtHeader.getW8(), mGBTWordSize); // write GBT header for current packet
      if (mVerbose) {
        LOG(INFO) << "Filling RU data";
        printRDH(&rdh);
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
          rdh.pageCnt++;                                     // flag new page
          rdh.stop = nGBTWordsNeeded < maxGBTWordsPerPacket; // flag if this is the last packet of multi-packet
          rdh.blockLength = 0xffff;                          // (nGBTWordsNeeded % maxGBTWordsPerPacket + 2) * mGBTWordSize; // record payload size
          // update remaining size, using padded GBT words (as CRU writes)
          rdh.memorySize = rdh.headerSize + (nGBTWordsNeeded + 2) * o2::itsmft::GBTPaddedWordLength;
          if (rdh.memorySize > MaxGBTPacketBytes) {
            rdh.memorySize = MaxGBTPacketBytes;
          }
          rdh.offsetToNext = mImposeMaxPage ? MaxGBTPacketBytes : rdh.memorySize;
          link->data.ensureFreeCapacity(MaxGBTPacketBytes);
          link->data.addFast(reinterpret_cast<uint8_t*>(&rdh), rdh.headerSize); // write RDH for current packet
          link->nTriggers++;                                                    // acknowledge the page, note: here we count pages, not triggers
          if (mVerbose) {
            printRDH(&rdh);
          }
          gbtHeader.setPacketID(rdh.pageCnt);
          link->data.addFast(gbtHeader.getW8(), mGBTWordSize); // write GBT header for current packet
          if (mVerbose) {
            gbtHeader.printX(mPadding128);
          }
          nGBTWordsInPacket = 0; // reset counter of words in the packet
        }
      } while (nGBTWordsNeeded);

      gbtTrailer.setLanesStop(link->lanes);
      gbtTrailer.setPacketState(0x1 << GBTDataTrailer::PacketDone);
      link->data.addFast(gbtTrailer.getW8(), mGBTWordSize); // write GBT trailer for the last packet
      if (mVerbose) {
        gbtTrailer.printX(mPadding128);
      }
      // NOTE: here we don't pad the page to 8KB, will do this when flushing everything to the sink

      if (minPages > link->nTriggers) {
        minPages = link->nTriggers;
      }

    } // loop over links of RU
    ruData.clearTrigger();
    ruData.nChipsFired = 0;
    return minPages == dummyNPages ? 0 : minPages;
  }

  //___________________________________________________________________________________
  int flushSuperPages(int maxPages, PayLoadCont& sink)
  {
    // flush superpage (at most maxPages) of each link to the output,
    // return total number of pages flushed

    int totPages = 0;
    for (int ru = 0; ru < mMAP.getNRUs(); ru++) {
      auto* ruData = getRUDecode(ru);
      if (!ruData) {
        continue;
      }
      for (int il = 0; il < MaxLinksPerRU; il++) {
        auto link = ruData->links[il].get();
        if (!link || link->data.isEmpty()) {
          continue;
        }
        int nPages = 0;
        sink.ensureFreeCapacity(maxPages * MaxGBTPacketBytes);
        const auto* ptrIni = link->data.getPtr();
        while (nPages < maxPages && !link->data.isEmpty()) {
          const auto ptr = link->data.getPtr();
          o2::header::RAWDataHeader* rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);
          sink.addFast(ptr, rdh->memorySize);                    // copy header + payload
          sink.fillFast(0, MaxGBTPacketBytes - rdh->memorySize); // complete with 0's till the end of the page
          link->data.setPtr(ptr + rdh->memorySize);
          link->nTriggers--; // here we count pages, not triggers
          nPages++;
        }
        totPages += nPages;
        link->data.moveUnusedToHead();
      } // loop over links
    }   // loop over RUs
    return totPages;
  }

  ///================================== Decoding methods ========================

  //_____________________________________________________________________________
  void printRDH(const o2::header::RAWDataHeader* h)
  {
    if (!h) {
      printf("Provided RDH pointer is null\n");
      return;
    }
    printf("RDH| Ver:%2u Hsz:%2u Blgt:%4u FEEId:0x%04x PBit:%u\n",
           uint32_t(h->version), uint32_t(h->headerSize), uint32_t(h->blockLength), uint32_t(h->feeId), uint32_t(h->priority));
    printf("RDH|[CRU: Offs:%5u Msz:%4u LnkId:0x%02x Packet:%3u CRUId:0x%04x]\n",
           uint32_t(h->offsetToNext), uint32_t(h->memorySize), uint32_t(h->linkID), uint32_t(h->packetCounter), uint32_t(h->cruID));
    printf("RDH| TrgOrb:%9u HBOrb:%9u TrgBC:%4u HBBC:%4u TrgType:%u\n",
           uint32_t(h->triggerOrbit), uint32_t(h->heartbeatOrbit), uint32_t(h->triggerBC), uint32_t(h->heartbeatBC),
           uint32_t(h->triggerType));
    printf("RDH| DetField:0x%05x Par:0x%04x Stop:0x%04x PageCnt:%5u\n",
           uint32_t(h->detectorField), uint32_t(h->par), uint32_t(h->stop), uint32_t(h->pageCnt));
  }

  //_____________________________________
  size_t cacheLinksData(PayLoadCont& buffer)
  {
    // distribute data from the single buffer among the links caches

    LOG(INFO) << "Caching links data, currently in cache: " << mMinTriggersCached << " triggers";
    auto nRead = loadInput(buffer);
    if (buffer.isEmpty()) {
      return nRead;
    }
    enum LinkFlag : int8_t { NotUpdated,
                             Updated,
                             HasEnoughTriggers };
    LinkFlag linkFlags[ChipMappingITS::getNRUs()][3] = {NotUpdated}; // flag that enough triggeres were loaded for this link
    int nLEnoughTriggers = 0;                                        // number of links for we which enough number of triggers were loaded
    auto ptr = buffer.getPtr();
    o2::header::RAWDataHeader* rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);

    do {
      if (!isRDHHeuristic(rdh)) {   // does it look like RDH?
        if (!findNextRDH(buffer)) { // try to recover the pointer
          break;                    // no data to continue
        }
        ptr = buffer.getPtr();
        rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);
      }
      if (mVerbose) {
        printRDH(rdh);
      }

      int ruIDSW = mMAP.FEEId2RUSW(rdh->feeId);
#ifdef _RAW_READER_ERROR_CHECKS_
      if (ruIDSW >= mMAP.getNRUs()) {
        mDecodingStat.errorCounts[RawDecodingStat::ErrInvalidFEEId]++;
        LOG(ERROR) << mDecodingStat.ErrNames[RawDecodingStat::ErrInvalidFEEId]
                   << " : FEEId:" << OUTHEX(rdh->feeId, 4) << ", skipping CRU page";
        printRDH(rdh);
        ptr += rdh->offsetToNext;
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
      mMAP.expandFEEId(rdh->feeId, lr, ruOnLr, linkIDinRU);
      auto link = ruDecode.links[linkIDinRU].get();
      if (link) {                                                                                                    // was there any data seen on this link before?
        const auto rdhPrev = reinterpret_cast<o2::header::RAWDataHeader*>(link->data.getEnd() - link->lastPageSize); // last stored RDH
        if (isSameRUandTrigger(rdhPrev, rdh)) {
          newTrigger = false;
        }
      } else { // a new link was added
        LOG(INFO) << "Adding new GBT LINK FEEId:" << OUTHEX(rdh->feeId, 4);
        ruDecode.links[linkIDinRU] = std::make_unique<GBTLink>();
        link = ruDecode.links[linkIDinRU].get();
        link->statistics.ruLinkID = linkIDinRU;
        mNLinks++;
      }
      if (linkFlags[ruIDSW][linkIDinRU] == NotUpdated) {
        link->data.moveUnusedToHead(); // reuse space of already processed data
        linkFlags[ruIDSW][linkIDinRU] = Updated;
      }
      // copy data to the buffer of the link and memorize its RDH pointer
      link->data.add(ptr, rdh->memorySize);
      link->lastPageSize = rdh->memorySize; // account new added size
      auto rdhC = reinterpret_cast<o2::header::RAWDataHeader*>(link->data.getEnd() - link->lastPageSize);
      rdhC->offsetToNext = rdh->memorySize; // since we skip 0-s, we have to modify the offset

      if (newTrigger) {
        link->nTriggers++; // acknowledge 1st trigger
        if (link->nTriggers >= mMinTriggersToCache && linkFlags[ruIDSW][linkIDinRU] != HasEnoughTriggers) {
          nLEnoughTriggers++;
          linkFlags[ruIDSW][linkIDinRU] = HasEnoughTriggers;
        }
      }

      ptr += rdh->offsetToNext;
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
        for (auto& link : ruDecData.links) {
          if (link && link->nTriggers < mMinTriggersCached) {
            mMinTriggersCached = link->nTriggers;
          }
        }
      }
    }
    LOG(INFO) << "Cached at least " << mMinTriggersCached << " triggers on " << mNLinks << " links of " << mNRUs << " RUs";

    return nRead;
  }

  //_____________________________________
  int decodeNextTrigger()
  {
    // Decode next trigger from the cached links data and decrease cached triggers counter, return N links decoded
    if (mMinTriggersCached < 1) {
      return 0;
    }
    int nlinks = 0;
    for (int ir = mNRUs; ir--;) {
      auto& ruDecode = mRUDecodeVec[ir];
      if (!nlinks) {                        // on 1st occasion extract trigger data
        for (auto& link : ruDecode.links) { // loop over links to fill cable buffers
          if (link && !link->data.isEmpty()) {
            const auto rdh = reinterpret_cast<const o2::header::RAWDataHeader*>(link->data.getPtr());
            mInteractionRecord.bc = rdh->triggerBC;
            mInteractionRecord.orbit = rdh->triggerOrbit;
            mTrigger = rdh->triggerType;
            mInteractionRecordHB.bc = rdh->heartbeatBC;
            mInteractionRecordHB.orbit = rdh->heartbeatOrbit;
            break;
          }
        }
      }

      nlinks += decodeNextRUData(ruDecode);
      mDecodingStat.nRUsProcessed++;
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
    ruDecData.clearTrigger();
    bool aborted = false;
    for (auto& link : ruDecData.links) { // loop over links to fill cable buffers
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
      ptr += o2::itsmft::GBTPaddedWordLength;
      buffer.setPtr(ptr);
      if (!buffer.isEmpty()) {
        rdh = reinterpret_cast<o2::header::RAWDataHeader*>(ptr);
      } else {
        break;
      }
    } while (!(goodRDH = isRDHHeuristic(rdh)));
    LOG(INFO) << "End of pointer recovery after skipping " << scan << " GBT words, RDH is"
              << (goodRDH ? "" : " not") << " found";
    return goodRDH;
  }

  //_____________________________________
  bool isRDHHeuristic(const o2::header::RAWDataHeader* rdh)
  {
    /// heuristic check if this is indeed an RDH
    return (!rdh || rdh->headerSize != sizeof(o2::header::RAWDataHeader) || rdh->zero0 != 0 ||
            rdh->zero41 != 0 || rdh->zero42 != 0 || rdh->word5 != 0 || rdh->zero6 != 0)
             ? false
             : true;
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
    auto rdh = reinterpret_cast<const o2::header::RAWDataHeader*>(raw);

#ifdef _RAW_READER_ERROR_CHECKS_
    if (!isRDHHeuristic(rdh)) {
      LOG(ERROR) << "Page does not start with RDH";
      printRDH(rdh);
      for (int i = 0; i < 4; i++) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw + i * 16);
        gbtD->printX(mPadding128);
      }
      raw += mGBTWordSize;
      aborted = true;
      return raw;
    }
#endif

    int ruIDSW = mMAP.FEEId2RUSW(rdh->feeId);
#ifdef _RAW_READER_ERROR_CHECKS_
    if (ruIDSW >= mMAP.getNRUs()) {
      mDecodingStat.errorCounts[RawDecodingStat::ErrInvalidFEEId]++;
      LOG(ERROR) << mDecodingStat.ErrNames[RawDecodingStat::ErrInvalidFEEId]
                 << " : FEEId:" << OUTHEX(rdh->feeId, 4) << ", skipping CRU page";
      printRDH(rdh);
      raw += rdh->offsetToNext;
      return raw;
    }

    if (ruIDSW != ruDecData.ruInfo->idSW) { // should not happen with cached data
      LOG(ERROR) << "RDG RU IDSW " << ruIDSW << " differs from expected " << ruDecData.ruInfo->idSW;
      printRDH(rdh);
    }
#endif

    uint16_t lr, ruOnLr, linkIDinRU;
    mMAP.expandFEEId(rdh->feeId, lr, ruOnLr, linkIDinRU);
    auto ruLink = ruDecData.links[linkIDinRU].get();
    auto& ruLinkStat = ruLink->statistics;
    ruLink->lastRDH = rdh;
    ruLinkStat.nPackets++;

#ifdef _RAW_READER_ERROR_CHECKS_
    if (rdh->packetCounter > ruLink->packetCounter + 1) {
      ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump]++;
      LOG(ERROR) << ruLinkStat.ErrNames[GBTLinkDecodingStat::ErrPacketCounterJump]
                 << " : FEEId:" << OUTHEX(rdh->feeId, 4) << ": jump from " << int(ruLink->packetCounter)
                 << " to " << int(rdh->packetCounter);
      printRDH(rdh);
    }
#endif

    ruDecData.nCables = ruDecData.ruInfo->nCables;
    while (1) {
      ruLink->packetCounter = rdh->packetCounter;

      mDecodingStat.nBytesProcessed += rdh->memorySize;
      mDecodingStat.nPagesProcessed++;
      raw += rdh->headerSize;
      int nGBTWords = (rdh->memorySize - rdh->headerSize) / mGBTWordSize - 2; // number of GBT words excluding header/trailer
      auto gbtH = reinterpret_cast<const o2::itsmft::GBTDataHeader*>(raw);    // process GBT header

#ifdef _RAW_READER_ERROR_CHECKS_
      if (mVerbose) {
        printRDH(rdh);
        gbtH->printX(mPadding128);
        LOG(INFO) << "Expect " << nGBTWords << " GBT words";
      }

      if (!gbtH->isDataHeader()) {
        gbtH->printX(mPadding128);
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " GBT payload header was expected, abort page decoding";
        printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
        gbtH->printX(mPadding128);
        aborted = true;
        return raw;
      }

      if (gbtH->getPacketID() != rdh->pageCnt) {
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Different GBT header " << gbtH->getPacketID()
                   << " and RDH page " << rdh->pageCnt << " counters";
        printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt]++;
      }

      if (ruLink->lanesActive == ruLink->lanesStop) { // all lanes received their stop, new page 0 expected
        if (rdh->pageCnt) {                           // flag lanes of this FEE
          LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Non-0 page counter (" << rdh->pageCnt << ") while all lanes were stopped";
          printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]++;
        }
      }

      ruLink->lanesActive = gbtH->getLanes(); // TODO do we need to update this for every page?

      if (~(mMAP.getCablesOnRUType(ruDecData.ruInfo->ruType)) & ruLink->lanesActive) { // are there wrong lanes?
        std::bitset<32> expectL(mMAP.getCablesOnRUType(ruDecData.ruInfo->ruType)), gotL(ruLink->lanesActive);
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Active lanes pattern " << gotL
                   << " conflicts with expected " << expectL << " for given RU type, skip page";
        printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrInvalidActiveLanes]++;
        raw = ((uint8_t*)rdh) + rdh->offsetToNext; // jump to the next packet
        return raw;
      }

      if (!rdh->pageCnt) { // reset flags
        ruLink->lanesStop = 0;
        ruLink->lanesWithData = 0;
      }

#endif
      raw += mGBTWordSize;
      for (int iw = 0; iw < nGBTWords; iw++, raw += mGBTWordSize) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw);
        // TODO: need to clarify if the nGBTWords from the rdh->memorySize is reliable estimate of the real payload, at the moment this is not the case

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
        ruDecData.cableLinkID[cableSW] = linkIDinRU;
        ruLink->lanesWithData |= 0x1 << cableSW;    // flag that the data was seen on this lane
        if (ruLink->lanesStop & (0x1 << cableSW)) { // make sure stopped lanes do not transmit the data
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane]++;
          LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Data received for stopped lane " << cableHW << " (sw:" << cableSW << ")";
          printRDH(rdh);
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
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << std::dec
                   << " GBT payload trailer was expected, abort page decoding NW" << nGBTWords;
        printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer]++;
        aborted = true;
        return raw;
      }

      ruLink->lanesTimeOut |= gbtT->getLanesTimeout(); // register timeouts
      ruLink->lanesStop |= gbtT->getLanesStop();       // register stops
#endif
      raw += mGBTWordSize;
      // we finished the GBT page, see if there is a continuation and if it belongs to the same multipacket

      if (!rdh->offsetToNext) { // RS TODO: what the last page in memory will contain as offsetToNext, is it 0?
        break;
      }

      raw = ((uint8_t*)rdh) + rdh->offsetToNext; // jump to the next packet:
      auto rdhN = reinterpret_cast<const o2::header::RAWDataHeader*>(raw);
      // check if data of given RU are over, i.e. we the page counter was wrapped to 0 (should be enough!) or other RU/trigger started
      if (!isSameRUandTrigger(rdh, rdhN)) {

#ifdef _RAW_READER_ERROR_CHECKS_
        // make sure all lane stops for finished page are received
        if ((ruLink->lanesActive & ~ruLink->lanesStop) && nGBTWords) {
          if (rdh->triggerType != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
            std::bitset<32> active(ruLink->lanesActive), stopped(ruLink->lanesStop);
            LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " end of FEE data but not all lanes received stop"
                       << "| active: " << active << " stopped: " << stopped;
            printRDH(rdh);
            ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes]++;
          }
        }

        // make sure all active lanes (except those in time-out) have sent some data
        if ((~ruLink->lanesWithData & ruLink->lanesActive) != ruLink->lanesTimeOut && nGBTWords) {
          std::bitset<32> withData(ruLink->lanesWithData), active(ruLink->lanesActive), timeOut(ruLink->lanesTimeOut);
          LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Lanes not in time-out but not sending data"
                     << "\n| with data: " << withData << " active: " << active << " timeOut: " << timeOut;
          printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane]++;
        }
#endif
        // accumulate packet states
        ruLinkStat.packetStates[gbtT->getPacketState()]++;

        break;
      }
#ifdef _RAW_READER_ERROR_CHECKS_
      // check if the page counter increases
      if (rdhN->pageCnt != rdh->pageCnt + 1) {
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Discontinuity in the RDH page counter of the same RU trigger: old "
                   << rdh->pageCnt << " new: " << rdhN->pageCnt;
        printRDH(rdh);
        printRDH(rdhN);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPageCounterDiscontinuity]++;
      }
#endif
      rdh = rdhN;
      ruLink->lastRDH = rdh;
    }

#ifdef _RAW_READER_ERROR_CHECKS_
//    if (rdh->pageCnt && !rdh->stop) {
//      LOG(WARNING) << "Last packet(" << rdh->pageCnt << ") of GBT multi-packet is reached w/o STOP set in the RDH";
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
    auto rdh = reinterpret_cast<const o2::header::RAWDataHeader*>(raw);
#ifdef _RAW_READER_ERROR_CHECKS_
    if (!isRDHHeuristic(rdh)) {
      LOG(ERROR) << "Page does not start with RDH";
      printRDH(rdh);
      for (int i = 0; i < 4; i++) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw + i * 16);
        gbtD->printX(mPadding128);
      }
      aborted = true;
      return raw;
    }
    int ruIDSWD = mMAP.FEEId2RUSW(rdh->feeId);
    if (ruIDSWD >= mMAP.getNRUs()) {
      mDecodingStat.errorCounts[RawDecodingStat::ErrInvalidFEEId]++;
      LOG(ERROR) << mDecodingStat.ErrNames[RawDecodingStat::ErrInvalidFEEId]
                 << " : FEEId:" << OUTHEX(rdh->feeId, 4) << ", skipping CRU page";
      printRDH(rdh);
      raw += rdh->offsetToNext;
      return raw;
    }
#endif
    uint16_t lr, ruOnLr, linkIDinRU;
    mMAP.expandFEEId(rdh->feeId, lr, ruOnLr, linkIDinRU);
    int ruIDSW = mMAP.FEEId2RUSW(rdh->feeId);
    auto& ruDecode = getCreateRUDecode(ruIDSW);
    auto ruInfo = mMAP.getRUInfoSW(ruIDSW);

    if (!ruDecode.links[linkIDinRU].get()) {
      ruDecode.links[linkIDinRU] = std::make_unique<GBTLink>();
      ruDecode.links[linkIDinRU].get()->statistics.ruLinkID = linkIDinRU;
      mNLinks++;
    }

    mInteractionRecord.bc = rdh->triggerBC;
    mInteractionRecord.orbit = rdh->triggerOrbit;

    mTrigger = rdh->triggerType;

    mInteractionRecordHB.bc = rdh->heartbeatBC;
    mInteractionRecordHB.orbit = rdh->heartbeatOrbit;

    auto ruLink = ruDecode.links[linkIDinRU].get();
    auto& ruLinkStat = ruLink->statistics;
    ruLink->lastRDH = rdh;
    ruLinkStat.nPackets++;

#ifdef _RAW_READER_ERROR_CHECKS_
    if (rdh->packetCounter > ruLink->packetCounter + 1) {
      ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump]++;
      LOG(ERROR) << ruLinkStat.ErrNames[GBTLinkDecodingStat::ErrPacketCounterJump]
                 << " : FEEId:" << OUTHEX(rdh->feeId, 4) << ": jump from " << int(ruLink->packetCounter)
                 << " to " << int(rdh->packetCounter);
      printRDH(rdh);
    }
#endif
    ruLink->packetCounter = rdh->packetCounter;

    int sizeAtEntry = outBuffer.getSize(); // save the size of outbuffer size at entry, in case of severe error we will need to rewind to it.

    while (1) {
      mDecodingStat.nPagesProcessed++;
      mDecodingStat.nBytesProcessed += rdh->memorySize;
      raw += rdh->headerSize;
      // number of 128 b GBT words excluding header/trailer
      int nGBTWords = (rdh->memorySize - rdh->headerSize) / o2::itsmft::GBTPaddedWordLength - 2;
      auto gbtH = reinterpret_cast<const o2::itsmft::GBTDataHeader*>(raw); // process GBT header

#ifdef _RAW_READER_ERROR_CHECKS_
      if (mVerbose) {
        printRDH(rdh);
        gbtH->printX(true);
        LOG(INFO) << "Expect " << nGBTWords << " GBT words";
      }
      if (!gbtH->isDataHeader()) {
        gbtH->printX(true);
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " GBT payload header was expected, abort page decoding";
        printRDH(rdh);
        gbtH->printX(true);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
        aborted = true;
        outBuffer.shrinkToSize(sizeAtEntry); // reset output buffer to initial state
        return raw;
      }
      if (gbtH->getPacketID() != rdh->pageCnt) {
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Different GBT header " << gbtH->getPacketID()
                   << " and RDH page " << rdh->pageCnt << " counters";
        printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt]++;
      }

      if (ruLink->lanesActive == ruLink->lanesStop) { // all lanes received their stop, new page 0 expected
        if (rdh->pageCnt) {                           // flag lanes of this FEE
          LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Non-0 page counter (" << rdh->pageCnt << ") while all lanes were stopped";
          printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]++;
        }
      }

      ruLink->lanesActive = gbtH->getLanes(); // TODO do we need to update this for every page?

      if (!rdh->pageCnt) { // reset flags
        ruLink->lanesStop = 0;
        ruLink->lanesWithData = 0;
      }

#endif
      // start writting skimmed data for this page, making sure the buffer has enough free slots
      outBuffer.ensureFreeCapacity(8 * 1024);
      auto rdhS = reinterpret_cast<o2::header::RAWDataHeader*>(outBuffer.getEnd()); // save RDH and make saved copy editable
      outBuffer.addFast(reinterpret_cast<const uint8_t*>(rdh), rdh->headerSize);

      outBuffer.addFast(reinterpret_cast<const uint8_t*>(gbtH), mGBTWordSize); // save gbt header w/o 128b padding

      raw += o2::itsmft::GBTPaddedWordLength;
      for (int iw = 0; iw < nGBTWords; iw++, raw += o2::itsmft::GBTPaddedWordLength) {
        auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(raw);
        // TODO: need to clarify if the nGBTWords from the rdh->memorySize is reliable estimate of the real payload, at the moment this is not the case

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
        ruLink->lanesWithData |= 0x1 << cableSW;    // flag that the data was seen on this lane
        if (ruLink->lanesStop & (0x1 << cableSW)) { // make sure stopped lanes do not transmit the data
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane]++;
          LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Data received for stopped lane " << cableHW << " (sw:" << cableSW << ")";
          printRDH(rdh);
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
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " GBT payload trailer was expected, abort page decoding at NW" << nGBTWords;
        printRDH(rdh);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer]++;
        aborted = true;
        outBuffer.shrinkToSize(sizeAtEntry); // reset output buffer to initial state
        return raw;
      }

      ruLink->lanesTimeOut |= gbtT->getLanesTimeout(); // register timeouts
      ruLink->lanesStop |= gbtT->getLanesStop();       // register stops
#endif

      outBuffer.addFast(reinterpret_cast<const uint8_t*>(gbtT), mGBTWordSize); // save gbt trailer w/o 128b padding

      raw += o2::itsmft::GBTPaddedWordLength;

      // we finished the GBT page, register in the stored RDH the memory size and new offset
      rdhS->memorySize = rdhS->headerSize + (2 + nGBTWords) * mGBTWordSize;
      rdhS->offsetToNext = rdhS->memorySize;

      if (!rdh->offsetToNext) { // RS TODO: what the last page in memory will contain as offsetToNext, is it 0?
        break;
      }

      raw = ((uint8_t*)rdh) + rdh->offsetToNext; // jump to the next packet:
      auto rdhN = reinterpret_cast<const o2::header::RAWDataHeader*>(raw);
      // check if data of given RU are over, i.e. we the page counter was wrapped to 0 (should be enough!) or other RU/trigger started
      if (!isSameRUandTrigger(rdh, rdhN)) {

#ifdef _RAW_READER_ERROR_CHECKS_
        // make sure all lane stops for finished page are received
        if (ruLink->lanesActive != ruLink->lanesStop && nGBTWords) {
          if (rdh->triggerType != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
            std::bitset<32> active(ruLink->lanesActive), stopped(ruLink->lanesStop);
            LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " end of FEE data but not all lanes received stop"
                       << "| active: " << active << " stopped: " << stopped;
            printRDH(rdh);
            ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes]++;
          }
        }

        // make sure all active lanes (except those in time-out) have sent some data
        if ((~ruLink->lanesWithData & ruLink->lanesActive) != ruLink->lanesTimeOut && nGBTWords) {
          std::bitset<32> withData(ruLink->lanesWithData), active(ruLink->lanesActive), timeOut(ruLink->lanesTimeOut);
          LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Lanes not in time-out but not sending data"
                     << "| with data: " << withData << " active: " << active << " timeOut: " << timeOut;
          printRDH(rdh);
          ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane]++;
        }

        // accumulate packet states
        ruLinkStat.packetStates[gbtT->getPacketState()]++;
#endif

        break;
      }
#ifdef _RAW_READER_ERROR_CHECKS_
      // check if the page counter increases
      if (rdhN->pageCnt != rdh->pageCnt + 1) {
        LOG(ERROR) << "FEEId:" << OUTHEX(rdh->feeId, 4) << " Discontinuity in the RDH page counter of the same RU trigger: old "
                   << rdh->pageCnt << " new: " << rdhN->pageCnt;
        printRDH(rdh);
        printRDH(rdhN);
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrPageCounterDiscontinuity]++;
      }
#endif
      rdh = rdhN;
      ruLink->lastRDH = rdh;
    }

#ifdef _RAW_READER_ERROR_CHECKS_
//    if (rdh->pageCnt && !rdh->stop) {
//      LOG(WARNING) << "Last packet(" << rdh->pageCnt << ") of GBT multi-packet is reached w/o STOP set in the RDH";
//    }
#endif

    return raw;
  }

  //_____________________________________
  bool isSameRUandTrigger(const o2::header::RAWDataHeader* rdhOld, const o2::header::RAWDataHeader* rdhNew) const
  {
    /// check if the rdhNew is just a continuation of the data described by the rdhOld
    if (rdhNew->pageCnt == 0 || rdhNew->feeId != rdhOld->feeId ||
        rdhNew->triggerOrbit != rdhOld->triggerOrbit ||
        rdhNew->triggerBC != rdhOld->triggerBC ||
        rdhNew->heartbeatOrbit != rdhOld->heartbeatOrbit ||
        rdhNew->heartbeatBC != rdhOld->heartbeatBC ||
        !(rdhNew->triggerType & rdhOld->triggerType)) {
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
      auto& ruLinkStat = decData.links[decData.cableLinkID[icab]]->statistics;

      // make sure the lane data starts with chip header or empty chip
      uint8_t h;
      if (cableData.current(h) && !mCoder.isChipHeaderOrEmpty(h)) {
        LOG(ERROR) << "FEEId:" << OUTHEX(decData.ruInfo->idHW, 4) << " cable " << icab
                   << " data does not start with ChipHeader or ChipEmpty";
        ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrCableDataHeadWrong]++;
        printRDH(decData.links[decData.cableLinkID[icab]]->lastRDH);
      }
#endif

      while ((res = mCoder.decodeChip(*chipData, cableData))) { // we register only chips with hits or errors flags set
        if (res > 0) {
#ifdef _RAW_READER_ERROR_CHECKS_
          // for the IB staves check if the cable ID is the same as the chip ID on the module
          if (decData.ruInfo->ruType == 0) { // ATTENTION: this is a hack tailored for temporary check
            if (chipData->getChipID() != icab) {
              LOG(ERROR) << "FEEId:" << OUTHEX(decData.ruInfo->idHW, 4) << " IB cable " << icab
                         << " shipped chip ID= " << chipData->getChipID();
              ruLinkStat.errorCounts[GBTLinkDecodingStat::ErrIBChipLaneMismatch]++;
              printRDH(decData.links[decData.cableLinkID[icab]]->lastRDH);
            }
          }
#endif
          // convert HW chip id within the module to absolute chip id
          chipData->setChipID(mMAP.getGlobalChipID(chipData->getChipID(), decData.cableHWID[icab], *decData.ruInfo));
          chipData->setInteractionRecord(mInteractionRecord);
          chipData->setTrigger(mTrigger);
          mDecodingStat.nNonEmptyChips++;
          mDecodingStat.nHitsDecoded += chipData->getData().size();
          ntot += res;
          // fetch next free chip
          if (++decData.nChipsFired < MaxChipsPerRU) {
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
    LOG(INFO) << "opening raw data input file " << filename;
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
    if (mRUEntry[idSW] < 0 || ruLink >= MaxLinksPerRU || !mRUDecodeVec[mRUEntry[idSW]].links[ruLink]) {
      return nullptr;
    } else {
      return &mRUDecodeVec[mRUEntry[idSW]].links[ruLink]->statistics;
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
  const std::array<RUDecodeData, ChipMappingITS::getNRUs()>& getRUDecodeVec() const { return mRUDecodeVec; }

  const std::array<int, ChipMappingITS::getNRUs()>& getRUEntries() const { return mRUEntry; }

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
      LOG(INFO) << "Defining container for RU " << ruSW << " at slot " << mRUEntry[ruSW];
    }
    return mRUDecodeVec[mRUEntry[ruSW]];
  }

 private:
  std::ifstream mIOFile;
  Coder mCoder;
  Mapping mMAP;
  int mVerbose = 0;        //! verbosity level
  int mCurRUDecodeID = -1; //! index of currently processed RUDecode container

  PayLoadCont mRawBuffer; //! buffer for binary raw data file IO

  std::array<RUDecodeData, ChipMappingITS::getNRUs()> mRUDecodeVec; // decoding buffers for all active RUs
  std::array<int, ChipMappingITS::getNRUs()> mRUEntry;              //! entry of the RU with given SW ID in the mRUDecodeVec
  int mNRUs = 0;                                                    //! total number of RUs seen
  int mNLinks = 0;                                                  //! total number of GBT links seen

  //! min number of triggers to cache per link (keep this > N pages per CRU superpage)
  int mMinTriggersToCache = NCRUPagesPerSuperpage + 10;
  int mMinTriggersCached = 0; //! actual minimum (among different links) number of triggers to cache

  // statistics
  RawDecodingStat mDecodingStat; //! global decoding statistics

  TStopwatch mSWIO; //! timer for IO operations

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
