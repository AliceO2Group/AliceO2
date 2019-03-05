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
#include <memory>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string_view>
#include <array>
#include <bitset>

#define _RAW_READER_ERROR_CHECKS_

namespace o2
{
namespace ITSMFT
{

constexpr int MaxCablesPerRU = 28;          // max numbet of cables RU can readout
constexpr int MaxChipsPerRU = 196;          // max number of chips the RU can readout
constexpr int MaxGBTPacketBytes = 8 * 1024; // Max size of GBT packet in bytes (8KB)

struct RUDecodingStat {

  // counters for format checks
  enum DecErrors : int {
    ErrGarbageAfterPayload,      // garbage (non-0) detected after CRU page payload is over
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
    NErrorsDefined
  };

  uint32_t lanesActive = 0;   // lanes declared by the payload header
  uint32_t lanesStop = 0;     // lanes received stop in the payload trailer
  uint32_t lanesTimeOut = 0;  // lanes received timeout
  uint32_t lanesWithData = 0; // lanes with data transmitted

  uint32_t nPackets = 0;                                                   // total number of packets
  std::array<int, NErrorsDefined> errorCounts = {};                        // error counters
  std::array<int, GBTDataTrailer::MaxStateCombinations> packetStates = {}; // packet status from the trailer

  //_____________________________________________________
  void clear()
  {
    nPackets = 0;
    errorCounts.fill(0);
    packetStates.fill(0);
    lanesActive = lanesStop = lanesTimeOut = lanesWithData = 0;
  }

  //_____________________________________________________
  void print(bool skipEmpty = true) const
  {
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
    printf("Packet States Statistics (total packets: %d)\n", nPackets);
    for (int i = 0; i < GBTDataTrailer::MaxStateCombinations; i++) {
      if (packetStates[i]) {
        std::bitset<GBTDataTrailer::NStatesDefined> patt(i);
        printf("counts for triggers B[%s] : %d\n", patt.to_string().c_str(), packetStates[i]);
      }
    }
  }

  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "Garbage (non-0) detected after CRU page payload is over",           // ErrGarbageAfterPayload
    "RDH page counters for the same RU/trigger are not continuous",      // ErrPageCounterDiscontinuity
    "RDH ang GBT header page counters are not consistent",               // ErrRDHvsGBTHPageCnt
    "GBT payload header was expected but not found",                     // ErrMissingGBTHeader
    "GBT payload trailer was expected but not found",                    // ErrMissingGBTTrailer
    "All lanes were stopped but the page counter in not 0",              // ErrNonZeroPageAfterStop
    "End of FEE data reached while not all lanes received stop",         // ErrUnstoppedLanes
    "Data was received for stopped lane",                                // ErrDataForStoppedLane
    "No data was seen for lane (which was not in timeout)",              // ErrNoDataForActiveLane
    "ChipID (on module) was different from the lane ID on the IB stave", // ErrIBChipLaneMismatch
    "Cable data does not start with chip header or empty chip"           // ErrCableDataHeadWrong
  };

  ClassDefNV(RUDecodingStat, 1);
};

constexpr std::array<std::string_view, RUDecodingStat::NErrorsDefined> RUDecodingStat::ErrNames;

struct RawDecodingStat {
  using ULL = unsigned long long;
  uint64_t nPagesProcessed = 0; // total number of pages processed
  uint64_t nRUsProcessed = 0;   // total number of RUs processed (1 RU may take a few pages)
  uint64_t nBytesProcessed = 0; // total number of bytes (rdh->memorySize) processed
  uint64_t nNonEmptyChips = 0;  // number of non-empty chips found
  uint64_t nHitsDecoded = 0;    // number of hits found

  RawDecodingStat() = default;

  void clear()
  {
    nPagesProcessed = 0;
    nRUsProcessed = 0;
    nBytesProcessed = 0;
    nNonEmptyChips = 0;
    nHitsDecoded = 0;
  }

  void print() const
  {
    printf("\nDecoding statistics\n");
    printf("%llu bytes for %llu RUs processed in %llu pages\n", (ULL)nBytesProcessed, (ULL)nRUsProcessed, (ULL)nPagesProcessed);
    printf("%llu hits found in %llu non-empty chips\n", (ULL)nHitsDecoded, (ULL)nNonEmptyChips);
  }

  ClassDefNV(RawDecodingStat, 1);
};

struct RUEncodeData {
  // Scatter-gather list for the data of different cables data created during encoding,
  // plust some aux data for needed for the encoding
  std::array<int, MaxCablesPerRU> cableEntryInRaw{}; // entry of given cable data in raw buffer
  std::array<int, MaxCablesPerRU> cableEndInRaw{};   // end of given cable data in raw buffer
  std::array<uint8_t, MaxCablesPerRU> cableHWID{};   // HW id's of the cables
  PayLoadCont buffer;                                // buffer to hold alpide converted data

  o2::InteractionRecord bcData;

  int id = -1;     // secuential ID of the RU
  int type = -1;   // type of the RU (or stave it is serving)
  int nChips = 0;  // number of chips served by this RU
  int nCables = 0; // number of cables served by this RU
  //
  int chipIDFirst2Conv = 0; // 1st chip whose data are still not converted to raw

  o2::ITSMFT::ChipInfo chInfo; // info on the chip currenly processed

  void clear()
  {
    std::fill_n(cableEndInRaw.begin(), nCables, 0);
    nChips = 0;
    nCables = 0;
    id = type = -1;
    chipIDFirst2Conv = 0;
  }
};

struct RUDecodeData {
  std::array<PayLoadCont, MaxCablesPerRU> cableData;              // cable data in compressed ALPIDE format
  std::array<uint8_t, MaxCablesPerRU> cableHWID;                  // HW ID of cable whose data is in the corresponding slot of cableData
  std::array<o2::ITSMFT::ChipPixelData, MaxChipsPerRU> chipsData; // fully decoded data

  int nCables = 0;         // total number of cables
  int nChipsFired = 0;     // number of chips with data or with errors
  int lastChipChecked = 0; // last chips checked among nChipsFired
  const RUInfo* ruInfo = nullptr;

  void clear()
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

template <class Mapping = o2::ITSMFT::ChipMappingITS>
class RawPixelReader : public PixelReader
{
  using Coder = o2::ITSMFT::AlpideCoder;

 public:
  static constexpr bool isPadding128() { return Padding128Data; }
  static constexpr bool getGBTWordSize() { return GBTWordSize; }

  RawPixelReader() = default;
  ~RawPixelReader() override
  {
    mSWIO.Stop();
    printf("RawPixelReader IO time: ");
    mSWIO.Print();
  }

  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) override
  {
    // decode new RU if no cached non-empty chips
    while (mRUDecode.lastChipChecked >= mRUDecode.nChipsFired && decodeNextRUData()) {
    }
    if (mRUDecode.lastChipChecked < mRUDecode.nChipsFired) {
      auto& chipData = mRUDecode.chipsData[mRUDecode.lastChipChecked++];
      int id = chipData.getChipID();
      chipDataVec[id].swap(chipData);
      //      chipDataVec[id].print();
      return &chipDataVec[id];
    }
    return nullptr;
  }

  void init() override{};

  void clear()
  {
    mDecodingStat.clear();
    for (auto& ruStat : mRUDecodingStat) {
      ruStat.clear();
    }
    mIOFile.close();
    mRawBuffer.clear();
    mRUDecode.clear();
  }

  int digits2raw(const std::vector<o2::ITSMFT::Digit>& digiVec, int from, int ndig, const o2::InteractionRecord& bcData,
                 PayLoadCont& sink, uint8_t ruSWMin = 0, uint8_t ruSWMax = 0xff)
  {
    // convert vector of digits to binary vector (no reset is applied)
    constexpr uint16_t DummyChip = 0xffff;
    int nDigTot = digiVec.size();
    assert(from < nDigTot);
    int oldsize = sink.getSize(), last = (from + ndig <= nDigTot) ? from + ndig : nDigTot;
    int curChipID = DummyChip; // currently processed SW chip id

    o2::ITSMFT::ChipPixelData chipData; // data of currently processed chip

    mRUEncode.clear();
    chipData.clear();
    chipData.setChipID(DummyChip);
    int emptyRUID = 0;
    bool ignore = false;

    mRUEncode.bcData = bcData;
    for (int id = from; id < last; id++) {
      const auto& dig = digiVec[id];
      if (dig.getChipIndex() != curChipID) { // new chip data start

        // since a new chip data start, convert digits of previously processed chip to Alpide format in the temp. buffer
        if (curChipID != DummyChip && !ignore) {
          convertChipData(mRUEncode, chipData);
        }
        curChipID = dig.getChipIndex();
        MAP.getChipInfoSW(curChipID, mRUEncode.chInfo);

        if (mRUEncode.chInfo.ru != mRUEncode.id) { // new RU data starts, flush the previous one
          if (mRUEncode.id != -1) {
            for (; emptyRUID < mRUEncode.id; emptyRUID++) {       // flush empty raw data for RUs w/o digits
              if (emptyRUID >= ruSWMin && emptyRUID <= ruSWMax) { // do we want this RU?
                flushEmptyRU(emptyRUID, mRUEncode.bcData, sink);
              }
            }
            emptyRUID = mRUEncode.id + 1; // flushing of next empty RU will start from this one
            if (!ignore) {
              flushRUData(mRUEncode, sink); // flush already converted RU data
            }
          }
          mRUEncode.id = mRUEncode.chInfo.ru; // and update with new RU identifier
          mRUEncode.type = mRUEncode.chInfo.ruType;
          mRUEncode.nChips = MAP.getNChipsOnRUType(mRUEncode.chInfo.ruType);
          mRUEncode.nCables = MAP.getNCablesOnRUType(mRUEncode.chInfo.ruType);
          ignore = mRUEncode.id < ruSWMin || mRUEncode.id > ruSWMax;
        }
        chipData.setChipID(mRUEncode.chInfo.chOnRU->chipOnModuleHW); // HW chip ID in module
      }
      if (!ignore) {                           // add only if the RU is not to be rejected
        chipData.getData().emplace_back(&dig); // add new digit to the container
      }
    }
    if (!ignore) {
      convertChipData(mRUEncode, chipData); // finish what is left in the buffer
    }
    for (; emptyRUID < mRUEncode.id; emptyRUID++) {       // flush empty raw data for RUs w/o digits
      if (emptyRUID >= ruSWMin && emptyRUID <= ruSWMax) { // do we want this RU?
        flushEmptyRU(emptyRUID, mRUEncode.bcData, sink);
      }
    }
    if (!ignore) {
      flushRUData(mRUEncode, sink);
    }
    // flush empty raw data for RUs w/o digits
    for (emptyRUID = mRUEncode.chInfo.ru + 1; emptyRUID < MAP.getNRUs(); emptyRUID++) {
      if (emptyRUID >= ruSWMin && emptyRUID <= ruSWMax) { // do we want this RU?
        flushEmptyRU(emptyRUID, mRUEncode.bcData, sink);
      }
    }

    return last - from;
  }

  //___________________________________________________________________________________
  void convertChipData(RUEncodeData& ruData, o2::ITSMFT::ChipPixelData& chData)
  {
    // convert digits of single chip within the RU to Alpide format, storing raw data in the
    // internal buffer of the ruData
    static o2::ITSMFT::ChipPixelData dummyData;
    // add empty chip records for all digit-less chips with ID < currently processed chipID
    convertEmptyChips(ruData, ruData.chInfo.chOnRU->id);

    convertChipDigits(ruData, chData); // convert previously filled chip data to raw

    chData.clear();
    ruData.chipIDFirst2Conv = ruData.chInfo.chOnRU->id + 1; // flag 1st unconverted chip
  }

  //___________________________________________________________________________________
  void convertChipDigits(RUEncodeData& ruData, o2::ITSMFT::ChipPixelData& pixData)
  {
    ///< convert digits of single chip to Alpide format
    const auto& chip = *ruData.chInfo.chOnRU;

    ruData.cableHWID[chip.cableSW] = chip.cableHW;                    // register the cable HW ID
    if (!chip.chipOnCable) {                                          // is this a 1st chip served by this cable (i.e. master)?
      ruData.cableEntryInRaw[chip.cableSW] = ruData.buffer.getSize(); // start of this cable data
    }

    auto& pixels = pixData.getData();
    std::sort(pixels.begin(), pixels.end(),
              [](auto lhs, auto rhs) {
                if (lhs.getRow() < rhs.getRow())
                  return true;
                if (lhs.getRow() > rhs.getRow())
                  return false;
                return lhs.getCol() < rhs.getCol();
              });
    ruData.buffer.ensureFreeCapacity(40 * (2 + pixels.size())); // make sure buffer has enough capacity
    mCoder.encodeChip(ruData.buffer, pixData, chip.chipOnModuleHW, ruData.bcData.bc);

    ruData.cableEndInRaw[chip.cableSW] = ruData.buffer.getSize(); // current end of this cable data
  }

  //______________________________________________________
  void convertEmptyChips(RUEncodeData& ruData, int upto)
  {
    // add empty chip markers
    int chipIDSW = ruData.chipIDFirst2Conv;
    int cableID = 0;
    auto& buffer = ruData.buffer;
    buffer.ensureFreeCapacity(ruData.nChips << 2); // we need at least 2 bytes per empty chip, here I ensure 4
    for (; chipIDSW < upto; chipIDSW++) {          // flag chips w/o data

      const auto& chip = *MAP.getChipOnRUInfo(ruData.type, chipIDSW);
      ruData.cableHWID[chip.cableSW] = chip.cableHW;             // register the cable HW ID
      if (!chip.chipOnCable) {                                   // is this a 1st chip served by this cable (i.e. master)?
        ruData.cableEntryInRaw[chip.cableSW] = buffer.getSize(); // start of this cable data
      }
      mCoder.addEmptyChip(buffer, chip.chipOnModuleHW, ruData.bcData.bc);
      ruData.cableEndInRaw[chip.cableSW] = buffer.getSize(); // current end of this cable data
    }
    ruData.chipIDFirst2Conv = upto;
  }

  //_____________________________________________________________________________
  void flushEmptyRU(int ruID, const o2::InteractionRecord& bcData, PayLoadCont& sink)
  {
    // create raw data for empty RU

    mRUEncodeEmpty.type = MAP.getRUType(ruID);
    mRUEncodeEmpty.id = ruID;
    mRUEncodeEmpty.nChips = MAP.getNChipsOnRUType(mRUEncodeEmpty.type);
    mRUEncodeEmpty.nCables = MAP.getNCablesOnRUType(mRUEncodeEmpty.type);
    mRUEncodeEmpty.bcData = bcData;
    mRUEncodeEmpty.chipIDFirst2Conv = 0;
    if (mVerbose) {
      LOG(INFO) << "Flushing empty RU#" << mRUEncodeEmpty.id
                << " Orbit:" << mRUEncodeEmpty.bcData.orbit << " BC: " << mRUEncodeEmpty.bcData.bc;
    }
    convertEmptyChips(mRUEncodeEmpty, mRUEncodeEmpty.nChips);
    flushRUData(mRUEncodeEmpty, sink);
  }

  //_____________________________________________________________________________
  void printRDH(const o2::header::RAWDataHeader& h)
  {
    printf("RDH| Ver:%2u Hsz:%2u Blgt:%4u FEEId:0x%04x PBit:%u\n",
           uint32_t(h.version), uint32_t(h.headerSize), uint32_t(h.blockLength), uint32_t(h.feeId), uint32_t(h.priority));
    printf("RDH|[CRU: Offs:%5u Msz:%4u LnkId:0x%02x Packet:%3u CRUId:0x%04x]\n",
           uint32_t(h.offsetToNext), uint32_t(h.memorySize), uint32_t(h.linkID), uint32_t(h.packetCounter), uint32_t(h.cruID));
    printf("RDH| TrgOrb:%9u HBOrb:%9u TrgBC:%4u HBBC:%4u TrgType:%u\n",
           uint32_t(h.triggerOrbit), uint32_t(h.heartbeatOrbit), uint32_t(h.triggerBC), uint32_t(h.heartbeatBC),
           uint32_t(h.triggerType));
    printf("RDH| DetField:0x%05x Par:0x%04x Stop:0x%04x PageCnt:%5u\n", uint32_t(h.detectorField), uint32_t(h.par), uint32_t(h.stop), uint32_t(h.pageCnt));
  }

  //___________________________________________________________________________________
  void flushRUData(RUEncodeData& ruData, PayLoadCont& sink)
  {
    constexpr uint8_t zero16[o2::ITSMFT::GBTPaddedWordLength] = { 0 }; // to speedup padding
    if (ruData.id < 0) {
      return;
    }
    convertEmptyChips(ruData, ruData.nChips); // fill empty chips up to the last chip of the RU

    // calculate number of GBT words needed to store this data (with 9 bytes per GBT word)
    int nGBTWordsNeeded = 0;
    for (int i = 0; i < ruData.nCables; i++) {
      int start = ruData.cableEntryInRaw[i];
      int end = ruData.cableEndInRaw[i];
      int nb = end > start ? end - start : 0;
      nGBTWordsNeeded += nb ? 1 + (nb - 1) / 9 : 0;
    }
    //
    // prepare RDH
    // 4x128 bit, represented as 8 64-bit words
    o2::header::RAWDataHeader rdh;
    rdh.headerSize = 0x40; // 4*128 bits;
    rdh.feeId = MAP.RUSW2HW(ruData.id);
    rdh.triggerOrbit = rdh.heartbeatOrbit = ruData.bcData.orbit;
    rdh.triggerBC = rdh.heartbeatBC = ruData.bcData.bc;
    rdh.triggerType = o2::trigger::PhT;
    rdh.detectorField = MAP.getRUDetectorField();
    rdh.par = 0;
    o2::ITSMFT::GBTDataHeader gbtHeader(0, MAP.getCablesOnRUType(ruData.type));
    o2::ITSMFT::GBTDataTrailer gbtTrailer;

    // max real payload words (accounting for GBT header and trailer) per packet
    int maxGBTWordsPerPacket = (MaxGBTPacketBytes - rdh.headerSize) / GBTWordSize - 2;

    if (sink.getFreeCapacity() < 2 * MaxGBTPacketBytes) { // make sure there is enough capacity
      sink.expand(sink.getCapacity() + 10 * MaxGBTPacketBytes);
    }

    rdh.blockLength = 0xffff;                                              //MaxGBTPacketBytes;      ITS does not fill it     // total packet size: always use max size ? in bits ?
    rdh.memorySize = rdh.headerSize + (nGBTWordsNeeded + 2) * GBTWordSize; // update remaining size
    if (rdh.memorySize > MaxGBTPacketBytes) {
      rdh.memorySize = MaxGBTPacketBytes;
    }
    rdh.offsetToNext = MaxGBTPacketBytes; // save offset to next packet: always max size?
    rdh.stop = 0;

    sink.addFast(reinterpret_cast<uint8_t*>(&rdh), rdh.headerSize); // write RDH for current packet

    gbtHeader.setPacketID(rdh.pageCnt);
    sink.addFast(gbtHeader.getW8(), GBTWordSize); // write GBT header for current packet

    if (mVerbose) {
      LOG(INFO) << "Flushing RU data";
      printRDH(rdh);
      gbtHeader.printX(Padding128Data);
    }

    int nGBTWordsInPacket = 0;
    do {
      for (int icab = 0; icab < ruData.nCables; icab++) {
        int &start(ruData.cableEntryInRaw[icab]), &end(ruData.cableEndInRaw[icab]), nb = end - start;
        if (nb > 0) { // write 80b word only if there is something to write
          if (nb > 9) {
            nb = 9;
          }
          int gbtWordStart = sink.getSize(); // beginning of the current GBT word in the sink
          sink.add(ruData.buffer.data() + start, nb);
          // fill the rest of the GBT word by 0
          sink.add(zero16, GBTWordSize - nb);
          sink[gbtWordStart + 9] = MAP.getGBTHeaderRUType(ruData.type, ruData.cableHWID[icab]); // set cable flag
          start += nb;
          nGBTWordsNeeded--;
          if (mVerbose) {
            ((GBTData*)(sink.data() + gbtWordStart))->printX(Padding128Data);
          }
          if (++nGBTWordsInPacket == maxGBTWordsPerPacket) { // check if new GBT packet must be created
            break;
          }
        } // storing data of single cable
      }   // loop over cables

      if (nGBTWordsNeeded && nGBTWordsInPacket >= maxGBTWordsPerPacket) {
        // more data to write, write trailer and add new GBT packet
        sink.add(gbtTrailer.getW8(), GBTWordSize); // write GBT trailer for current packet
        if (mVerbose) {
          gbtTrailer.printX(Padding128Data);
        }
        rdh.pageCnt++;                                                         // flag new page
        rdh.stop = nGBTWordsNeeded < maxGBTWordsPerPacket;                     // flag if this is the last packet of multi-packet
        rdh.blockLength = 0xffff;                                              // (nGBTWordsNeeded % maxGBTWordsPerPacket + 2) * GBTWordSize; // record payload size
        rdh.memorySize = rdh.headerSize + (nGBTWordsNeeded + 2) * GBTWordSize; // update remaining size
        if (rdh.memorySize > MaxGBTPacketBytes) {
          rdh.memorySize = MaxGBTPacketBytes;
        }
        rdh.offsetToNext = MaxGBTPacketBytes;                       // save offset to next packet
        sink.add(reinterpret_cast<uint8_t*>(&rdh), rdh.headerSize); // write RDH for new packet
        if (mVerbose) {
          printRDH(rdh);
        }
        gbtHeader.setPacketID(rdh.pageCnt);
        sink.addFast(gbtHeader.getW8(), GBTWordSize); // write GBT header for current packet
        if (mVerbose) {
          gbtHeader.printX(Padding128Data);
        }
        nGBTWordsInPacket = 0; // reset counter of words in the packet
      }

    } while (nGBTWordsNeeded);

    gbtTrailer.setLanesStop(MAP.getCablesOnRUType(ruData.type));
    gbtTrailer.setPacketState(0x1 << GBTDataTrailer::PacketDone);

    sink.add(gbtTrailer.getW8(), GBTWordSize); // write GBT trailer for the last packet
    if (mVerbose) {
      gbtTrailer.printX(Padding128Data);
    }

    // at the moment the GBT page must be aligned to MaxGBTPacketBytes, fill by 0s
    while (nGBTWordsInPacket++ < maxGBTWordsPerPacket) {
      sink.add(zero16, GBTWordSize);
      if (mVerbose) {
        ((GBTData*)zero16)->printX(Padding128Data);
      }
    }

    ruData.clear();
  }

  //_____________________________________
  int decodeNextRUData()
  {
    if (mIOFile) {
      loadInput(); // if needed, upload additional data to the buffer
    }
    if (!mRawBuffer.isEmpty()) {
      bool aborted = false;

      //      printf("Unused: %d Size: %d Offs: %d\n", mRawBuffer.getUnusedSize(),  mRawBuffer.getSize(), mRawBuffer.getOffset() );
      auto ptr = decodeRUData(mRawBuffer.getPtr(), aborted);

      if (!aborted) {
        mRawBuffer.setPtr(ptr);
        if (mRawBuffer.isEmpty()) {
          mRawBuffer.clear();
          return 1;
        }
      } else { // try to seek to the next RDH
        LOG(ERROR) << "Will seek the next RDH after aborting (unreliable!!)";
        auto ptrTail = mRawBuffer.getPtr() + mRawBuffer.getUnusedSize();
        ptr += ((ptr - mRawBuffer.getPtr()) / GBTWordSize) * GBTWordSize; // jump to the next GBT word
        while (ptr + GBTWordSize <= ptrTail) {
          auto* rdh = reinterpret_cast<const o2::header::RAWDataHeader*>(ptr);
          if (isRDHHeuristic(rdh)) { // this heuristics is not reliable...
            LOG(WARNING) << "Assuming that found a new RDH";
            printRDH(*rdh);
            mRawBuffer.setPtr(ptr);
            break;
          }
          ptr += GBTWordSize;
        }
        mRawBuffer.setPtr(ptr);
        if (mRawBuffer.isEmpty()) {
          mRawBuffer.clear();
          return 0; // did not find new RDH
        }
      }
      return 1;
    }
    return 0;
  }

  //_____________________________________
  bool isRDHHeuristic(const o2::header::RAWDataHeader* rdh)
  {
    /// heuristic check if this is indeed an RDH
    return (!rdh || rdh->headerSize != sizeof(o2::header::RAWDataHeader) || rdh->zero0 != 0 || rdh->zero1 != 0 ||
            rdh->zero41 != 0 || rdh->zero42 != 0 || rdh->word5 != 0 || rdh->zero6 != 0)
             ? false
             : true;
  }

  //_____________________________________
  uint8_t* decodeRUData(uint8_t* raw, bool& aborted)
  {
    /// Decode raw data of single RU (possibly in a few GBT packets)
    /// No check is done if the necessary data are fully contained in the raw buffer.
    /// Return the pointer on the last raw data byte after decoding the RU
    /// In case of unrecoverable error set aborted to true

    aborted = false;
    mRUDecode.clear();

    // data must start by RDH
    auto rdh = reinterpret_cast<const o2::header::RAWDataHeader*>(raw);

#ifdef _RAW_READER_ERROR_CHECKS_
    if (!isRDHHeuristic(rdh)) {
      LOG(ERROR) << "Page does not start with RDH";
      for (int i = 0; i < 4; i++) {
        auto gbtD = reinterpret_cast<const o2::ITSMFT::GBTData*>(raw + i * 16);
        gbtD->printX(Padding128Data);
      }
      raw += GBTWordSize;
      aborted = true;
      return raw;
    }
#endif

    int ruIDSW = MAP.RUHW2SW(rdh->feeId);
    //    LOG(INFO) << "Decoding RU:" << rdh->feeId << " swID: " << ruIDSW << " Orbit:" << rdh->triggerOrbit << " BC: " << rdh->triggerBC;

    mInteractionRecord.bc = rdh->triggerBC;
    mInteractionRecord.orbit = rdh->triggerOrbit;
    mTrigger = rdh->triggerType;

    auto& currRU = mRUDecodingStat[ruIDSW];
    currRU.nPackets++;
    mDecodingStat.nRUsProcessed++;

    mRUDecode.ruInfo = MAP.getRUInfoSW(ruIDSW); // info on the stave being decoded
    mRUDecode.nCables = mRUDecode.ruInfo->nCables;
    while (1) {
      mDecodingStat.nPagesProcessed++;
      mDecodingStat.nBytesProcessed += rdh->memorySize;
      raw += rdh->headerSize;
      int nGBTWords = (rdh->memorySize - rdh->headerSize) / GBTWordSize - 2; // number of GBT words excluding header/trailer
      auto gbtH = reinterpret_cast<const o2::ITSMFT::GBTDataHeader*>(raw);   // process GBT header

#ifdef _RAW_READER_ERROR_CHECKS_
      if (mVerbose) {
        printRDH(*rdh);
        gbtH->printX(Padding128Data);
        LOG(INFO) << "Expect " << nGBTWords << " GBT words";
      }
      if (!gbtH->isDataHeader()) {
        gbtH->printX(Padding128Data);
        LOG(ERROR) << "FEE#" << rdh->feeId << " GBT payload header was expected, abort page decoding";
        gbtH->printX(Padding128Data);
        currRU.errorCounts[RUDecodingStat::ErrMissingGBTHeader]++;
        aborted = true;
        return raw;
      }
      if (gbtH->getPacketID() != rdh->pageCnt) {
        LOG(ERROR) << "FEE#" << rdh->feeId << " Different GBT header " << gbtH->getPacketID()
                   << " and RDH page " << rdh->pageCnt << " counters";
        currRU.errorCounts[RUDecodingStat::ErrRDHvsGBTHPageCnt]++;
      }

      if (currRU.lanesActive == currRU.lanesStop) { // all lanes received their stop, new page 0 expected
        if (rdh->pageCnt) {                         // flag lanes of this FEE
          LOG(ERROR) << "FEE#" << rdh->feeId << " Non-0 page counter (" << rdh->pageCnt << ") while all lanes were stopped";
          currRU.errorCounts[RUDecodingStat::ErrNonZeroPageAfterStop]++;
        }
      }

      currRU.lanesActive = gbtH->getLanes(); // TODO do we need to update this for every page?

      if (!rdh->pageCnt) { // reset flags
        currRU.lanesStop = 0;
        currRU.lanesWithData = 0;
      }

#endif
      raw += GBTWordSize;
      for (int iw = 0; iw < nGBTWords; iw++, raw += GBTWordSize) {
        auto gbtD = reinterpret_cast<const o2::ITSMFT::GBTData*>(raw);
        // TODO: need to clarify if the nGBTWords from the rdh->memorySize is reliable estimate of the real payload, at the moment this is not the case

        if (mVerbose) {
          printf("W%4d |", iw);
          gbtD->printX(Padding128Data);
        }
        if (gbtD->isDataTrailer()) {
          nGBTWords = iw;
          break; // this means that the nGBTWords estimate was wrong
        }

        int cableHW = gbtD->getCableID();
        int cableSW = MAP.cableHW2SW(mRUDecode.ruInfo->ruType, cableHW);
        mRUDecode.cableData[cableSW].add(gbtD->getW8(), 9);
        mRUDecode.cableHWID[cableSW] = cableHW;

#ifdef _RAW_READER_ERROR_CHECKS_
        currRU.lanesWithData |= 0x1 << cableSW;    // flag that the data was seen on this lane
        if (currRU.lanesStop & (0x1 << cableSW)) { // make sure stopped lanes do not transmit the data
          currRU.errorCounts[RUDecodingStat::ErrDataForStoppedLane]++;
          LOG(ERROR) << "FEE#" << rdh->feeId << " Data received for stopped lane " << cableHW << " (sw:" << cableSW << ")";
        }
#endif

      } // we are at the trailer, packet is over, check if there are more for the same ru

      auto gbtT = reinterpret_cast<const o2::ITSMFT::GBTDataTrailer*>(raw); // process GBT trailer
#ifdef _RAW_READER_ERROR_CHECKS_

      if (mVerbose) {
        printRDH(*rdh);
        gbtT->printX(Padding128Data);
      }

      if (!gbtT->isDataTrailer()) {
        gbtT->printX(Padding128Data);
        LOG(ERROR) << "FEE#" << rdh->feeId << " GBT payload trailer was expected, abort page decoding NW" << nGBTWords;
        currRU.errorCounts[RUDecodingStat::ErrMissingGBTTrailer]++;
        aborted = true;
        return raw;
      }

      currRU.lanesTimeOut |= gbtT->getLanesTimeout(); // register timeouts
      currRU.lanesStop |= gbtT->getLanesStop();       // register stops
#endif
      raw += GBTWordSize;
      // we finished the GBT page, see if there is a continuation and if it belongs to the same multipacket

#ifdef _RAW_READER_ERROR_CHECKS_
      // make sure we have only 0's till the end of the page

      // TODO: not clear if the block length is constantly 8KB or can change, at the moment assume it is
      int nWLeft = (MaxGBTPacketBytes - rdh->headerSize) / GBTWordSize - (nGBTWords + 2);
      for (int iw = 0; iw < nWLeft; iw++) {
        auto gbtD = reinterpret_cast<const o2::ITSMFT::GBTData*>(raw);
        raw += GBTWordSize;
        if (mVerbose) {
          printf("P%4d |", iw);
          gbtD->printX(Padding128Data);
        }
        for (int ib = GBTWordSize; ib--;) {
          if (gbtD->getByte(ib)) {
            LOG(ERROR) << "FEE#" << rdh->feeId << " Non-0 data detected after payload trailer";
            currRU.errorCounts[RUDecodingStat::ErrGarbageAfterPayload]++;
            nWLeft = 0; // do not continue the check
            break;
          }
        }
      }
#endif

      if (!rdh->offsetToNext) { // RS TODO: what the last page in memory will contain as offsetToNext, is it 0?
        break;
      }

      raw = ((uint8_t*)rdh) + rdh->offsetToNext; // jump to the next packet:
      auto rdhN = reinterpret_cast<const o2::header::RAWDataHeader*>(raw);
      // check if data of given RU are over, i.e. we the page counter was wrapped to 0 (should be enough!) or other RU/trigger started
      if (rdhN->pageCnt == 0 || rdhN->feeId != rdh->feeId ||
          rdhN->triggerOrbit != rdh->triggerOrbit ||
          rdhN->triggerBC != rdh->triggerBC ||
          rdhN->heartbeatOrbit != rdh->heartbeatOrbit ||
          rdhN->heartbeatBC != rdh->heartbeatBC ||
          rdhN->triggerType != rdh->triggerType) {
#ifdef _RAW_READER_ERROR_CHECKS_
        // make sure all lane stops for finished page are received
        if (currRU.lanesActive != currRU.lanesStop) {
          if (rdh->triggerType != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
            LOG(ERROR) << "FEE#" << rdh->feeId << " end of FEE data but not all lanes received stop";
            currRU.errorCounts[RUDecodingStat::ErrUnstoppedLanes]++;
          }
        }

        // make sure all active lanes (except those in time-out) have sent some data
        if ((~currRU.lanesWithData & currRU.lanesActive) != currRU.lanesTimeOut) {
          LOG(ERROR) << "FEE#" << rdh->feeId << " Lanes not in time-out but not sending data";
          currRU.errorCounts[RUDecodingStat::ErrNoDataForActiveLane]++;
        }

        // accumulate packet states
        currRU.packetStates[gbtT->getPacketState()]++;
#endif

        break;
      }
#ifdef _RAW_READER_ERROR_CHECKS_
      // check if the page counter increases
      if (rdhN->pageCnt != rdh->pageCnt + 1) {
        LOG(ERROR) << "FEE#" << rdh->feeId << " Discontinuity in the RDH page counter of the same RU trigger: old "
                   << rdh->pageCnt << " new: " << rdhN->pageCnt;
        currRU.errorCounts[RUDecodingStat::ErrPageCounterDiscontinuity]++;
      }
#endif
      rdh = rdhN;
    }

#ifdef _RAW_READER_ERROR_CHECKS_
//    if (rdh->pageCnt && !rdh->stop) {
//      LOG(WARNING) << "Last packet(" << rdh->pageCnt << ") of GBT multi-packet is reached w/o STOP set in the RDH";
//    }
#endif

    // decode Alpide data from the compressed RU Data
    decodeAlpideData(mRUDecode);

    return raw;
  }

  //_____________________________________
  int decodeAlpideData(RUDecodeData& decData)
  {
    /// decode the ALPIDE data from the buffer of single lane

    auto* chipData = &decData.chipsData[0];
    auto& currRU = mRUDecodingStat[decData.ruInfo->idSW];

    chipData->clear();
    decData.nChipsFired = decData.lastChipChecked = 0;
    int ntot = 0;
    for (int icab = 0; icab < decData.nCables; icab++) {
      auto& cableData = decData.cableData[icab];
      int res = 0;

#ifdef _RAW_READER_ERROR_CHECKS_
      // make sure the lane data starts with chip header or empty chip
      uint8_t h;
      if (!cableData.current(h) || !mCoder.isChipHeaderOrEmpty(h)) {
        LOG(ERROR) << "FEE#" << decData.ruInfo->idHW << " cable " << icab << " data does not start with ChipHeader or ChipEmpty";
        currRU.errorCounts[RUDecodingStat::ErrCableDataHeadWrong]++;
      }
#endif

      while ((res = mCoder.decodeChip(*chipData, cableData))) { // we register only chips with hits or errors flags set
        if (res > 0) {

#ifdef _RAW_READER_ERROR_CHECKS_
          // for the IB staves check if the cable ID is the same as the chip ID on the module
          if (decData.ruInfo->ruType == 0) { // ATTENTION: this is a hack tailored for temporary check
            if (chipData->getChipID() != icab) {
              LOG(ERROR) << "FEE#" << decData.ruInfo->idHW << " IB cable " << icab << " shipped chip ID= " << chipData->getChipID();
              currRU.errorCounts[RUDecodingStat::ErrIBChipLaneMismatch]++;
            }
          }
#endif
          // convert HW chip id within the module to absolute chip id
          chipData->setChipID(MAP.getGlobalChipID(chipData->getChipID(), decData.cableHWID[icab], *decData.ruInfo));
          chipData->setInteractionRecord(mInteractionRecord);
          chipData->setTrigger(mTrigger);
          mDecodingStat.nNonEmptyChips++;
          mDecodingStat.nHitsDecoded += chipData->getData().size();
          // fetch next free chip
          chipData = &decData.chipsData[++decData.nChipsFired];
          chipData->clear();
          ntot += res;
        }
      }
    }
    return ntot;
  }

  //_____________________________________
  bool getNextChipData(ChipPixelData& chipData) override
  {
    /// read single chip data to the provided container

    // decode new RU if no cached non-empty chips
    while (mRUDecode.lastChipChecked >= mRUDecode.nChipsFired && decodeNextRUData()) {
    }
    if (mRUDecode.lastChipChecked < mRUDecode.nChipsFired) {
      chipData.swap(mRUDecode.chipsData[mRUDecode.lastChipChecked++]);
      return true;
    }
    return false;
  }

  //_____________________________________
  void openInput(const std::string filename)
  {
    // open input for raw data decoding from file
    mSWIO.Stop();
    mSWIO.Start();
    LOG(INFO) << "opening raw data input file " << filename << FairLogger::endl;
    mIOFile.open(filename.c_str(), std::ifstream::binary);
    assert(mIOFile.good());
    mRawBuffer.clear();
    mRawBuffer.expand(RawBufferSize);
    mSWIO.Stop();
  }

  //_____________________________________
  int loadInput()
  {
    /// assure the buffers are large enough
    static_assert(RawBufferMargin > MaxGBTPacketBytes * 100 &&
                    RawBufferSize > 3 * RawBufferMargin,
                  "raw buffer size is too small");

    if (!mIOFile) {
      return 0;
    }
    if (mRawBuffer.getUnusedSize() > RawBufferMargin) { // bytes read but not used yet are enough
      return 0;
    }
    mSWIO.Start(false);
    auto readFromFile = [this](uint8_t* ptr, int n) {
      mIOFile.read(reinterpret_cast<char*>(ptr), n);
      return (int)mIOFile.gcount(); // fread( ptr, sizeof(uint8_t), n, mIOFile);
    };
    auto nread = mRawBuffer.append(readFromFile);
    mSWIO.Stop();
    return nread;
  }

  // get vector of RUs statistics
  const std::array<RUDecodingStat, Mapping::getNRUs()>& getRUDecodingStat();

  // get statics of FEE with sequential idSW
  const RUDecodingStat& getRUDecodingStatSW(uint16_t idSW) const
  {
    return mRUDecodingStat[idSW];
  }

  // get statics of FEE with given HW id
  const RUDecodingStat& getRUDecodingStatHW(uint16_t idHW) const
  {
    int idsw = 0xffff;
    assert(idHW < mRUDecodingStat.size() && (idsw = MAP.RUHW2SW(idHW)) != 0xffff);
    return getRUDecodingStatSW(idsw);
  }

  // get global decoding statistics
  const RawDecodingStat& getDecodingStat() const { return mDecodingStat; }

  void setVerbosity(int v) { mVerbose = v; }
  int getVerbosity() const { return mVerbose; }

  Mapping& getMapping() { return MAP; }

  const RUDecodeData& getRUDecodeData() const { return mRUDecode; }

 private:
  std::ifstream mIOFile;
  Coder mCoder;
  Mapping MAP;
  int mVerbose = 0;            //! verbosity level
  RUEncodeData mRUEncode;      //! buffer for the digits to convert
  RUEncodeData mRUEncodeEmpty; //! placeholder for empty RU data
  RUDecodeData mRUDecode;      //! buffer for decoded data

  PayLoadCont mRawBuffer; //! buffer for binary raw data

  // statistics
  std::array<RUDecodingStat, Mapping::getNRUs()> mRUDecodingStat; //! statics of decoding per FEE
  RawDecodingStat mDecodingStat;                                  //! global decoding statistics

  TStopwatch mSWIO; //! timer for IO operations

  static constexpr int RawBufferMargin = 5000000;                      // keep uploaded at least this amount
  static constexpr int RawBufferSize = 10000000 + 2 * RawBufferMargin; // size in MB
  static constexpr bool Padding128Data = true;                         // is CRU payload supposed to be padded to 128 bits

  // number of bytes the GBT word, including optional padding to 128 bits
  static constexpr int GBTWordSize = Padding128Data ? ITSMFT::GBTPaddedWordLength : o2::ITSMFT::GBTWordLength;

  ClassDefOverride(RawPixelReader, 1);
};

template <class Mapping>
constexpr int RawPixelReader<Mapping>::RawBufferMargin;

template <class Mapping>
constexpr int RawPixelReader<Mapping>::RawBufferSize;

template <class Mapping>
constexpr bool RawPixelReader<Mapping>::Padding128Data;

} // namespace ITSMFT
} // namespace o2

#endif /* ALICEO2_ITS_RAWPIXELREADER_H */
