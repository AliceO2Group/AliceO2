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

#ifndef ALICEO2_ITSMFT_ALPIDE_CODER_H
#define ALICEO2_ITSMFT_ALPIDE_CODER_H
#include <Rtypes.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <cstdint>
#include "Framework/Logger.h"
#include "PayLoadCont.h"
#include <map>
#include <fmt/format.h>
#include <iomanip>

#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/DecodingStat.h"
#include "DataFormatsITSMFT/NoiseMap.h"

#define ALPIDE_DECODING_STAT

/// \file AlpideCoder.h
/// \brief class for the ALPIDE data decoding/encoding
/// \author Ruben Shahoyan, ruben.shahoyan@cern.ch

namespace o2
{
namespace itsmft
{

/// Decoder / Encoder of ALPIDE payload stream.
/// All decoding methods are static. Only a few encoding methods are non-static but can be made so
/// if needed (will require to make the encoding buffers external to this class)

class AlpideCoder
{

 public:
  struct HitsRecord { // single record for hits (i.e. DATASHORT or DATALONG)
    HitsRecord() = default;
    ~HitsRecord() = default;
    HitsRecord(uint8_t r, uint8_t dc, uint16_t adr, uint8_t hmap) : region(r), dcolumn(dc), address(adr), hitmap(hmap) {}
    uint8_t region = 0;   // region ID
    uint8_t dcolumn = 0;  // double column ID
    uint16_t address = 0; // address in double column
    uint8_t hitmap = 0;   // hitmap for extra hits

    ClassDefNV(HitsRecord, 1); // TODO remove
  };

  struct PixLink { // single pixel on the selected row, referring eventually to the next pixel on the same row
    PixLink(short r = 0, short c = 0, int next = -1) : row(r), col(c), nextInRow(next) {}
    short row = 0;
    short col = 0;
    int nextInRow = -1; // index of the next pixel (link) on the same row

    ClassDefNV(PixLink, 1); // TODO remove
  };
  //
  static constexpr uint32_t ExpectChipHeader = 0x1 << 0;
  static constexpr uint32_t ExpectChipTrailer = 0x1 << 1;
  static constexpr uint32_t ExpectChipEmpty = 0x1 << 2;
  static constexpr uint32_t ExpectRegion = 0x1 << 3;
  static constexpr uint32_t ExpectData = 0x1 << 4;
  static constexpr uint32_t ExpectBUSY = 0x1 << 5;
  static constexpr uint32_t ExpectNextChip = ExpectChipHeader | ExpectChipEmpty;
  static constexpr int NRows = 512;
  static constexpr int RowMask = NRows - 1;
  static constexpr int NCols = 1024;
  static constexpr int NRegions = 32;
  static constexpr int NDColInReg = NCols / NRegions / 2;
  static constexpr int HitMapSize = 7;

  // masks for records components
  static constexpr uint32_t MaskEncoder = 0x3c00;                 // encoder (double column) ID takes 4 bit max (0:15)
  static constexpr uint32_t MaskPixID = 0x3ff;                    // pixel ID within encoder (double column) takes 10 bit max (0:1023)
  static constexpr uint32_t MaskDColID = MaskEncoder | MaskPixID; // mask for encoder + dcolumn combination
  static constexpr uint32_t MaskRegion = 0x1f;                    // region ID takes 5 bits max (0:31)
  static constexpr uint32_t MaskChipID = 0x0f;                    // chip id in module takes 4 bit max
  static constexpr uint32_t MaskROFlags = 0x0f;                   // RO flags in chip trailer takes 4 bit max
  static constexpr uint8_t MaskErrBusyViolation = 0x1 << 3;
  static constexpr uint8_t MaskErrDataOverrun = 0x3 << 2;
  static constexpr uint8_t MaskErrFatal = 0x7 << 1;
  static constexpr uint8_t MaskErrFlushedIncomplete = 0x1 << 2;
  static constexpr uint8_t MaskErrStrobeExtended = 0x1 << 1;
  static constexpr uint32_t MaskTimeStamp = 0xff;                 // Time stamps as BUNCH_COUNTER[10:3] bits
  static constexpr uint32_t MaskReserved = 0xff;                  // mask for reserved byte
  static constexpr uint32_t MaskHitMap = 0x7f;                    // mask for hit map: at most 7 hits in bits (0:6)
  //
  // flags for data records
  static constexpr uint32_t REGION = 0xc0;      // flag for region
  static constexpr uint32_t REGION_MASK = 0xe0; // mask for detecting the region
  static constexpr uint32_t CHIPHEADER = 0xa0;  // flag for chip header
  static constexpr uint32_t CHIPTRAILER = 0xb0; // flag for chip trailer
  static constexpr uint32_t CHIPEMPTY = 0xe0;   // flag for empty chip
  static constexpr uint32_t DATALONG = 0x0000;  // flag for DATALONG
  static constexpr uint32_t DATASHORT = 0x4000; // flag for DATASHORT
  static constexpr uint32_t BUSYOFF = 0xf0;     // flag for BUSY_OFF
  static constexpr uint32_t BUSYON = 0xf1;      // flag for BUSY_ON
  static constexpr uint32_t ERROR_MASK = 0xf0;  // flag for all error triggers

  // true if corresponds to DATALONG or DATASHORT: highest bit must be 0
  static bool isData(uint16_t v) { return (v & (0x1 << 15)) == 0; }
  static bool isData(uint8_t v) { return (v & (0x1 << 7)) == 0; }

  static constexpr int Error = -1;     // flag for decoding error
  static constexpr int EOFFlag = -100; // flag for EOF in reading

  AlpideCoder() = default;
  ~AlpideCoder() = default;

  static bool isEmptyChip(uint8_t b) { return (b & CHIPEMPTY) == CHIPEMPTY; }

  static void setNoisyPixels(const NoiseMap* noise) { mNoisyPixels = noise; }

  /// decode alpide data for the next non-empty chip from the buffer
  template <class T, typename CG>
  static int decodeChip(ChipPixelData& chipData, T& buffer, std::vector<uint16_t>& seenChips, CG cidGetter)
  {
    // read record for single non-empty chip, updating on change module and cycle.
    // return number of records filled (>0), EOFFlag or Error
    //
    bool needSorting = false; // if DColumns order is wrong, do explicit reordering
    auto roErrHandler = [&chipData](uint8_t roErr) {
#ifdef ALPIDE_DECODING_STAT
      if (roErr == MaskErrBusyViolation) {
        chipData.setError(ChipStat::BusyViolation);
      } else if (roErr == MaskErrDataOverrun) {
        chipData.setError(ChipStat::DataOverrun);
      } else if (roErr == MaskErrFatal) {
        chipData.setError(ChipStat::Fatal);
      } else if (roErr == MaskErrFlushedIncomplete) {
        chipData.setError(ChipStat::FlushedIncomplete);
      } else if (roErr == MaskErrStrobeExtended) {
        chipData.setError(ChipStat::StrobeExtended);
      }
#endif
    };

    uint8_t dataC = 0, timestamp = 0;
    uint16_t dataS = 0, region = 0;
#ifdef ALPIDE_DECODING_STAT
    uint16_t rowPrev = 0xffff;
#endif
    //
    int nRightCHits = 0;               // counter for the hits in the right column of the current double column
    std::uint16_t rightColHits[NRows]; // buffer for the accumulation of hits in the right column
    std::uint16_t colDPrev = 0xffff;   // previously processed double column (to dected change of the double column)

    uint32_t expectInp = ExpectNextChip; // data must always start with chip header or chip empty flag

    chipData.clear();
    bool dataSeen = false;
    LOG(debug) << "NewEntry";
    while (buffer.next(dataC)) {
      //
      LOGP(debug, "dataC: {:#x} expect {:#b}", int(dataC), int(expectInp));

      // Busy ON / OFF can appear at any point of the data stream, checking it with priority
      if (dataC == BUSYON) {
#ifdef ALPIDE_DECODING_STAT
        chipData.setError(ChipStat::BusyOn);
#endif
        continue;
      }
      if (dataC == BUSYOFF) {
#ifdef ALPIDE_DECODING_STAT
        chipData.setError(ChipStat::BusyOff);
#endif
        continue;
      }

      if ((expectInp & ExpectChipEmpty) && isChipEmpty(dataC)) { // empty chip was expected
        uint16_t chipIDGlo = cidGetter(dataC & MaskChipID);
        if (chipIDGlo == 0xffff) {
          chipData.setChipID(chipIDGlo);
#ifdef ALPIDE_DECODING_STAT
          chipData.setErrorInfo(dataC & MaskChipID);
          chipData.setError(ChipStat::WrongAlpideChipID);
#endif
          chipData.getData().clear();
          return unexpectedEOF("CHIP_EMPTY:WrongChipID"); // abandon cable data
        }
        chipData.setChipID(chipIDGlo); // here we set the global chip ID
        if (!buffer.next(timestamp)) {
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::TruncatedChipEmpty);
#endif
          return unexpectedEOF("CHIP_EMPTY:Timestamp"); // abandon cable data
        }
        seenChips.push_back(chipIDGlo);
        chipData.resetChipID();
        expectInp = ExpectNextChip;
        continue;
      }

      if ((expectInp & ExpectChipHeader) && isChipHeader(dataC)) { // chip header was expected
        uint16_t chipIDGlo = cidGetter(dataC & MaskChipID);
        if (chipIDGlo == 0xffff) {
          chipData.setChipID(chipIDGlo);
#ifdef ALPIDE_DECODING_STAT
          chipData.setErrorInfo(dataC & MaskChipID);
          chipData.setError(ChipStat::WrongAlpideChipID);
#endif
          chipData.getData().clear();
          return unexpectedEOF("CHIP_EMPTY:WrongChipID"); // abandon cable data
        }
        chipData.setChipID(chipIDGlo); // here we set the global chip ID
        if (!buffer.next(timestamp)) {
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::TruncatedChipHeader);
#endif
          return unexpectedEOF("CHIP_HEADER"); // abandon cable data
        }
        expectInp = ExpectRegion; // now expect region info
        dataSeen = false;
        continue;
      }

      // region info ?
      if ((expectInp & ExpectRegion) && (dataC & REGION_MASK) == REGION) { // chip header was seen, or hit data read
        region = dataC & MaskRegion;
        expectInp = ExpectData;
        continue;
      }

      if ((expectInp & ExpectChipTrailer) && isChipTrailer(dataC)) { // chip trailer was expected
        expectInp = ExpectNextChip;
        chipData.setROFlags(dataC & MaskROFlags);
#ifdef ALPIDE_DECODING_STAT
        uint8_t roErr = dataC & MaskROFlags;
        if (roErr) {
          roErrHandler(roErr);
        }
#endif
        // in case there are entries in the "right" columns buffer, add them to the container
        if (nRightCHits) {
          colDPrev++;
          for (int ihr = 0; ihr < nRightCHits; ihr++) {
            addHit(chipData, rightColHits[ihr], colDPrev);
          }
        }

        if (!dataSeen && !chipData.isErrorSet()) {
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::TrailerAfterHeader);
#endif
          return unexpectedEOF("Trailer after header"); // abandon cable data
        }
        break;
      }

      // hit info ?
      if ((expectInp & ExpectData)) {
        dataSeen = true;
        if (isData(dataC)) { // region header was seen, expect data
                             // note that here we are checking on the byte rather than the short, need complete to ushort
          dataS = dataC << 8;
          if (!buffer.next(dataC)) {
#ifdef ALPIDE_DECODING_STAT
            chipData.setError(ChipStat::TruncatedRegion);
#endif
            return unexpectedEOF("CHIPDATA"); // abandon cable data
          }
          dataS |= dataC;
          LOGP(debug, "dataC: {:#x} dataS: {:#x} expect {:#b} in ExpectData", int(dataC), int(dataS), int(expectInp));

          // we are decoding the pixel addres, if this is a DATALONG, we will fetch the mask later
          uint16_t dColID = (dataS & MaskEncoder) >> 10;
          uint16_t pixID = dataS & MaskPixID;

          // convert data to usual row/pixel format
          uint16_t row = pixID >> 1;
          // abs id of left column in double column
          uint16_t colD = (region * NDColInReg + dColID) << 1; // TODO consider <<4 instead of *NDColInReg?
          bool rightC = (row & 0x1) ? !(pixID & 0x1) : (pixID & 0x1); // true for right column / lalse for left

          if (row == rowPrev && colD == colDPrev) {
            // this is a special test to exclude repeated data of the same pixel fired
#ifdef ALPIDE_DECODING_STAT
            chipData.setError(ChipStat::RepeatingPixel);
            chipData.addErrorInfo((uint64_t(colD + rightC) << 16) | uint64_t(row));
#endif
            if ((dataS & (~MaskDColID)) == DATALONG) { // skip pattern w/o decoding
              uint8_t hitsPattern = 0;
              if (!buffer.next(hitsPattern)) {
#ifdef ALPIDE_DECODING_STAT
                chipData.setError(ChipStat::TruncatedLondData);
#endif
                return unexpectedEOF("CHIP_DATA_LONG:Pattern"); // abandon cable data
              }
              if (hitsPattern & (~MaskHitMap)) {
#ifdef ALPIDE_DECODING_STAT
                chipData.setError(ChipStat::WrongDataLongPattern);
#endif
                return unexpectedEOF("CHIP_DATA_LONG:Pattern"); // abandon cable data
              }
              LOGP(debug, "hitsPattern: {:#b} expect {:#b}", int(hitsPattern), int(expectInp));
            }
            expectInp = ExpectChipTrailer | ExpectData | ExpectRegion;
            continue; // end of DATA(SHORT or LONG) processing
          } else if (colD != colDPrev) {
            // if we start new double column, transfer the hits accumulated in the right column buffer of prev. double column
            if (colD < colDPrev && colDPrev != 0xffff) {
#ifdef ALPIDE_DECODING_STAT
              chipData.setError(ChipStat::WrongDColOrder); // abandon cable data
#endif
              return unexpectedEOF("Wrong column order"); // abandon cable data
              needSorting = true;                         // effectively disabled
            }
            colDPrev++;
            for (int ihr = 0; ihr < nRightCHits; ihr++) {
              addHit(chipData, rightColHits[ihr], colDPrev);
            }
            nRightCHits = 0; // reset the buffer
          }
          rowPrev = row;
          colDPrev = colD;

          // we want to have hits sorted in column/row, so the hits in right column of given double column
          // are first collected in the temporary buffer
          // real columnt id is col = colD + 1;
          if (rightC) {
            rightColHits[nRightCHits++] = row; // col = colD+1
          } else {
            addHit(chipData, row, colD); // col = colD, left column hits are added directly to the container
          }

          if ((dataS & (~MaskDColID)) == DATALONG) { // multiple hits ?
            uint8_t hitsPattern = 0;
            if (!buffer.next(hitsPattern)) {
#ifdef ALPIDE_DECODING_STAT
              chipData.setError(ChipStat::TruncatedLondData);
#endif
              return unexpectedEOF("CHIP_DATA_LONG:Pattern"); // abandon cable data
            }
            LOGP(debug, "hitsPattern: {:#b} expect {:#b}", int(hitsPattern), int(expectInp));
            if (hitsPattern & (~MaskHitMap)) {
#ifdef ALPIDE_DECODING_STAT
              chipData.setError(ChipStat::WrongDataLongPattern);
#endif
              return unexpectedEOF("CHIP_DATA_LONG:Pattern"); // abandon cable data
            }
            for (int ip = 0; ip < HitMapSize; ip++) {
              if (hitsPattern & (0x1 << ip)) {
                uint16_t addr = pixID + ip + 1, rowE = addr >> 1;
                if (addr & ~MaskPixID) {
#ifdef ALPIDE_DECODING_STAT
                  chipData.setError(ChipStat::WrongRow);
#endif
                  return unexpectedEOF(fmt::format("Non-existing encoder {} decoded, DataLong was {:x}", pixID, dataS)); // abandon cable data
                }
                rightC = ((rowE & 0x1) ? !(addr & 0x1) : (addr & 0x1)); // true for right column / lalse for left
                // the real columnt is int colE = colD + rightC;
                if (rightC) { // same as above
                  rightColHits[nRightCHits++] = rowE;
                } else {
                  addHit(chipData, rowE, colD + rightC); // left column hits are added directly to the container
                }
              }
            }
          }
        } else if (ChipStat::getAPENonCritical(dataC) >= 0) { // check for recoverable APE, if on: continue with ExpectChipTrailer | ExpectData | ExpectRegion expectation
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::DecErrors(ChipStat::getAPENonCritical(dataC)));
#endif
        } else {
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::NoDataFound);
#endif
          return unexpectedEOF(fmt::format("Expected DataShort or DataLong mask, got {:x}", dataS)); // abandon cable data
        }
        expectInp = ExpectChipTrailer | ExpectData | ExpectRegion;
        continue; // end of DATA(SHORT or LONG) processing
      }

      if (!dataC) {
        if (expectInp == ExpectNextChip) {
          continue;
        }
        chipData.setError(ChipStat::TruncatedBuffer);
        return unexpectedEOF("Abandon on 0-padding"); // abandon cable data
      }

      // in case of BUSY VIOLATION the Trailer may come directly after the Header
      if ((expectInp & ExpectRegion) && isChipTrailer(dataC) && (dataC & MaskROFlags)) {
        expectInp = ExpectNextChip;
        chipData.setROFlags(dataC & MaskROFlags);
        roErrHandler(dataC & MaskROFlags);
        break;
      }

      // check for APE errors, see https://alice.its.cern.ch/jira/browse/O2-1717?focusedCommentId=274714&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-274714
      bool fatalAPE = false;
      auto codeAPE = ChipStat::getAPECode(dataC, fatalAPE);
      if (codeAPE >= 0) {
#ifdef ALPIDE_DECODING_STAT
        chipData.setError(ChipStat::DecErrors(codeAPE));
#endif
        if (fatalAPE) {
          return unexpectedEOF(fmt::format("APE error {:#02x} [expectation = {:#02x}]", int(dataC), int(expectInp))); // abandon cable data
        } else {
          LOGP(error, "Code should not have entered here, APE: {:#02x}, expectation: {:#02x}", codeAPE, int(expectInp));
          return unexpectedEOF(fmt::format("APE error {:#02x} [expectation = {:#02x}]", int(dataC), int(expectInp))); // abandon cable data
        }
      }
#ifdef ALPIDE_DECODING_STAT
      chipData.setError(ChipStat::UnknownWord);
      // fill the error buffer with a few bytes of wrong data
      const uint8_t* begPtr = buffer.data();
      const uint8_t* endPtr = buffer.getEnd();
      const uint8_t* curPtr = buffer.getPtr();
      size_t offsBack = std::min(ChipPixelData::MAXDATAERRBYTES - ChipPixelData::MAXDATAERRBYTES_AFTER, size_t(curPtr - begPtr));
      size_t offsAfter = std::min(ChipPixelData::MAXDATAERRBYTES_AFTER, size_t(endPtr - curPtr));
      std::memcpy(chipData.getRawErrBuff().data(), curPtr - offsBack, offsBack + offsAfter);
      chipData.setNBytesInRawBuff(offsBack + offsAfter);
#endif
      return unexpectedEOF(fmt::format("Unknown word 0x{:x} [expectation = 0x{:x}]", int(dataC), int(expectInp))); // abandon cable data
    }

    if (!(expectInp & ExpectNextChip)) {
#ifdef ALPIDE_DECODING_STAT
      chipData.setError(ChipStat::TruncatedRegion);
#endif
      return unexpectedEOF("Missing CHIP_TRAILER"); // abandon cable data
    }

    if (needSorting && chipData.getData().size()) { // d.columns were in a wrong order, need to sort the data, RS: effectively disabled
      LOGP(error, "This code path should have been disabled");
      auto& pixData = chipData.getData();
      std::sort(pixData.begin(), pixData.end(),
                [](PixelData& a, PixelData& b) { return a.getCol() < b.getCol() || (a.getCol() == b.getCol() && a.getRowDirect() < b.getRowDirect()); });
      // if the columns ordering was wrong, detection of same pixel fired twice might have failed, make sure there are no duplicates
      auto currPix = pixData.begin(), prevPix = currPix++;
      while (currPix != pixData.end()) {
        if (prevPix->getCol() == currPix->getCol() && prevPix->getRowDirect() == currPix->getRowDirect()) {
          currPix = pixData.erase(prevPix);
        }
        prevPix = currPix++;
      }
    }
    if (chipData.getData().size()) {
      seenChips.push_back(chipData.getChipID());
    }
    return chipData.getData().size();
  }

  /// Verifies the decoder by comparing the contents a cable by re-encoding seen
  /// chips back into the ALPIDE format.
  template <typename LG, typename CG>
  static bool verifyDecodedCable(
    std::map<int, ChipPixelData*>& seenChips, PayLoadCont& buffer,
    std::vector<uint16_t>& seenChipIDs, LG lidGetter, CG cidGetter)
  {
    PayLoadCont reconstructedData;

    // Ensure the length of the reconstructed buffer.
    int bufferLength = 0;
    for (auto it = seenChips.begin(); it != seenChips.end(); ++it) {
      bufferLength += it->second->getData().size();
    }
    bufferLength += seenChipIDs.size() * 2;
    reconstructedData.ensureFreeCapacity(40 * bufferLength);

    // Encode the seen chips in the order they were decoded.
    for (int ID : seenChipIDs) {
      ChipPixelData currentChip;
      int localID = lidGetter(ID);
      if (seenChips.count(ID)) {
        currentChip = *seenChips[ID];
      }
      AlpideCoder encoder;
      encoder.encodeChip(reconstructedData, currentChip, localID,
                         /*dummy bc*/ 0, currentChip.getROFlags());
    }

    // Pad the end of the reconstructed buffer with the zero bytes.
    if (buffer.getSize() > reconstructedData.getSize())
      reconstructedData.fill(0x00,
                             buffer.getSize() - reconstructedData.getSize());

    auto hexToString = [](uint8_t v) {
      std::stringstream ss;
      ss << "0x" << std::setfill('0') << std::setw(2) << std::hex
         << std::uppercase << (0xFF & v);
      return ss.str();
    };

    auto reportError = [&](std::string message) {
      LOG(error) << "Error during decoder verification: " << message;
      LOG(debug) << "Raw Data:";
      buffer.rewind();
      uint8_t dataC = 0;
      int index = 1;
      while (buffer.next(dataC)) {
        LOG(debug) << index++ << ". " << hexToString(dataC);
      }
      LOG(debug) << "Reconstructed Data:";
      reconstructedData.rewind();
      index = 1;
      while (reconstructedData.next(dataC)) {
        LOG(debug) << index++ << ". " << hexToString(dataC);
      }
    };

    // The reconstructed buffer is very similar to the original data flow
    // with the exception to several pieces of information that get lost
    // during the decoding:
    // 1. Error trigger words: these are absent in the reconstructed buffer.
    //    In case of BUSYON/BUSYOFF words, the verification is allowed to
    //    continue.
    // 2. Bunch counter for frame: the reconstructed buffer does contain the
    //    corresponding words, but has a dummy value.

    ChipPixelData* currentChip = nullptr;
    if (seenChipIDs.size())
      currentChip = seenChips[seenChipIDs[0]];
    buffer.rewind();
    while (true) {
      uint8_t dataRec = 0;
      uint8_t dataRaw = 0;

      if (reconstructedData.isEmpty() || buffer.isEmpty()) {
        // If either buffer is empty, verify that both buffers reached the end.
        // If one of the streams is non-empty, then verify that the remaining
        // bytes are zeroes.
        if (reconstructedData.isEmpty() && buffer.isEmpty()) {
          break;
        }
        PayLoadCont& nonEmptyBuffer =
          !buffer.isEmpty() ? buffer : reconstructedData;
        uint8_t dataC = 0;
        while (nonEmptyBuffer.next(dataC)) {
          if (dataC != 0x00) {
            reportError("Buffer sizes mismatch.");
            return false;
          }
        }
        break;
      }

      reconstructedData.current(dataRec);
      buffer.current(dataRaw);
      if (dataRaw == dataRec) {
        if (isChipHeaderOrEmpty(dataRaw)) {
          uint16_t ID = cidGetter(dataRaw & MaskChipID);
          if (seenChips.count(ID)) {
            currentChip = seenChips[ID];
          } else {
            currentChip = nullptr;
          }
          // If the data correspond to the CHIPHEADER or CHIPEMPTY data words,
          // skip the next byte that represent bunch counters.
          buffer.next(dataRaw);
          reconstructedData.next(dataRec);
        }
        buffer.next(dataRaw);
        reconstructedData.next(dataRec);
        continue;
      }

      if (dataRaw == BUSYON || dataRaw == BUSYOFF) {
        // Placement of BUSYON and BUSYOFF triggers is arbitrary, just ignore
        // the byte in the raw stream and move forward.
        buffer.next(dataRaw);
        continue;
      }

      VerifierMismatchResult res =
        handleVerifierMismatch(buffer, reconstructedData, currentChip);
      switch (res) {
        case VerifierMismatchResult::RESOLVED:
          LOG(debug) << "Mismatch " << hexToString(dataRaw) << " / "
                     << hexToString(dataRec)
                     << " was resolved, able to continue verification";
          continue;
        case VerifierMismatchResult::EXPECTED_MISMATCH:
          LOG(debug) << "Mismatch " << hexToString(dataRaw) << " / "
                     << hexToString(dataRec)
                     << " was expected, aborting the verification";
          return true;
        case VerifierMismatchResult::UNEXPECTED_MISMATCH: {
          // If the read bytes is not related to the special cases, report
          // error.
          std::stringstream errorStream;
          errorStream
            << "Unexpected byte mismatch during decoder verification. "
               "Expected: "
            << hexToString(dataRaw)
            << ", Reconstructed: " << hexToString(dataRec);
          reportError(errorStream.str());
        }
      }
      return false;
    }
    return true;
  }

  enum VerifierMismatchResult {
    UNEXPECTED_MISMATCH, // Genuine mismatch, stop verification
    EXPECTED_MISMATCH,   // Mismatch expected, need to abort verification
    RESOLVED             // Mismatch resolved, can continue
  };

  static VerifierMismatchResult handleVerifierMismatch(
    PayLoadCont& buffer, PayLoadCont& reconstructedData,
    ChipPixelData* currentChip)
  {
    VerifierMismatchResult res = VerifierMismatchResult::UNEXPECTED_MISMATCH;
    uint8_t dataRec = 0;
    uint8_t dataRaw = 0;
    reconstructedData.current(dataRec);
    buffer.current(dataRaw);
    auto inner = [&](int errIdx) {
      if (res != VerifierMismatchResult::UNEXPECTED_MISMATCH ||
          dataRaw == dataRec) {
        // The mismatch was resolved, no need to check the rest of the errors
        return;
      }
      switch (errIdx) {
        case ChipStat::BusyViolation:
        case ChipStat::DataOverrun:
        case ChipStat::Fatal:
        case ChipStat::BusyOn:
        case ChipStat::BusyOff:
        case ChipStat::TruncatedChipEmpty:
        case ChipStat::TruncatedChipHeader:
        case ChipStat::TruncatedRegion:
        case ChipStat::TruncatedLondData:
        case ChipStat::WrongDataLongPattern:
        case ChipStat::NoDataFound:
        case ChipStat::UnknownWord:
        case ChipStat::RepeatingPixel:
        case ChipStat::WrongRow:
        case ChipStat::APE_STRIP_START:
        case ChipStat::APE_ILLEGAL_CHIPID:
        case ChipStat::APE_DET_TIMEOUT:
        case ChipStat::APE_OOT:
        case ChipStat::APE_PROTOCOL_ERROR:
        case ChipStat::APE_LANE_FIFO_OVERFLOW_ERROR:
        case ChipStat::APE_FSM_ERROR:
        case ChipStat::APE_PENDING_DETECTOR_EVENT_LIMIT:
        case ChipStat::APE_PENDING_LANE_EVENT_LIMIT:
        case ChipStat::APE_O2N_ERROR:
        case ChipStat::APE_RATE_MISSING_TRG_ERROR:
        case ChipStat::APE_PE_DATA_MISSING:
        case ChipStat::APE_OOT_DATA_MISSING:
        case ChipStat::WrongDColOrder:
        case ChipStat::InterleavedChipData:
        case ChipStat::TruncatedBuffer:
        case ChipStat::TrailerAfterHeader:
        case ChipStat::FlushedIncomplete:
        case ChipStat::StrobeExtended:
        case ChipStat::WrongAlpideChipID:
          break;
        default:
          LOG(error) << "Unknown error set by chip during verifier mismatch";
      }
    };
    if (currentChip) {
      currentChip->forEachSetError(inner);
    }
    return res;
  }

  static bool isChipEmpty(uint8_t v) { return (v & (~MaskChipID)) == CHIPEMPTY; }
  static bool isChipHeader(uint8_t v) { return (v & (~MaskChipID)) == CHIPHEADER; }
  static bool isChipTrailer(uint8_t v) { return (v & (~MaskChipID)) == CHIPTRAILER; }

  /// check if the byte corresponds to chip_header or chip_empty flag
  static bool isChipHeaderOrEmpty(uint8_t v)
  {
    return isChipHeader(v) || isChipEmpty(v);
  }
  // methods to use for data encoding

  static uint8_t bc2TimeStamp(int bc) { return (bc >> 3) & MaskTimeStamp; }
  static uint16_t timeStamp2BC(uint8_t ts) { return uint16_t(ts) << 3; }

  int encodeChip(PayLoadCont& buffer, const o2::itsmft::ChipPixelData& chipData,
                 uint16_t chipInModule, uint16_t bc, uint16_t roflags = 0);

  // Add empty record for the chip with chipID within its module for the bc
  void addEmptyChip(PayLoadCont& buffer, int chipInMod, int bc)
  {
    buffer.addFast(makeChipEmpty(chipInMod, bc));
  }
  //
  void print() const;
  void reset();

 private:
  /// Output a non-noisy fired pixel
  static void addHit(ChipPixelData& chipData, short row, short col)
  {
    if (mNoisyPixels) {
      auto chipID = chipData.getChipID();
      if (mNoisyPixels->isNoisy(chipID, row, col)) {
        return;
      }
    }
    LOGP(debug, "Add hit#{} at r:{}/c:{} of chip:{}", chipData.getData().size(), row, col, chipData.getChipID());
    chipData.getData().emplace_back(row, col);
  }

  ///< add pixed to compressed matrix, the data must be provided sorted in row/col, no check is done
  void addPixel(short row, short col)
  {
    int last = mPix2Encode.size();
    mPix2Encode.emplace_back(row, col);
    if (last && row == mPix2Encode[last - 1].row) { // extend current row
      mPix2Encode[last - 1].nextInRow = last;       // refer to new link in the same row
    } else {                                        // create new row
      mFirstInRow.push_back(last);
    }
  }

  ///< prepare chip header: 1010<chip id[3:0]><BUNCH COUNTER FOR FRAME[10:3] >
  static uint16_t makeChipHeader(short chipID, short bc)
  {
    uint16_t v = CHIPHEADER | (MaskChipID & chipID);
    v = (v << 8) | bc2TimeStamp(bc);
    return v;
  }

  ///< prepare chip trailer: 1011<readout flags[3:0]>
  static uint8_t makeChipTrailer(short roflags)
  {
    uint8_t v = CHIPTRAILER | (MaskROFlags & roflags);
    return v;
  }

  ///< prepare chip empty marker: 1110<chip id[3:0]><BUNCH COUNTER FOR FRAME[10:3] >
  static uint16_t makeChipEmpty(short chipID, short bc)
  {
    uint16_t v = CHIPEMPTY | (MaskChipID & chipID);
    v = (v << 8) | bc2TimeStamp(bc);
    return v;
  }

  ///< packs the address of region
  static uint8_t makeRegion(short reg)
  {
    uint8_t v = REGION | (reg & MaskRegion);
    return v;
  }

  ///< packs the address for data short
  static uint16_t makeDataShort(short encoder, short address)
  {
    uint16_t v = DATASHORT | (MaskEncoder & (encoder << 10)) | (address & MaskPixID);
    return v;
  }

  // packs the address for data long
  static uint16_t makeDataLong(short encoder, short address)
  {
    uint16_t v = DATALONG | (MaskEncoder & (encoder << 10)) | (address & MaskPixID);
    return v;
  }

  // ENCODING: converting hitmap to raw data
  int procDoubleCol(PayLoadCont& buffer, short reg, short dcol);

  ///< process region (16 double columns)
  int procRegion(PayLoadCont& buffer, short reg)
  {
    int nfound = 0;
    for (int idc = 0; idc < NDColInReg; idc++) {
      nfound += procDoubleCol(buffer, reg, idc);
    }
    return nfound;
  }

  void resetMap();

  ///< error message on unexpected EOF
  static int unexpectedEOF(const std::string& message)
  {
    LOG(debug) << message;
    return Error;
  }

  // =====================================================================
  //

  static const NoiseMap* mNoisyPixels;

  // cluster map used for the ENCODING only
  std::vector<int> mFirstInRow;     //! entry of 1st pixel of each non-empty row in the mPix2Encode
  std::vector<PixLink> mPix2Encode; //! pool of links: fired pixel + index of the next one in the row
  //
  ClassDefNV(AlpideCoder, 3);
};

} // namespace itsmft
} // namespace o2

#endif
