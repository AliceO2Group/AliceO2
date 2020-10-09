// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  static constexpr int NRows = 512;
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
  static constexpr uint32_t MaskTimeStamp = 0xff;                 // Time stamps as BUNCH_COUNTER[10:3] bits
  static constexpr uint32_t MaskReserved = 0xff;                  // mask for reserved byte
  static constexpr uint32_t MaskHitMap = 0x7f;                    // mask for hit map: at most 7 hits in bits (0:6)
  //
  // flags for data records
  static constexpr uint32_t REGION = 0xc0;      // flag for region
  static constexpr uint32_t CHIPHEADER = 0xa0;  // flag for chip header
  static constexpr uint32_t CHIPTRAILER = 0xb0; // flag for chip trailer
  static constexpr uint32_t CHIPEMPTY = 0xe0;   // flag for empty chip
  static constexpr uint32_t DATALONG = 0x0000;  // flag for DATALONG
  static constexpr uint32_t DATASHORT = 0x4000; // flag for DATASHORT
  static constexpr uint32_t BUSYOFF = 0xf0;     // flag for BUSY_OFF
  static constexpr uint32_t BUSYON = 0xf1;      // flag for BUSY_ON

  // true if corresponds to DATALONG or DATASHORT: highest bit must be 0
  static bool isData(uint16_t v) { return (v & (0x1 << 15)) == 0; }
  static bool isData(uint8_t v) { return (v & (0x1 << 7)) == 0; }

  static constexpr int Error = -1;     // flag for decoding error
  static constexpr int EOFFlag = -100; // flag for EOF in reading

  AlpideCoder() = default;
  ~AlpideCoder() = default;

  static bool isEmptyChip(uint8_t b) { return (b & CHIPEMPTY) == CHIPEMPTY; }

  static void setNoisyPixels(const NoiseMap* noise) { mNoisyPixels = noise; }
  static void setNoiseThreshold(int t) { mNoiseThreshold = t; }

  /// decode alpide data for the next non-empty chip from the buffer
  template <class T>
  static int decodeChip(ChipPixelData& chipData, T& buffer)
  {
    // read record for single non-empty chip, updating on change module and cycle.
    // return number of records filled (>0), EOFFlag or Error
    //
    uint8_t dataC = 0, timestamp = 0;
    uint16_t dataS = 0, region = 0;
    //
    int nRightCHits = 0;               // counter for the hits in the right column of the current double column
    std::uint16_t rightColHits[NRows]; // buffer for the accumulation of hits in the right column
    std::uint16_t colDPrev = 0xffff;   // previously processed double column (to dected change of the double column)

    uint32_t expectInp = ExpectChipHeader | ExpectChipEmpty; // data must always start with chip header or chip empty flag

    chipData.clear();

    while (buffer.next(dataC)) {
      //
      // ---------- chip info ?
      uint8_t dataCM = dataC & (~MaskChipID);
      //
      if ((expectInp & ExpectChipEmpty) && dataCM == CHIPEMPTY) { // empty chip was expected
        //chipData.setChipID(dataC & MaskChipID);                   // here we set the chip ID within the module // now set upstream
        if (!buffer.next(timestamp)) {
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::TruncatedChipEmpty);
#endif
          return unexpectedEOF("CHIP_EMPTY:Timestamp");
        }
        expectInp = ExpectChipHeader | ExpectChipEmpty;
        continue;
      }

      if ((expectInp & ExpectChipHeader) && dataCM == CHIPHEADER) { // chip header was expected
        //chipData.setChipID(dataC & MaskChipID);                     // here we set the chip ID within the module // now set upstream
        if (!buffer.next(timestamp)) {
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::TruncatedChipHeader);
#endif
          return unexpectedEOF("CHIP_HEADER");
        }
        expectInp = ExpectRegion; // now expect region info
        continue;
      }

      // region info ?
      if ((expectInp & ExpectRegion) && (dataC & REGION) == REGION) { // chip header was seen, or hit data read
        region = dataC & MaskRegion;
        expectInp = ExpectData;
        continue;
      }

      if ((expectInp & ExpectChipTrailer) && dataCM == CHIPTRAILER) { // chip trailer was expected
        expectInp = ExpectChipHeader | ExpectChipEmpty;
        chipData.setROFlags(dataC & MaskROFlags);
#ifdef ALPIDE_DECODING_STAT
        uint8_t roErr = dataC & MaskROFlags;
        if (roErr) {
          if (roErr == MaskErrBusyViolation) {
            chipData.setError(ChipStat::BusyViolation);
          } else if (roErr == MaskErrDataOverrun) {
            chipData.setError(ChipStat::DataOverrun);
          } else if (roErr == MaskErrFatal) {
            chipData.setError(ChipStat::Fatal);
          }
        }
#endif
        // in case there are entries in the "right" columns buffer, add them to the container
        if (nRightCHits) {
          colDPrev++;
          for (int ihr = 0; ihr < nRightCHits; ihr++) {
            addHit(chipData, rightColHits[ihr], colDPrev);
          }
        }
        break;
      }

      // hit info ?
      if ((expectInp & ExpectData)) {
        if (isData(dataC)) { // region header was seen, expect data
                             // note that here we are checking on the byte rather than the short, need complete to ushort
          dataS = dataC << 8;
          if (!buffer.next(dataC)) {
#ifdef ALPIDE_DECODING_STAT
            chipData.setError(ChipStat::TruncatedRegion);
#endif
            return unexpectedEOF("CHIPDATA");
          }
          dataS |= dataC;
          // we are decoding the pixel addres, if this is a DATALONG, we will fetch the mask later
          uint16_t dColID = (dataS & MaskEncoder) >> 10;
          uint16_t pixID = dataS & MaskPixID;

          // convert data to usual row/pixel format
          uint16_t row = pixID >> 1;
          // abs id of left column in double column
          uint16_t colD = (region * NDColInReg + dColID) << 1; // TODO consider <<4 instead of *NDColInReg?

          // if we start new double column, transfer the hits accumulated in the right column buffer of prev. double column
          if (colD != colDPrev) {
            colDPrev++;
            for (int ihr = 0; ihr < nRightCHits; ihr++) {
              addHit(chipData, rightColHits[ihr], colDPrev);
            }
            colDPrev = colD;
            nRightCHits = 0; // reset the buffer
          }

          bool rightC = (row & 0x1) ? !(pixID & 0x1) : (pixID & 0x1); // true for right column / lalse for left

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
              return unexpectedEOF("CHIP_DATA_LONG:Pattern");
            }
#ifdef ALPIDE_DECODING_STAT
            if (hitsPattern & (~MaskHitMap)) {
              chipData.setError(ChipStat::WrongDataLongPattern);
            }
#endif
            for (int ip = 0; ip < HitMapSize; ip++) {
              if (hitsPattern & (0x1 << ip)) {
                uint16_t addr = pixID + ip + 1, rowE = addr >> 1;
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
        } else {
#ifdef ALPIDE_DECODING_STAT
          chipData.setError(ChipStat::NoDataFound);
#endif
          LOG(ERROR) << "Expected DataShort or DataLong mask, got : " << dataS;
          return Error;
        }
        expectInp = ExpectChipTrailer | ExpectData | ExpectRegion;
        continue; // end of DATA(SHORT or LONG) processing
      }

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

      if (!dataC) {
        buffer.clear(); // 0 padding reached (end of the cable data), no point in continuing
        break;
      }
#ifdef ALPIDE_DECODING_STAT
      chipData.setError(ChipStat::UnknownWord);
#endif
      return unexpectedEOF(fmt::format("Unknown word 0x{:x} [expectation = 0x{:x}]", int(dataC), int(expectInp))); // error
    }

    return chipData.getData().size();
  }

  /// check if the byte corresponds to chip_header or chip_empty flag
  static bool isChipHeaderOrEmpty(uint8_t v)
  {
    v &= (~MaskChipID);
    return (v == CHIPEMPTY) || (v == CHIPHEADER);
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
  //
  template <class T>
  static int getChipID(T& buffer)
  {
    uint8_t id = 0;
    return (buffer.current(id) && isChipHeaderOrEmpty(id)) ? (id & AlpideCoder::MaskChipID) : -1;
  }

 private:
  /// Output a non-noisy fired pixel
  static void addHit(ChipPixelData& chipData, short row, short col)
  {
    if (mNoisyPixels) {
      auto chipID = chipData.getChipID();
      if (mNoisyPixels->getNoiseLevel(chipID, row, col) > mNoiseThreshold) {
        return;
      }
    }

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
  static int unexpectedEOF(const std::string& message);

  // =====================================================================
  //

  static const NoiseMap* mNoisyPixels;
  static int mNoiseThreshold;

  // cluster map used for the ENCODING only
  std::vector<int> mFirstInRow;     //! entry of 1st pixel of each non-empty row in the mPix2Encode
  std::vector<PixLink> mPix2Encode; //! pool of links: fired pixel + index of the next one in the row
  //
  ClassDefNV(AlpideCoder, 2);
};

} // namespace itsmft
} // namespace o2

#endif
