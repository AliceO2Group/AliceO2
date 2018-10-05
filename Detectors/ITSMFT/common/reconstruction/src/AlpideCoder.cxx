// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlpideCoder.cxx
/// \brief Implementation of the ALPIDE data decoding/encoding

#include "ITSMFTReconstruction/AlpideCoder.h"
#include <TClass.h>

//#define _RAW_READER_DEBUG_ // to produce debug output during decoding

using namespace o2::ITSMFT;

//_____________________________________
void AlpideCoder::print() const
{
  for (auto id : mFirstInRow) {
    auto link = mPix2Encode[id];
    printf("r%3d | c%4d", link.row, link.col);
    while (link.nextInRow != -1) {
      link = mPix2Encode[link.nextInRow];
      printf(" %3d", link.col);
    }
    printf("\n");
  }
}

//_____________________________________
void AlpideCoder::reset()
{
  // reset before processing next chip
  mFirstInRow.clear();
  mPix2Encode.clear();
}

//_____________________________________
int AlpideCoder::encodeChip(PayLoadCont& buffer, const o2::ITSMFT::ChipPixelData& chipData,
                            uint16_t chipInModule, uint16_t bc, uint16_t roflags)
{
  // Encode chip data into provided buffer. Data must be provided sorted in row/col, no check is done
  int nfound = 0;
  auto pixels = chipData.getData();
  if (pixels.size()) { // non-empty chip
    for (const auto& pix : pixels) {
      addPixel(pix.getRow(), pix.getCol());
    }
    buffer.addFast(makeChipHeader(chipInModule, bc)); // chip header
    for (int ir = NRegions; ir--;) {
      nfound += procRegion(buffer, ir);
    }
    buffer.addFast(makeChipTrailer(roflags));
    resetMap();
  } else {
    buffer.addFast(makeChipEmpty(chipInModule, bc));
  }
  //
  return nfound;
}

//_____________________________________
int AlpideCoder::procDoubleCol(PayLoadCont& buffer, short reg, short dcol)
{
  // process double column: encoding
  short hits[2 * NRows];
  int nHits = 0, nData = 0;
  //
  int nr = mFirstInRow.size();
  short col0 = ((reg * NDColInReg + dcol) << 1), col1 = col0 + 1; // 1st,2nd column of double column
  int prevRow = -1;
  for (int ir = 0; ir < nr; ir++) {
    int linkID = mFirstInRow[ir];
    if (linkID == -1 || mPix2Encode[linkID].col > col1) { // no pixels left on this row or higher column IDs
      continue;
    }
    short rowID = mPix2Encode[linkID].row;
    // process fired pixels
    bool left = 0, right = 0;
    if (mPix2Encode[linkID].col == col0) { // pixel in left column
      left = 1;
      linkID = mPix2Encode[linkID].nextInRow; // unlink processed pixel
    }
    if (linkID != -1 && mPix2Encode[linkID].col == col1) { // pixel in right column
      right = 1;
      linkID = mPix2Encode[linkID].nextInRow; // unlink processed pixel
    }
    short addr0 = rowID << 1;
    if (rowID & 0x1) { // odd rows: right to left numbering
      if (right) {
        hits[nHits++] = addr0;
      }
      if (left) {
        hits[nHits++] = addr0 + 1;
      }
    } else { // even rows: left to right numbering
      if (left) {
        hits[nHits++] = addr0;
      }
      if (right) {
        hits[nHits++] = addr0 + 1;
      }
    }
    if (linkID == -1) {
      mFirstInRow[ir] = -1;
      continue;
    }                         // this row is finished
    mFirstInRow[ir] = linkID; // link remaining hit pixels to row
  }
  //
  int ih = 0;
  if (nHits) {
    buffer.addFast(makeRegion(reg)); // flag region start
  }

  while ((ih < nHits)) {
    short addrE, addrW = hits[ih++]; // address of the reference hit
    uint8_t mask = 0, npx = 0;
    short addrLim = addrW + HitMapSize + 1; // 1+address of furthest hit can be put in the map
    while ((ih < nHits && (addrE = hits[ih]) < addrLim)) {
      mask |= 0x1 << (addrE - addrW - 1);
      ih++;
    }
    if (mask) { // flag DATALONG
      buffer.addFast(makeDataLong(dcol, addrW));
      buffer.addFast(mask);
    } else {
      buffer.addFast(makeDataShort(dcol, addrW));
    }
    nData++;
  }
  //
  return nData;
}


//_____________________________________
void AlpideCoder::resetMap()
{
  // reset map of hits for current chip
  mFirstInRow.clear();
  mPix2Encode.clear();
}

//_____________________________________
int AlpideCoder::decodeChip(ChipPixelData& chipData, PayLoadCont& buffer)
{
  // read record for single non-empty chip, updating on change module and cycle.
  // return number of records filled (>0), EOFFlag or Error
  // NOTE: decoder does not clean the chipData buffers, should be done outside
  //
  uint8_t dataC = 0, timestamp = 0;
  uint16_t dataS = 0, region = 0;
  //
  int nRightCHits = 0;               // counter for the hits in the right column of the current double column
  std::uint16_t rightColHits[NRows]; // buffer for the accumulation of hits in the right column
  std::uint16_t colDPrev = 0xffff;   // previously processed double column (to dected change of the double column)

  mExpectInp = ExpectChipHeader | ExpectChipEmpty; // data must always start with chip header or chip empty flag

  while (buffer.next(dataC)) {
    //
    // ---------- chip info ?
    uint8_t dataCM = dataC & (~MaskChipID);
    //
    if ((mExpectInp & ExpectChipEmpty) && dataCM == CHIPEMPTY) { // chip trailer was expected
      chipData.setChipID(dataC & MaskChipID);                    // here we set the chip ID within the module
      if (!buffer.next(timestamp)) {
        return unexpectedEOF("CHIP_EMPTY:Timestamp");
      }
      mExpectInp = ExpectChipHeader | ExpectChipEmpty;
      continue;
    }

    if ((mExpectInp & ExpectChipHeader) && dataCM == CHIPHEADER) { // chip header was expected
      chipData.setChipID(dataC & MaskChipID);                      // here we set the chip ID within the module
      if (!buffer.next(timestamp)) {
        return unexpectedEOF("CHIP_HEADER");
      }
      mExpectInp = ExpectRegion; // now expect region info
      continue;
    }

    // region info ?
    if ((mExpectInp & ExpectRegion) && (dataC & REGION) == REGION) { // chip header was seen, or hit data read
      region = dataC & MaskRegion;
      mExpectInp = ExpectData;
      continue;
    }

    if ((mExpectInp & ExpectChipTrailer) && dataCM == CHIPTRAILER) { // chip trailer was expected
      mExpectInp = ExpectChipHeader | ExpectChipEmpty;

      // in case there are entries in the "right" columns buffer, add them to the container
      if (nRightCHits) {
        colDPrev++;
        for (int ihr = 0; ihr < nRightCHits; ihr++) {
          chipData.getData().emplace_back(rightColHits[ihr], colDPrev);
        }
      }
      break;
    }

    // hit info ?
    if ((mExpectInp & ExpectData)) {
      if (isData(dataC)) { // region header was seen, expect data
        // TODO note that here we are checking on the byte rather than the short, need to stepBack

        buffer.stepBack(); // need to reinterpred as short
        if (!buffer.next(dataS)) {
          return unexpectedEOF("CHIPDATA");
        }
        // we are decoding the pixel addres, if this is a DATALONG, we will fetch the mask later
        uint16_t dColID = (dataS & MaskEncoder) >> 10;
        uint16_t pixID = dataS & MaskPixID;

        // convert data to usual row/pixel format
        uint16_t row = pixID >> 1;
        // abs id of left column in double column
        uint16_t colD = (region * NDColInReg + dColID) << 1; // TODO consider <<4 instead of *NDColInReg?

#ifdef _RAW_READER_DEBUG_
        printf("Reg:%3d DCol:%2d Addr:%3d | ", region, dColID, pixID);
#endif

        // if we start new double column, transfer the hits accumulated in the right column buffer of prev. double column
        if (colD != colDPrev) {
          colDPrev++;
          for (int ihr = 0; ihr < nRightCHits; ihr++) {
            chipData.getData().emplace_back(rightColHits[ihr], colDPrev);
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
          chipData.getData().emplace_back(row, colD); // col = colD, left column hits are added directly to the container
        }
#ifdef _RAW_READER_DEBUG_
        printf("%04d/%03d ", colD + rightC, row);
#endif
        if ((dataS & (~MaskDColID)) == DATALONG) { // multiple hits ?
          uint8_t hitsPattern = 0;
          if (!buffer.next(hitsPattern)) {
            return unexpectedEOF("CHIP_DATA_LONG:Pattern");
          }
#ifdef _RAW_READER_DEBUG_
          printf(" [ ");
#endif
          for (int ip = 0; ip < HitMapSize; ip++) {
            if (hitsPattern & (0x1 << ip)) {
              uint16_t addr = pixID + ip + 1, rowE = addr >> 1;
              rightC = ((rowE & 0x1) ? !(addr & 0x1) : (addr & 0x1)); // true for right column / lalse for left
              // the real columnt is int colE = colD + rightC;
              if (rightC) { // same as above
                rightColHits[nRightCHits++] = rowE;
              } else {
                chipData.getData().emplace_back(rowE, colD + rightC); // left column hits are added directly to the container
              }
#ifdef _RAW_READER_DEBUG_
              printf("%04d/%03d ", colD + rightC, rowE);
#endif
            }
          }
#ifdef _RAW_READER_DEBUG_
          printf("]\n");
#endif
        }
#ifdef _RAW_READER_DEBUG_
        else {
          printf("\n");
        }
#endif
      } else {
        LOG(ERROR) << "Expected DataShort or DataLong mask, got : " << dataS;
        return Error;
      }
      mExpectInp = ExpectChipTrailer | ExpectData | ExpectRegion;
      continue; // end of DATA(SHORT or LONG) processing
    }

    if (!dataC) {
      break; // 0 padding reached (end of the cable data)
    }
    return unexpectedEOF("Unknown word"); // either error
  }

  return chipData.getData().size();
}
