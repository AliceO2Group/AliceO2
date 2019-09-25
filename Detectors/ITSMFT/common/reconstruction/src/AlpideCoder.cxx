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

using namespace o2::itsmft;

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
int AlpideCoder::encodeChip(PayLoadCont& buffer, const o2::itsmft::ChipPixelData& chipData,
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
    for (int ir = 0; ir < NRegions; ir++) {
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
    uint8_t mask = 0;
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
