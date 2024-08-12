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

/// \file AlpideCoder.cxx
/// \brief Implementation of the ALPIDE data decoding/encoding

#include "ITSMFTReconstruction/AlpideCoder.h"

using namespace o2::itsmft;

const NoiseMap* AlpideCoder::mNoisyPixels = nullptr;

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
      // For each region, we encode a REGION HEADER flag immediately
      // to ensure its uniqueness.
      buffer.addFast(makeRegion(ir));
      int nfoundInRegion = procRegion(buffer, ir);
      nfound += nfoundInRegion;
      // If the region was unpopulated, we remove REGION HEADER flag.
      if (!nfoundInRegion) {
        buffer.erase(1);
      }
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
  std::array<short, 2 * NRows> hits;
  int nHits = 0, nData = 0;
  //
  int nr = mFirstInRow.size();
  short col0 = ((reg * NDColInReg + dcol) << 1), col1 = col0 + 1; // 1st,2nd column of double column
  int prevRow = -1;
  for (int ir = 0; ir < nr; ir++) {
    int linkID = mFirstInRow[ir];
    if (linkID == -1) { // no pixels left on this row
      continue;
    }
    if (mPix2Encode[linkID].col > col1) { // all following hits will have higher columns
      break;
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
  if (nHits > 1) {
    std::sort(hits.begin(), hits.begin() + nHits);
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
