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
#include <FairLogger.h>
#include <TClass.h>

using namespace o2::ITSMFT;

//_____________________________________
AlpideCoder::AlpideCoder()
{
  LOG(INFO) << "ALPIDE (De)Coder" << FairLogger::endl;
  LOG(INFO) << "ATTENTION: Currently ROFrameID is limited to UShort" << FairLogger::endl;
  mTimerIO.Stop();
}

//_____________________________________
AlpideCoder::~AlpideCoder()
{
  closeIO();
}

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
int AlpideCoder::encodeChip(short chipInModule, short framestartdata, short roflags)
{
  //print();
  // process chip data
  int nfound = 0;
  // chip header
  addToBuffer(makeChipHeader(chipInModule, framestartdata));
  for (int ir = NRegions; ir--;) {
    nfound += procRegion(ir);
  }
  if (nfound) {
    addToBuffer(makeChipTrailer(roflags, 0));
  } else {
    eraseInBuffer(SizeChipHeader);
    addToBuffer(makeChipEmpty(chipInModule, framestartdata, 0), SizeChipEmpty);
  }
  //
  resetMap();
  return nfound;
}

//_____________________________________
int AlpideCoder::procDoubleCol(short reg, short dcol)
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
  while ((ih < nHits)) {
    short addrE, addrW = hits[ih++]; // address of the reference hit
    UChar_t mask = 0, npx = 0;
    short addrLim = addrW + HitMapSize + 1; // 1+address of furthest hit can be put in the map
    while ((ih < nHits && (addrE = hits[ih]) < addrLim)) {
      mask |= 0x1 << (addrE - addrW - 1);
      ih++;
    }
    if (mask) { // flag DATALONG
      addToBuffer(makeDataLong(dcol, addrW));
      addToBuffer(mask);
    } else {
      addToBuffer(makeDataShort(dcol, addrW));
    }
    nData++;
  }
  //
  return nData;
}

//_____________________________________
void AlpideCoder::expandBuffer(int add)
{
#ifdef _DEBUG_PIX_CONV_
  printf("expandBuffer: %d -> %d\n", mWrBufferSize, mWrBufferSize + add);
#endif
  UChar_t* bfcopy = new UChar_t[(mWrBufferSize += add)];
  if (mWrBufferFill) {
    memcpy(bfcopy, mWrBuffer, mWrBufferFill * sizeof(UChar_t));
  }
  delete[] mWrBuffer;
  mWrBuffer = bfcopy;
}

//_____________________________________
bool AlpideCoder::openOutput(const std::string filename)
{
  // open output for raw data
  LOG(INFO) << "opening raw data output file " << filename << FairLogger::endl;
  mIOFile = fopen(filename.data(), "wb");
  return mIOFile != nullptr;
}

//_____________________________________
void AlpideCoder::openInput(const std::string filename)
{
  // open raw data input
  LOG(INFO) << "opening raw data input file: " << filename << FairLogger::endl;
  if (!(mIOFile = fopen(filename.data(), "rb"))) {
    LOG(FATAL) << "Failed to open input file" << filename << FairLogger::endl;
  }
  mWrBufferFill = 0;
  mBufferPointer = mWrBuffer;
  mBufferEnd = mWrBuffer;
  mExpectInp = ExpectModuleHeader;
  loadInBuffer();
}

//_____________________________________
void AlpideCoder::closeIO()
{
  // close io
  if (mIOFile) {
    flushBuffer();
    fclose(mIOFile);
  }
  mIOFile = nullptr;
  LOG(INFO) << Class()->GetName() << " Closed IO | total time spent in IO: " << FairLogger::endl;
  mTimerIO.Print();
}

//_____________________________________
bool AlpideCoder::flushBuffer()
{
  // flush current content of buffer
  if (!mWrBufferFill) {
    return false;
  }
  if (!mIOFile) {
    LOG(FATAL) << "Output handler is not created" << FairLogger::endl;
  }
  mTimerIO.Start(false);
  fwrite(mWrBuffer, sizeof(char), mWrBufferFill, mIOFile);
  mTimerIO.Stop();
  mWrBufferFill = 0; // reset the counter
  return true;
}

//_____________________________________
int AlpideCoder::loadInBuffer(int chunk)
{
  // upload next chunk of data to buffer
  int save = mBufferEnd - mBufferPointer;
  // move unprocessed part to the beginning of the buffer
  if (save > 0) {
    memmove(mWrBuffer, mBufferPointer, save);
  } else {
    save = 0;
  }
  //
  if (mWrBufferSize < (chunk + save)) {
    expandBuffer(chunk + save + 1000);
  }
  mBufferPointer = mWrBuffer;
#ifdef _DEBUG_PIX_CONV_
  printf("loadInBuffer: %d bytes placed with offset %d\n", chunk, save);
#endif
  mTimerIO.Start(false);
  int nc = (int)fread(mWrBuffer + save, sizeof(char), chunk, mIOFile);
  mTimerIO.Stop();
  mWrBufferFill = save + nc;
  mBufferEnd = mWrBuffer + mWrBufferFill;
  return mWrBufferFill;
}

//_____________________________________
void AlpideCoder::resetMap()
{
  // reset map of hits for current chip
  mFirstInRow.clear();
  mPix2Encode.clear();
}

//_____________________________________
int AlpideCoder::readChipData(std::vector<AlpideCoder::HitsRecord>& hits,
                              UShort_t& chip,   // updated on change, don't modify returned value
                              UShort_t& module, // updated on change, don't modify returned value
                              UShort_t& cycle)  // updated on change, don't modify returned value
{
  // read record for single non-empty chip, updating on change module and cycle.
  // return number of records filled (>0), EOFFlag or Error
  //
  hits.clear();
  UChar_t dataC = 0;
  UChar_t region = 0;
  UChar_t framestartdata = 0;
  UShort_t dataS = 0;
  UInt_t dataI = 0;
  //
  int nHitsRec = 0;
  //
  while (1) {
    //
    if (!getFromBuffer(dataC)) {
      return EOFFlag;
    }
    //
    if (mExpectInp & ExpectModuleHeader && dataC == MODULEHEADER) { // new module header
      if (!getFromBuffer(module) || !getFromBuffer(cycle)) {
        return unexpectedEOF("MODULE_HEADER");
      }
      mExpectInp = ExpectModuleTrailer | ExpectChipHeader | ExpectChipEmpty;
      continue;
    }
    //
    if (mExpectInp & ExpectModuleTrailer && dataC == MODULETRAILER) { // module trailer
      UShort_t moduleT, cycleT;
      if (!getFromBuffer(moduleT) || !getFromBuffer(cycleT)) {
        return unexpectedEOF("MODULE_HEADER");
      }
      if (moduleT != module || cycleT != cycle) {
        LOG(ERROR) << "Error: expected module trailer for module " << module << "/cycle " << cycle << ", got for module "
                   << moduleT << "/cycle " << cycleT << FairLogger::endl;
        return Error;
      }
      mExpectInp = ExpectModuleHeader;
      continue;
    }
    // ---------- chip info ?
    UChar_t dataCM = dataC & (~MaskChipID);
    //
    if ((mExpectInp & ExpectChipHeader) && dataCM == CHIPHEADER) { // chip header was expected
      chip = dataC & MaskChipID;
      if (!getFromBuffer(framestartdata)) {
        return unexpectedEOF("CHIP_HEADER");
      }
      mExpectInp = ExpectRegion; // now expect region info
      continue;
    }
    //
    if ((mExpectInp & ExpectChipEmpty) && dataCM == CHIPEMPTY) { // chip trailer was expected
      chip = dataC & MaskChipID;
      if (!getFromBuffer(framestartdata)) {
        return unexpectedEOF("CHIP_EMPTY:FrameStartData");
      }
      if (!getFromBuffer(dataC)) {
        return unexpectedEOF("CHIP_EMPTY:ReservedWord");
      }
      mExpectInp = ExpectModuleTrailer | ExpectChipHeader | ExpectChipEmpty;
      continue;
    }
    //
    if ((mExpectInp & ExpectChipTrailer) && dataCM == CHIPTRAILER) { // chip trailer was expected
      if (!getFromBuffer(framestartdata)) {
        return unexpectedEOF("CHIP_TRAILER:FrameStartData");
      }
      mExpectInp = ExpectModuleTrailer | ExpectChipHeader | ExpectChipEmpty;
      return hits.size();
    }
    // region info ?
    if ((mExpectInp & ExpectRegion) && (dataC & REGION) == REGION) { // chip header was seen, or hit data read
      region = dataC & MaskRegion;
      mExpectInp = ExpectData;
      continue;
    }
    // hit info ?
    if ((mExpectInp & ExpectData)) { // region header was seen, expect data
      stepBackInBuffer();            // need to reinterpred as short
      if (!getFromBuffer(dataS)) {
        return unexpectedEOF("CHIPDATA");
      }
      UShort_t dataSM = dataS & (~MaskDColID); // check hit data mask
      if (dataSM == DATASHORT) {               // single hit
        UChar_t dColID = (dataS & MaskEncoder) >> 10;
        UShort_t pixID = dataS & MaskPixID;
        hits.emplace_back(region, dColID, pixID, 0);
        mExpectInp = ExpectData | ExpectRegion | ExpectChipTrailer;
        continue;
      } else if (dataSM == DATALONG) { // multiple hits
        UChar_t dColID = (dataS & MaskEncoder) >> 10;
        UShort_t pixID = dataS & MaskPixID;
        UChar_t hitsPattern = 0;
        if (!getFromBuffer(hitsPattern)) {
          return unexpectedEOF("CHIP_DATA_LONG:Pattern");
        }
        hits.emplace_back(region, dColID, pixID, hitsPattern);
        mExpectInp = ExpectData | ExpectRegion | ExpectChipTrailer;
        continue;
      } else {
        LOG(ERROR) << "Expected DataShort or DataLong mask, got : " << dataSM << FairLogger::endl;
        return Error;
      }
    }
  }
}
