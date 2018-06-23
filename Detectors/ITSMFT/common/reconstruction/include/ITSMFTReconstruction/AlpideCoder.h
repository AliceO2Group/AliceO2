// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFT_ALPIDEDECODER_H
#define ALICEO2_ITSMFT_ALPIDEDECODER_H

#include <Rtypes.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <cstdint>
#include <TStopwatch.h>

/// \file AlpideCoder.h
/// \brief class for the ALPIDE data decoding/encoding
/// \author Ruben Shahoyan, ruben.shahoyan@cern.ch

#define _BIG_ENDIAN_FORMAT_ // store data in big endian format

//#define _DEBUG_ALPIDE_DECODER_ // uncomment for debug mode

namespace o2
{
namespace ITSMFT
{

class AlpideCoder
{

 public:
  struct HitsRecord { // single record for hits (i.e. DATASHORT or DATALONG)
    HitsRecord() = default;
    ~HitsRecord() = default;
    HitsRecord(UChar_t r, UChar_t dc, UShort_t adr, UChar_t hmap) : region(r), dcolumn(dc), address(adr), hitmap(hmap) {}
    UChar_t region = 0;   // region ID
    UChar_t dcolumn = 0;  // double column ID
    UShort_t address = 0; // address in double column
    UChar_t hitmap = 0;   // hitmap for extra hits
  };

  struct PixLink { // single pixel on the selected row, referring eventually to the next pixel on the same row
    PixLink(short r = 0, short c = 0, int next = -1) : row(r), col(c), nextInRow(next) {}
    short row = 0;
    short col = 0;
    int nextInRow = -1; // index of the next pixel (link) on the same row
  };
  //
  static constexpr int DefaultBufferSize = 10000000; /// size of the input data buffer

  static constexpr UInt_t ExpectModuleHeader = 0x1 << 0;
  static constexpr UInt_t ExpectModuleTrailer = 0x1 << 1;
  static constexpr UInt_t ExpectChipHeader = 0x1 << 2;
  static constexpr UInt_t ExpectChipTrailer = 0x1 << 3;
  static constexpr UInt_t ExpectChipEmpty = 0x1 << 4;
  static constexpr UInt_t ExpectRegion = 0x1 << 5;
  static constexpr UInt_t ExpectData = 0x1 << 6;
  static constexpr Int_t NRows = 512;
  static constexpr Int_t NCols = 1024;
  static constexpr Int_t NRegions = 32;
  static constexpr Int_t NDColInReg = NCols / NRegions / 2;
  static constexpr Int_t HitMapSize = 7;

  // masks for records components
  static constexpr UInt_t MaskEncoder = 0x3c00;                 // encoder (double column) ID takes 4 bit max (0:15)
  static constexpr UInt_t MaskPixID = 0x3ff;                    // pixel ID within encoder (double column) takes 10 bit max (0:1023)
  static constexpr UInt_t MaskDColID = MaskEncoder | MaskPixID; // mask for encoder + dcolumn combination
  static constexpr UInt_t MaskRegion = 0x1f;                    // region ID takes 5 bits max (0:31)
  static constexpr UInt_t MaskChipID = 0x0f;                    // chip id in module takes 4 bit max
  static constexpr UInt_t MaskROFlags = 0x0f;                   // RO flags in chip header takes 4 bit max
  static constexpr UInt_t MaskFrameStartData = 0xff;            // Frame start data takes 8 bit max
  static constexpr UInt_t MaskReserved = 0xff;                  // mask for reserved byte
  static constexpr UInt_t MaskHitMap = 0x7f;                    // mask for hit map: at most 7 hits in bits (0:6)
  static constexpr UInt_t MaskModuleID = 0xffff;                // THIS IS IMPROVISATION
  static constexpr UInt_t MaskCycleID = 0xffff;                 // THIS IS IMPROVISATION
  //
  // record sizes in bytes
  static constexpr UInt_t SizeRegion = 1;     // size of region marker in bytes
  static constexpr UInt_t SizeChipHeader = 2; // size of chip header in bytes
  static constexpr UInt_t SizeChipEmpty = 3;  // size of empty chip header in bytes
  static constexpr UInt_t SizeModuleData = 5; // size of the module headed/trailer
  //
  // flags for data records
  static constexpr UInt_t REGION = 0xc0;      // flag for region
  static constexpr UInt_t CHIPHEADER = 0xa0;  // flag for chip header
  static constexpr UInt_t CHIPTRAILER = 0xb0; // flag for chip trailer
  static constexpr UInt_t CHIPEMPTY = 0xe0;   // flag for empty chip
  static constexpr UInt_t DATALONG = 0x0000;  // flag for DATALONG
  static constexpr UInt_t DATASHORT = 0x4000; // flag for DATASHORT
  //
  static constexpr UInt_t MODULEHEADER = 0x01;  // THIS IS IMPROVISATION
  static constexpr UInt_t MODULETRAILER = 0x81; // THIS IS IMPROVISATION
  //
  static constexpr Int_t Error = -1;     // flag for decoding error
  static constexpr Int_t EOFFlag = -100; // flag for EOF in reading

  AlpideCoder();
  ~AlpideCoder();
  //
  // IO
  bool openOutput(const std::string filename);
  void openInput(const std::string filename);
  void closeIO();
  bool flushBuffer();
  int loadInBuffer(int chunk = DefaultBufferSize);
  const TStopwatch& getIOTimer() const { return (const TStopwatch&)mTimerIO; }
  //
  // reading raw data
  int readChipData(std::vector<AlpideCoder::HitsRecord>& hits,
                   UShort_t& chip,   // updated on change, don't modify returned value
                   UShort_t& module, // updated on change, don't modify returned value
                   UShort_t& cycle); // updated on change, don't modify returned value

  //
  void print() const;
  void reset();
  //
  // methods to use for data encoding

  // Add empty record for the chip with chipID within its module for the ROFrame = rof
  void addEmptyChip(int chipInMod, int rof) { addToBuffer(makeChipEmpty(chipInMod, rof, 0), SizeChipEmpty); }

  // Add header for the new module modID in ROFrame = rof
  void addModuleHeader(int modID, int rof) { addToBuffer(makeModuleHeader(modID, rof), SizeModuleData); }

  // Add trailer for the module modID in ROFrame = rof
  void addModuleTrailer(int modID, int rof) { addToBuffer(makeModuleTrailer(modID, rof), SizeModuleData); }

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

  // encode hit map for the chip with index chipID within its module for the ROFrame = rof
  void finishChipEncoding(int chipInMod, int rof) { encodeChip(chipInMod, rof, 0); }

 private:
  ///< prepare chip header: 1010<chip id[3:0]><frame start data[7:0]>
  UShort_t makeChipHeader(short chipID, short framestartdata)
  {
    UShort_t v = CHIPHEADER | (MaskChipID & chipID);
    v = (v << 8) | (framestartdata & MaskFrameStartData);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeChipHeader: chip:%d framdata:%d -> 0x%x\n", chipID, framestartdata, v);
#endif
    return v;
  }

  ///< prepare chip trailer: 1011<readout flags[3:0]><reserved[7:0]>
  UShort_t makeChipTrailer(short roflags, short reserved = 0)
  {
    UShort_t v = CHIPTRAILER | (MaskROFlags & roflags);
    v = (v << 8) | (reserved & MaskReserved);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeChipTrailer: ROflags:%d framdata:%d -> 0x%x\n", roflags, reserved, v);
#endif
    return v;
  }

  ///< prepare chip empty marker: 1110<chip id[3:0]><frame start data[7:0] ><reserved[7:0]>
  UInt_t makeChipEmpty(short chipID, short framestartdata, short reserved = 0)
  {
    UInt_t v = CHIPEMPTY | (MaskChipID & chipID);
    v = (((v << 8) | (framestartdata & MaskFrameStartData)) << 8) | (reserved & MaskReserved);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeChipEmpty: chip:%d framdata:%d -> 0x%x\n", chipID, framestartdata, v);
#endif
    return v;
  }

  ///< packs the address of region
  UChar_t makeRegion(short reg)
  {
    UChar_t v = REGION | (reg & MaskRegion);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeRegion: region:%d -> 0x%x\n", reg, v);
#endif
    return v;
  }

  ///< packs the address for data short
  UShort_t makeDataShort(short encoder, short address)
  {
    UShort_t v = DATASHORT | (MaskEncoder & (encoder << 10)) | (address & MaskPixID);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeDataShort: DCol:%d address:%d -> 0x%x\n", encoder, address, v);
#endif
    return v;
  }

  // packs the address for data short
  UShort_t makeDataLong(short encoder, short address)
  {
    UShort_t v = DATALONG | (MaskEncoder & (encoder << 10)) | (address & MaskPixID);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeDataLong: DCol:%d address:%d -> 0x%x\n", encoder, address, v);
#endif
    return v;
  }

  //
  // THIS IS IMPROVISATION

  // prepare module trailer: 10000001<moduleID[31:16]><cycle[15:0]>
  ULong64_t makeModuleTrailer(UShort_t module, UShort_t cycle)
  {
    ULong64_t v = MODULETRAILER;
    v = (v << 32) | ((MaskModuleID & module) << 16) | (cycle & MaskCycleID);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeModuleTrailer: Module:%d Cycle:%d -> 0x%lx\n", module, cycle, v);
#endif
    return v;
  }

  ///< prepare module header: 00000001<moduleID[31:16]><cycle[15:0]>
  ULong64_t makeModuleHeader(UShort_t module, UShort_t cycle)
  {
    ULong64_t v = MODULEHEADER;
    v = (v << 32) | ((MaskModuleID & module) << 16) | (cycle & MaskCycleID);
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("makeModuleHeader: Module:%d Cycle:%d -> 0x%lx\n", module, cycle, v);
#endif
    return v;
  }

  // ENCODING: converting hitmap to raw data
  int procDoubleCol(short reg, short dcol);

  ///< process region (16 double columns)
  int procRegion(short reg)
  {
    addToBuffer(makeRegion(reg));
    int nfound = 0;
    for (int idc = 0; idc < NDColInReg; idc++) {
      nfound += procDoubleCol(reg, idc);
    }
    if (!nfound) {
      eraseInBuffer(SizeRegion);
    }
    return nfound;
  }

  int encodeChip(short chipInModule, short framestartdata, short roflags = 0);

  void addToBuffer(UShort_t v);
  void addToBuffer(UChar_t v);
  void addToBuffer(UInt_t v, int nbytes);
  void addToBuffer(ULong64_t v, int nbytes);
  //
  void expandBuffer(int add = 1000);

  ///< erase last nbytes of buffer
  void eraseInBuffer(int nbytes)
  {
    mWrBufferFill -= nbytes;
#ifdef _DEBUG_ALPIDE_DECODER_
    printf("eraseInBuffer: %d\n", nbytes);
#endif
  }

  void resetMap();
  //

  // DECODING: converting raw data to hitmap

  ///< error message on unexpected EOF
  int unexpectedEOF(const char* message) const
  {
    printf("Error: unexpected EOF on %s\n", message);
    return Error;
  }

  ///< read character value from buffer
  bool getFromBuffer(UChar_t& v)
  {
    if (mBufferPointer >= mBufferEnd && !loadInBuffer()) {
      return false; // upload or finish
    }
    v = *mBufferPointer++;
    return true;
  }

  ///< read short value from buffer
  bool getFromBuffer(UShort_t& v)
  {
    if (mBufferPointer >= mBufferEnd - (sizeof(short) - 1) && !loadInBuffer()) {
      return false; // upload or finish
    }
#ifdef _BIG_ENDIAN_FORMAT_
    v = (*mBufferPointer++) << 8;
    v |= (*mBufferPointer++);
#else
    v = (*mBufferPointer++);
    v |= (*mBufferPointer++) << 8;
#endif
    return true;
  }

  ///< read nbytes characters to int from buffer (no check for nbytes>4)
  bool getFromBuffer(UInt_t& v, int nbytes)
  {
    if (mBufferPointer >= mBufferEnd - (nbytes - 1) && !loadInBuffer()) {
      return false; // upload or finish
    }
    v = 0;
#ifdef _BIG_ENDIAN_FORMAT_
    for (int ib = nbytes; ib--;)
      v |= (UInt_t)((*mBufferPointer++) << (8 << ib));
#else
    for (int ib = 0; ib < nbytes; ib++)
      v |= (UInt_t)((*mBufferPointer++) << (8 << ib));
#endif
    return true;
  }

  ///< read nbytes characters to int from buffer (no check for nbytes>8)
  bool getFromBuffer(ULong64_t& v, int nbytes)
  {
    // read nbytes characters to int from buffer (no check for nbytes>8)
    if (mBufferPointer >= mBufferEnd - (nbytes - 1) && !loadInBuffer()) {
      return false; // upload or finish
    }
    v = 0;
#ifdef _BIG_ENDIAN_FORMAT_
    for (int ib = nbytes; ib--;) {
      v |= (UInt_t)((*mBufferPointer++) << (8 << ib));
    }
#else
    for (int ib = 0; ib < nbytes; ib++) {
      v |= (UInt_t)((*mBufferPointer++) << (8 << ib));
    }
#endif
    return true;
  }

  ///< step back by 1 byte
  void stepBackInBuffer() { mBufferPointer--; }
  //
  short readNexttDCol(short* dest, short& dcol, short& region, short& chip, short& evid);
  //
  // =====================================================================
  TStopwatch mTimerIO;     //! timer for IO oparations
  FILE* mIOFile = nullptr; //! handler for output
  //
  // cluster map used for the ENCODING only
  std::vector<short> mFirstInRow;   //! entry of 1st pixel if each non-empty row in the mPix2Encode
  std::vector<PixLink> mPix2Encode; //! pool of links: fired pixel + index of the next one in the row
  //
  // members used for the DECODING only
  UChar_t* mWrBuffer = nullptr;      //! write buffer
  UChar_t* mBufferPointer = nullptr; //! current pointer in reading
  UChar_t* mBufferEnd = nullptr;     //! end of filled buffer + 1
  int mWrBufferSize = 0;             //! buffer size
  int mWrBufferFill = 0;             //! entries in the buffer
  //
  UInt_t mExpectInp = ExpectModuleHeader; //! type of input expected by reader
  //
  ClassDefNV(AlpideCoder, 1);
};

//_____________________________________
inline void AlpideCoder::addToBuffer(UChar_t v)
{
  // add character value to buffer
  if (mWrBufferFill >= mWrBufferSize - 1) {
    expandBuffer();
  }
//
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("addToBuffer:C 0x%x\n", v);
#endif
  //
  mWrBuffer[mWrBufferFill++] = v;
}

//_____________________________________
inline void AlpideCoder::addToBuffer(UShort_t v)
{
  // add short value to buffer
  if (mWrBufferFill >= mWrBufferSize - 2) {
    expandBuffer();
  }
//
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("addToBuffer:S 0x%x\n", v);
#endif
//
#ifdef _BIG_ENDIAN_FORMAT_
  mWrBuffer[mWrBufferFill++] = (v >> 8) & 0xff;
  mWrBuffer[mWrBufferFill++] = v & 0xff;
#else
  UShort_t* bfs = reinterpret_cast<UShort_t*>(mWrBuffer + mWrBufferFill);
  *bfs = v;
  mWrBufferFill += 2;
#endif
}

//_____________________________________
inline void AlpideCoder::addToBuffer(UInt_t v, int nbytes)
{
  // add 1-4 bytes to buffer
  if (mWrBufferFill >= mWrBufferSize - nbytes)
    expandBuffer();
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("addToBuffer:I%d 0x%x\n", nbytes, v);
#endif
//
#ifdef _BIG_ENDIAN_FORMAT_
  for (int ib = nbytes; ib--;) {
    mWrBuffer[mWrBufferFill + ib] = (UChar_t)(v & 0xff);
    v >>= 8;
  }
  mWrBufferFill += nbytes;
#else
  for (int ib = 0; ib < nbytes; ib++) {
    mWrBuffer[mWrBufferFill++] = (UChar_t)(v & 0xff);
    v >>= 8;
  }
#endif
  //
}

//_____________________________________
inline void AlpideCoder::addToBuffer(ULong64_t v, int nbytes)
{
  // add 1-8 bytes to buffer
  if (mWrBufferFill >= mWrBufferSize - nbytes) {
    expandBuffer();
  }
//
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("addToBuffer:LL%d 0x%lx\n", nbytes, v);
#endif
//
#ifdef _BIG_ENDIAN_FORMAT_
  for (int ib = nbytes; ib--;) {
    mWrBuffer[mWrBufferFill + ib] = (UChar_t)(v & 0xff);
    v >>= 8;
  }
  mWrBufferFill += nbytes;
#else
  for (int ib = 0; ib < nbytes; ib++) {
    mWrBuffer[mWrBufferFill++] = (UChar_t)(v & 0xff);
    v >>= 8;
  }
#endif
}
}
}

#endif
