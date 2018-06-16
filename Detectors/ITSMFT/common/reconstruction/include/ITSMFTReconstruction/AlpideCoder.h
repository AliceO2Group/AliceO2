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
    HitsRecord(std::uint8_t r, std::uint8_t dc, std::uint16_t adr, std::uint8_t hmap) : region(r), dcolumn(dc), address(adr), hitmap(hmap) {}
    std::uint8_t region = 0;   // region ID
    std::uint8_t dcolumn = 0;  // double column ID
    std::uint16_t address = 0; // address in double column
    std::uint8_t hitmap = 0;   // hitmap for extra hits
  };

  struct PixLink { // single pixel on the selected row, referring eventually to the next pixel on the same row
    PixLink(short r = 0, short c = 0, int next = -1) : row(r), col(c), nextInRow(next) {}
    short row = 0;
    short col = 0;
    int nextInRow = -1; // index of the next pixel (link) on the same row
  };
  //

  static constexpr std::uint32_t ExpectModuleHeader = 0x1 << 0;
  static constexpr std::uint32_t ExpectModuleTrailer = 0x1 << 1;
  static constexpr std::uint32_t ExpectChipHeader = 0x1 << 2;
  static constexpr std::uint32_t ExpectChipTrailer = 0x1 << 3;
  static constexpr std::uint32_t ExpectChipEmpty = 0x1 << 4;
  static constexpr std::uint32_t ExpectRegion = 0x1 << 5;
  static constexpr std::uint32_t ExpectData = 0x1 << 6;
  static constexpr std::int32_t NRows = 512;
  static constexpr std::int32_t NCols = 1024;
  static constexpr std::int32_t NRegions = 32;
  static constexpr std::int32_t NDColInReg = NCols / NRegions / 2;
  static constexpr std::int32_t HitMapSize = 7;

  // masks for records components
  static constexpr std::uint32_t MaskEncoder = 0x3c00;                 // encoder (double column) ID takes 4 bit max (0:15)
  static constexpr std::uint32_t MaskPixID = 0x3ff;                    // pixel ID within encoder (double column) takes 10 bit max (0:1023)
  static constexpr std::uint32_t MaskDColID = MaskEncoder | MaskPixID; // mask for encoder + dcolumn combination
  static constexpr std::uint32_t MaskRegion = 0x1f;                    // region ID takes 5 bits max (0:31)
  static constexpr std::uint32_t MaskChipID = 0x0f;                    // chip id in module takes 4 bit max
  static constexpr std::uint32_t MaskROFlags = 0x0f;                   // RO flags in chip header takes 4 bit max
  static constexpr std::uint32_t MaskFrameStartData = 0xff;            // Frame start data takes 8 bit max
  static constexpr std::uint32_t MaskReserved = 0xff;                  // mask for reserved byte
  static constexpr std::uint32_t MaskHitMap = 0x7f;                    // mask for hit map: at most 7 hits in bits (0:6)
  static constexpr std::uint32_t MaskModuleID = 0xffff;                // THIS IS IMPROVISATION
  static constexpr std::uint32_t MaskCycleID = 0xffff;                 // THIS IS IMPROVISATION
  //
  // record sizes in bytes
  static constexpr std::uint32_t SizeRegion = 1;     // size of region marker in bytes
  static constexpr std::uint32_t SizeChipHeader = 2; // size of chip header in bytes
  static constexpr std::uint32_t SizeChipEmpty = 3;  // size of empty chip header in bytes
  static constexpr std::uint32_t SizeModuleData = 5; // size of the module headed/trailer
  //
  // flags for data records
  static constexpr std::uint32_t REGION = 0xc0;      // flag for region
  static constexpr std::uint32_t CHIPHEADER = 0xa0;  // flag for chip header
  static constexpr std::uint32_t CHIPTRAILER = 0xb0; // flag for chip trailer
  static constexpr std::uint32_t CHIPEMPTY = 0xe0;   // flag for empty chip
  static constexpr std::uint32_t DATALONG = 0x0000;  // flag for DATALONG
  static constexpr std::uint32_t DATASHORT = 0x4000; // flag for DATASHORT
  //
  static constexpr std::uint32_t MODULEHEADER = 0x01;  // THIS IS IMPROVISATION
  static constexpr std::uint32_t MODULETRAILER = 0x81; // THIS IS IMPROVISATION
  //
  static constexpr std::int32_t Error = -1;     // flag for decoding error
  static constexpr std::int32_t EOFFlag = -100; // flag for EOF in reading

  AlpideCoder();
  ~AlpideCoder();
  //
  // IO
  bool openOutput(const std::string filename);
  void openInput(const std::string filename);
  void closeIO();
  bool flushBuffer();
  int loadInBuffer(int chunk = 1000000);
  //
  // reading raw data
  int readChipData(std::vector<AlpideCoder::HitsRecord>& hits,
                   std::uint16_t& chip,   // updated on change, don't modify returned value
                   std::uint16_t& module, // updated on change, don't modify returned value
                   std::uint16_t& cycle); // updated on change, don't modify returned value

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

  // add single pixel to current chip hit map
  void addPixel(short row, short col);

  // encode hit map for the chip with index chipID within its module for the ROFrame = rof
  void finishChipEncoding(int chipInMod, int rof) { encodeChip(chipInMod, rof, 0); }

 private:
  std::uint16_t makeChipHeader(short chipId, short framestartdata);
  std::uint16_t makeChipTrailer(short roflags, short reserved = 0);
  std::uint32_t makeChipEmpty(short chipId, short framestartdata, short reserved = 0);
  //
  std::uint8_t makeRegion(short reg);
  std::uint16_t makeDataShort(short encoder, short address);
  std::uint16_t makeDataLong(short encoder, short address);
  //
  // THIS IS IMPROVISATION
  std::uint64_t makeModuleHeader(std::uint16_t module, std::uint16_t cycle);
  std::uint64_t makeModuleTrailer(std::uint16_t module, std::uint16_t cycle);
  //

  // ENCODING: converting hitmap to raw data
  int procDoubleCol(short reg, short dcol);
  int procRegion(short reg);
  int encodeChip(short chipInModule, short framestartdata, short roflags = 0);

  void addToBuffer(std::uint16_t v);
  void addToBuffer(std::uint8_t v);
  void addToBuffer(std::uint32_t v, int nbytes);
  void addToBuffer(std::uint64_t v, int nbytes);
  //
  void expandBuffer(int add = 1000);
  void eraseInBuffer(int nbytes);
  void resetMap();
  //

  // DECODING: converting raw data to hitmap
  int unexpectedEOF(const char* message) const;
  bool getFromBuffer(std::uint8_t& v);
  bool getFromBuffer(std::uint16_t& v);
  bool getFromBuffer(std::uint32_t& v, int nbytes);
  bool getFromBuffer(std::uint64_t& v, int nbytes);
  void stepBackInBuffer();
  //
  short readNexttDCol(short* dest, short& dcol, short& region, short& chip, short& evid);
  //
  FILE* mIOFile = nullptr; //! handler for output
  //
  // cluster map used for the ENCODING only
  std::vector<short> mFirstInRow;   //! entry of 1st pixel if each non-empty row in the mPix2Encode
  std::vector<PixLink> mPix2Encode; //! pool of links: fired pixel + index of the next one in the row
  //
  // members used for the DECODING only
  std::uint8_t* mWrBuffer = nullptr;      //! write buffer
  std::uint8_t* mBufferPointer = nullptr; //! current pointer in reading
  std::uint8_t* mBufferEnd = nullptr;     //! end of filled buffer + 1
  int mWrBufferSize = 0;                  //! buffer size
  int mWrBufferFill = 0;                  //! entries in the buffer
  //
  std::uint32_t mExpectInp = ExpectModuleHeader; //! type of input expected by reader
  //
  ClassDefNV(AlpideCoder, 1);
};

//_____________________________________
inline std::uint64_t AlpideCoder::makeModuleHeader(std::uint16_t module, std::uint16_t cycle)
{
  // prepare module header: 00000001<moduleID[31:16]><cycle[15:0]>
  std::uint64_t v = MODULEHEADER;
  v = (v << 32) | ((MaskModuleID & module) << 16) | (cycle & MaskCycleID);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeModuleHeader: Module:%d Cycle:%d -> 0x%lx\n", module, cycle, v);
#endif
  return v;
}

//_____________________________________
inline std::uint64_t AlpideCoder::makeModuleTrailer(std::uint16_t module, std::uint16_t cycle)
{
  // prepare module trailer: 10000001<moduleID[31:16]><cycle[15:0]>
  std::uint64_t v = MODULETRAILER;
  v = (v << 32) | ((MaskModuleID & module) << 16) | (cycle & MaskCycleID);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeModuleTrailer: Module:%d Cycle:%d -> 0x%lx\n", module, cycle, v);
#endif
  return v;
}

//_____________________________________
inline std::uint16_t AlpideCoder::makeChipHeader(short chipID, short framestartdata)
{
  // prepare chip header: 1010<chip id[3:0]><frame start data[7:0]>
  std::uint16_t v = CHIPHEADER | (MaskChipID & chipID);
  v = (v << 8) | (framestartdata & MaskFrameStartData);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeChipHeader: chip:%d framdata:%d -> 0x%x\n", chipID, framestartdata, v);
#endif
  return v;
}

//_____________________________________
inline std::uint16_t AlpideCoder::makeChipTrailer(short roflags, short reserved)
{
  // prepare chip trailer: 1011<readout flags[3:0]><reserved[7:0]>
  std::uint16_t v = CHIPTRAILER | (MaskROFlags & roflags);
  v = (v << 8) | (reserved & MaskReserved);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeChipTrailer: ROflags:%d framdata:%d -> 0x%x\n", roflags, reserved, v);
#endif
  return v;
}

//_____________________________________
inline unsigned AlpideCoder::makeChipEmpty(short chipID, short framestartdata, short reserved)
{
  // prepare chip empty marker: 1110<chip id[3:0]><frame start data[7:0] ><reserved[7:0]>
  std::uint32_t v = CHIPEMPTY | (MaskChipID & chipID);
  v = (((v << 8) | (framestartdata & MaskFrameStartData)) << 8) | (reserved & MaskReserved);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeChipEmpty: chip:%d framdata:%d -> 0x%x\n", chipID, framestartdata, v);
#endif
  return v;
}

//_____________________________________
inline std::uint8_t AlpideCoder::makeRegion(short reg)
{
  // packs the address of region
  std::uint8_t v = REGION | (reg & MaskRegion);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeRegion: region:%d -> 0x%x\n", reg, v);
#endif
  return v;
}

//_____________________________________
inline std::uint16_t AlpideCoder::makeDataShort(short encoder, short address)
{
  // packs the address for data short
  std::uint16_t v = DATASHORT | (MaskEncoder & (encoder << 10)) | (address & MaskPixID);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeDataShort: DCol:%d address:%d -> 0x%x\n", encoder, address, v);
#endif
  return v;
}

//_____________________________________
inline std::uint16_t AlpideCoder::makeDataLong(short encoder, short address)
{
  // packs the address for data short
  std::uint16_t v = DATALONG | (MaskEncoder & (encoder << 10)) | (address & MaskPixID);
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("makeDataLong: DCol:%d address:%d -> 0x%x\n", encoder, address, v);
#endif
  return v;
}

//_____________________________________
inline void AlpideCoder::addToBuffer(std::uint8_t v)
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
inline void AlpideCoder::addToBuffer(std::uint16_t v)
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
  std::uint16_t* bfs = reinterpret_cast<std::uint16_t*>(mWrBuffer + mWrBufferFill);
  *bfs = v;
  mWrBufferFill += 2;
#endif
}

//_____________________________________
inline void AlpideCoder::addToBuffer(std::uint32_t v, int nbytes)
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
    mWrBuffer[mWrBufferFill + ib] = (std::uint8_t)(v & 0xff);
    v >>= 8;
  }
  mWrBufferFill += nbytes;
#else
  for (int ib = 0; ib < nbytes; ib++) {
    mWrBuffer[mWrBufferFill++] = (std::uint8_t)(v & 0xff);
    v >>= 8;
  }
#endif
  //
}

//_____________________________________
inline void AlpideCoder::addToBuffer(std::uint64_t v, int nbytes)
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
    mWrBuffer[mWrBufferFill + ib] = (std::uint8_t)(v & 0xff);
    v >>= 8;
  }
  mWrBufferFill += nbytes;
#else
  for (int ib = 0; ib < nbytes; ib++) {
    mWrBuffer[mWrBufferFill++] = (std::uint8_t)(v & 0xff);
    v >>= 8;
  }
#endif
}

//_____________________________________
inline void AlpideCoder::eraseInBuffer(int nbytes)
{
  // erase last nbytes of buffer
  mWrBufferFill -= nbytes;
#ifdef _DEBUG_ALPIDE_DECODER_
  printf("eraseInBuffer: %d\n", nbytes);
#endif
}

//_____________________________________
inline bool AlpideCoder::getFromBuffer(std::uint8_t& v)
{
  // read character value from buffer
  if (mBufferPointer >= mBufferEnd && !loadInBuffer()) {
    return false; // upload or finish
  }
  v = *mBufferPointer++;
  return true;
}

//_____________________________________
inline bool AlpideCoder::getFromBuffer(std::uint16_t& v)
{
  // read short value from buffer
  if (mBufferPointer >= mBufferEnd - (sizeof(short) - 1) && !loadInBuffer()) {
    return false; // upload or finish
  }
#ifdef _BIG_ENDIAN_FORMAT_
  v = (*mBufferPointer++) << 8;
  v |= (*mBufferPointer++);
#else
  v = (*mBufferPointer++);
  v |= (*mBufferPointer++) << 8;
//  v = * reinterpret_cast<std::uint16_t*>(mBufferPointer);
//  mBufferPointer+=sizeof(short);
#endif
  return true;
}

//_____________________________________
inline bool AlpideCoder::getFromBuffer(std::uint32_t& v, int nbytes)
{
  // read nbytes characters to int from buffer (no check for nbytes>4)
  if (mBufferPointer >= mBufferEnd - (nbytes - 1) && !loadInBuffer()) {
    return false; // upload or finish
  }
  v = 0;
#ifdef _BIG_ENDIAN_FORMAT_
  for (int ib = nbytes; ib--;)
    v |= (std::uint32_t)((*mBufferPointer++) << (8 << ib));
#else
  for (int ib = 0; ib < nbytes; ib++)
    v |= (std::uint32_t)((*mBufferPointer++) << (8 << ib));
#endif
  return true;
}

//_____________________________________
inline bool AlpideCoder::getFromBuffer(std::uint64_t& v, int nbytes)
{
  // read nbytes characters to int from buffer (no check for nbytes>8)
  if (mBufferPointer >= mBufferEnd - (nbytes - 1) && !loadInBuffer()) {
    return false; // upload or finish
  }
  v = 0;
#ifdef _BIG_ENDIAN_FORMAT_
  for (int ib = nbytes; ib--;) {
    v |= (std::uint32_t)((*mBufferPointer++) << (8 << ib));
  }
#else
  for (int ib = 0; ib < nbytes; ib++) {
    v |= (std::uint32_t)((*mBufferPointer++) << (8 << ib));
  }
#endif
  return true;
}

//_____________________________________
inline void AlpideCoder::stepBackInBuffer()
{
  // step back by 1 byte
  mBufferPointer--;
}
}
}

#endif
