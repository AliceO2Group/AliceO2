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

#ifndef ALICEO2_CPV_RAWFORMATS_H
#define ALICEO2_CPV_RAWFORMATS_H

#include <cstdint>
#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{

namespace cpv
{

// Pack information into 24 bit words
union PadWord {
  uint32_t mDataWord;
  struct {
    uint32_t charge : 12; ///< Bits  0 - 11 : charge
    uint32_t address : 6; ///< Bits 12 - 17 : address  (0..47)
    uint32_t gas : 3;     ///< Bits 18 - 20 : gasiplex (1..10)
    uint32_t dil : 2;     ///< Bits 21 - 22 : dilogic  (1..24)
    uint32_t zero : 1;    ///< Bits 23 - 23 : control bit (0 -> pad word, 1 -> FEE control word)
  };
  char mBytes[4];
};

class CpvWord
{
 public:
  CpvWord() = default;
  CpvWord(std::vector<char>::const_iterator b, std::vector<char>::const_iterator e)
  { // Reading
    // resposibility of coller to esure that
    // array will not end while reading
    for (int i = 0; i < 10 && b != e; i++, b++) {
      mBytes[i] = *b;
    }
  }
  ~CpvWord() = default;
  bool isOK() const
  {
    return (mBytes[9] < static_cast<unsigned char>(24));
  }
  short ccId() const { return short(mBytes[9]); }
  uint32_t cpvPadWord(int i) const
  {
    PadWord p = {0};
    p.mBytes[0] = mBytes[3 * i];
    p.mBytes[1] = mBytes[3 * i + 1];
    p.mBytes[2] = mBytes[3 * i + 2];
    return p.mDataWord;
  }

 public:
  unsigned char mBytes[10] = {0};
};

class CpvHeader
{
 public:
  CpvHeader() = default;
  CpvHeader(std::vector<char>::const_iterator b, std::vector<char>::const_iterator e)
  {                                               // reading header from file
    for (int i = 0; i < 10 && b != e; i++, b++) { // read up to 10 mBytes
      mBytes[i] = *b;
    }
  }
  CpvHeader(InteractionRecord orbitBC, bool isNoDataExpected, bool isDataContinued)
  { // writing header
    // header is 128-bit word.
    // |127-120|119-112|111-104|103-96|95-88 |87-80 |79-72|71-64|63-56|55-48|47-40|39-32|31-24|23-16|15-8 |7-0  |
    // byte15  byte14  byte13  byte12 byte11 byte10 byte9 byte8 byte7 byte6 byte5 byte4 byte3 byte2 byte1 byte0
    // byte = |76543210|
    mBytes[0] = (0x010 & 0x0ff);                                               // bits 11 - 0 trigger id (0x010 = physics trigger)
    mBytes[1] = ((0x010 & 0xf00) >> 8)                                         // bits 11 - 0 trigger id (0x010 = physics trigger)
                + 0b00100000 * isNoDataExpected + 0b0100000 * isDataContinued; // bit 13 (no data for this trigger) + bit 14 (payload continues from previous page)
    mBytes[2] = (orbitBC.bc & 0x00ff);                                         // bits 27 - 16 bunch crossing
    mBytes[3] = (orbitBC.bc & 0x0f00) >> 8;                                    // bits 27 - 16 bunch crossing
    mBytes[4] = (orbitBC.orbit & 0x000000ff);                                  // bits 63 - 32 orbit
    mBytes[5] = (orbitBC.orbit & 0x0000ff00) >> 8;                             // bits 63 - 32 orbit
    mBytes[6] = (orbitBC.orbit & 0x00ff0000) >> 16;                            // bits 63 - 32 orbit
    mBytes[7] = (orbitBC.orbit & 0xff000000) >> 24;                            // bits 63 - 32 orbit
    mBytes[8] = 0x00;                                                          // bits 64-71 reserved
    mBytes[9] = 0xe0;                                                          // word ID of cpv header (bits 79 - 72)
  }
  ~CpvHeader() = default;
  bool isOK() const { return (mBytes[9] == 0xe0); }
  bool isNoDataExpected() const { return mBytes[1] & 0b00100000; }
  bool isDataContinued() const { return mBytes[1] & 0b0100000; }
  uint16_t bc() const { return static_cast<uint16_t>(mBytes[2]) + static_cast<uint16_t>((mBytes[3] & 0x0f) << 8); }
  uint32_t orbit() const { return mBytes[4] + (mBytes[5] << 8) + (mBytes[6] << 16) + (mBytes[7] << 24); }

 public:
  unsigned char mBytes[10] = {0}; // 0 - 79 bits (10 bytes)
};

class CpvTrailer
{
 public:
  CpvTrailer() = default;
  CpvTrailer(std::vector<char>::const_iterator b, std::vector<char>::const_iterator e)
  {                                               // reading
    for (int i = 0; i < 10 && b != e; i++, b++) { // read up to 10 mBytes
      mBytes[i] = *b;
    }
  }
  CpvTrailer(unsigned short wordCounter, uint16_t bunchCrossing, bool isAllDataSent)
  {                                               // writing
    mBytes[0] = bunchCrossing & 0x00ff;           // bits 11 - 0 bunch crossing
    mBytes[1] = ((bunchCrossing & 0x0f00) >> 8)   // bits 11 - 0 bunch crossing
                + ((wordCounter & 0x0f) << 4);    // bits 20 - 12 wordCounter
    mBytes[2] = (wordCounter & 0b111110000) >> 4; // bits 20 - 12 wordCounter
    for (int i = 3; i < 8; i++) {
      mBytes[i] = 0; // bits 70 - 21 reserved
    }
    mBytes[8] = isAllDataSent * 0b10000000; // bit 71 all data is sent for current trigger
    mBytes[9] = char(0xf0);                 // word ID of cpv trailer
  }
  ~CpvTrailer() = default;
  bool isOK() const { return (mBytes[9] == 0xf0); }
  uint16_t wordCounter() const { return (mBytes[1] >> 4) + ((mBytes[2] & 0b00011111) << 4); }
  bool isAllDataSent() const { return (mBytes[8] & 0b10000000); }
  uint16_t bc() const { return mBytes[0] + ((mBytes[1] & 0x0f) << 8); }

 public:
  unsigned char mBytes[10] = {0};
};

} // namespace cpv

} // namespace o2

#endif
