// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

//Pack information into 24 bit words
union PadWord {
  uint32_t mDataWord;
  struct {
    uint32_t charge : 12; ///< Bits  0 - 11 : charge
    uint32_t address : 6; ///< Bits 12 - 17 : address  (0..47)
    uint32_t gas : 3;     ///< Bits 18 - 20 : gasiplex (1..10)
    uint32_t dil : 2;     ///< Bits 21 - 22 : dilogic  (1..24)
    uint32_t zero : 1;    ///< Bits 23 - 23 : zeroed
  };
  char bytes[4];
};

class cpvword
{
 public:
  cpvword() = default;
  cpvword(std::vector<char>::const_iterator b, std::vector<char>::const_iterator e)
  { //Reading
    //resposibility of coller to esure that
    //array will not end while reading
    for (int i = 0; i < 16 && b != e; i++, b++) {
      bytes[i] = *b;
    }
  }
  ~cpvword() = default;
  bool isOK() const
  {
    return (bytes[9] < static_cast<unsigned char>(24)) &&
           (bytes[15] == 0) && (bytes[14] == 0) && (bytes[13] == 0) && (bytes[12] == 0) && (bytes[11] == 0) && (bytes[10] == 0);
  }
  short ccId() const { return short(bytes[9]); }
  uint32_t cpvPadWord(int i) const
  {
    PadWord p = {0};
    p.bytes[0] = bytes[3 * i];
    p.bytes[1] = bytes[3 * i + 1];
    p.bytes[2] = bytes[3 * i + 2];
    return p.mDataWord;
  }

 public:
  unsigned char bytes[16] = {0};
};

class cpvheader
{
 public:
  cpvheader() = default;
  cpvheader(std::vector<char>::const_iterator b, std::vector<char>::const_iterator e)
  {                                               //reading header from file
    for (int i = 0; i < 16 && b != e; i++, b++) { //read up to 16 bytes
      bytes[i] = *b;
    }
  }
  cpvheader(InteractionRecord orbitBC, bool isNoDataExpected, bool isDataContinued)
  { //writing header
    //header is 128-bit word.
    //|127-120|119-112|111-104|103-96|95-88 |87-80 |79-72|71-64|63-56|55-48|47-40|39-32|31-24|23-16|15-8 |7-0  |
    // byte15  byte14  byte13  byte12 byte11 byte10 byte9 byte8 byte7 byte6 byte5 byte4 byte3 byte2 byte1 byte0
    //byte = |76543210|
    bytes[0] = (0x010 & 0x0ff);                                               //bits 11 - 0 trigger id (0x010 = physics trigger)
    bytes[1] = ((0x010 & 0xf00) >> 8)                                         //bits 11 - 0 trigger id (0x010 = physics trigger)
               + 0b00100000 * isNoDataExpected + 0b0100000 * isDataContinued; //bit 13 (no data for this trigger) + bit 14 (payload continues from previous page)
    bytes[2] = (orbitBC.bc & 0x00ff);                                         //bits 27 - 16 bunch crossing
    bytes[3] = (orbitBC.bc & 0x0f00) >> 8;                                    //bits 27 - 16 bunch crossing
    bytes[4] = (orbitBC.orbit & 0x000000ff);                                  //bits 63 - 32 orbit
    bytes[5] = (orbitBC.orbit & 0x0000ff00) >> 8;                             //bits 63 - 32 orbit
    bytes[6] = (orbitBC.orbit & 0x00ff0000) >> 16;                            //bits 63 - 32 orbit
    bytes[7] = (orbitBC.orbit & 0xff000000) >> 24;                            //bits 63 - 32 orbit
    bytes[8] = 0x00;                                                          //bits 64-71 reserved
    bytes[9] = 0xe0;                                                          //word ID of cpv header (bits 79 - 72)
    for (int i = 10; i < 16; i++) {
      bytes[i] = 0; //bits 127-80 must be zeros
    }
  }
  ~cpvheader() = default;
  bool isOK() const { return (bytes[9] == 0xe0) && (bytes[10] == 0) && (bytes[11] == 0) && (bytes[12] == 0) && (bytes[13] == 0) && (bytes[14] == 0) && (bytes[15] == 0); }
  bool isNoDataExpected() const { return bytes[1] & 0b00100000; }
  bool isDataContinued() const { return bytes[1] & 0b0100000; }
  uint16_t bc() const { return bytes[2] + ((bytes[3] & 0x0f) << 8); }
  uint32_t orbit() const { return bytes[4] + (bytes[5] << 8) + (bytes[6] << 16) + (bytes[7] << 24); }

 public:
  unsigned char bytes[16] = {0}; //0 - 127 bits (16 bytes)
};

class cpvtrailer
{
 public:
  cpvtrailer() = default;
  cpvtrailer(std::vector<char>::const_iterator b, std::vector<char>::const_iterator e)
  {                                               //reading
    for (int i = 0; i < 16 && b != e; i++, b++) { //read up to 16 bytes
      bytes[i] = *b;
    }
  }
  cpvtrailer(unsigned short wordCounter, uint16_t bunchCrossing, bool isAllDataSent)
  {                                              //writing
    bytes[0] = bunchCrossing & 0x00ff;           //bits 11 - 0 bunch crossing
    bytes[1] = ((bunchCrossing & 0x0f00) >> 8)   //bits 11 - 0 bunch crossing
               + ((wordCounter & 0x0f) << 4);    //bits 20 - 12 wordCounter
    bytes[2] = (wordCounter & 0b111110000) >> 4; //bits 20 - 12 wordCounter
    for (int i = 3; i < 8; i++) {
      bytes[i] = 0; //bits 70 - 21 reserved
    }
    bytes[8] = isAllDataSent * 0b10000000; //bit 71 all data is sent for current trigger
    bytes[9] = char(0xf0);                 //word ID of cpv trailer
    for (int i = 10; i < 16; i++) {
      bytes[i] = 0;
    }
  }
  ~cpvtrailer() = default;
  bool isOK() const { return (bytes[9] == 0xf0) && (bytes[10] == 0) && (bytes[11] == 0) && (bytes[12] == 0) && (bytes[13] == 0) && (bytes[14] == 0) && (bytes[15] == 0); }
  uint16_t wordCounter() const { return (bytes[1] >> 4) + ((bytes[2] & 0b00011111) << 4); }
  bool isAllDataSent() const { return (bytes[8] & 0b10000000); }
  uint16_t bc() const { return bytes[0] + ((bytes[1] & 0x0f) << 8); }

 public:
  unsigned char bytes[16] = {0};
};

} // namespace cpv

} // namespace o2

#endif
