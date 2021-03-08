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

class cpvtrailer
{
 public:
  cpvtrailer() = default;
  cpvtrailer(std::vector<char>::const_iterator b, std::vector<char>::const_iterator e)
  { //reading
    for (int i = 0; i < 16 && b != e; i++, b++) {
      bytes[i] = *b;
    }
  }
  cpvtrailer(unsigned short wordCounter)
  { //writing
    for (int i = 10; i < 16; i++) {
      bytes[i] = 0;
    }
    bytes[9] = char(0xf0);
    bytes[1] = (wordCounter & 0xff00) >> 8;
    bytes[0] = (wordCounter & 0x00ff);
  }
  ~cpvtrailer() = default;
  bool isOK() const { return (bytes[9] == 0xf0) && (bytes[10] == 0) && (bytes[11] == 0) && (bytes[12] == 0) && (bytes[13] == 0) && (bytes[14] == 0) && (bytes[15] == 0); }
  short status() const { return short(bytes[8]); }
  unsigned short wordCounter() const { return bytes[0] + (bytes[1] << 8); }

 public:
  unsigned char bytes[16] = {0};
};

} // namespace cpv

} // namespace o2

#endif
