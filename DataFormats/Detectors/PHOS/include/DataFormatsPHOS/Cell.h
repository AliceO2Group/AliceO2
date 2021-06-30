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

#ifndef ALICEO2_PHOS_CELL_H_
#define ALICEO2_PHOS_CELL_H_

#include <bitset>
#include "Rtypes.h"

// Structure:
// Bits 39: Cell type: 0=Low Gain, 1=High Gain
// Bits 27-38: 12 bit, Amplitude (HG, resolution 0.25 ADC count, LG resolution 1 ADC count, dynamic range 4096)
// Bits 14-26: 13 bit, Time (ns)
// Bits  0-13: 14 bit, Address: absID or TRU address

namespace o2
{
namespace phos
{

constexpr int kOffset = 1792;            // offset due to missing half of module 1: 56*32
constexpr int kNmaxCell = 14337;         //56*64*4 + 1 - kOffset
constexpr float kTimeAccuracy = 0.25e-9; //Time digitization step
constexpr float kTime0 = -500.e-9;       //Minimal time to be digitized with 13 bits -500,1548 ns

enum ChannelType_t {
  HIGH_GAIN, ///< High gain channel
  LOW_GAIN,  ///< Low gain channel
  TRU2x2,    ///< TRU channel, 2x2 trigger
  TRU4x4     ///< TRU channel, 4x4 trigger
};

class Cell
{
 public:
  Cell() = default;
  Cell(short absId, float energy, float time, ChannelType_t ctype);
  ~Cell() = default; // override

  void setAbsId(short absId);
  short getAbsId() const;

  //return pure TRUid (absId with subtracted readout channels offset)
  short getTRUId() const;

  void setTime(float time);
  float getTime() const;

  //make sure that type of Cell (HG/LG) set before filling energy: scale will be different!
  void setEnergy(float energy);
  float getEnergy() const;

  void setType(ChannelType_t ctype);
  ChannelType_t getType() const;

  void setLowGain();
  bool getLowGain() const;

  void setHighGain();
  bool getHighGain() const;

  bool getTRU() const;

  void setLong(ULong_t l);
  ULong_t getLong() const { return mBits.to_ulong(); }

  void PrintStream(std::ostream& stream) const;

  // raw access for CTF encoding
  uint16_t getPackedID() const { return getLong() & 0x3fff; }
  void setPackedID(uint16_t v) { mBits = (getLong() & 0xffffffc000) + (v & 0x3fff); }

  uint16_t getPackedTime() const { return (getLong() >> 14) & 0x1fff; }
  void setPackedTime(uint16_t v) { mBits = (getLong() & 0xfff8003fff) + (uint64_t(v & 0x1fff) << 14); }

  uint16_t getPackedEnergy() const { return (getLong() >> 27) & 0xfff; }
  void setPackedEnergy(uint16_t v) { mBits = (getLong() & 0x8007ffffff) + (uint64_t(v & 0xfff) << 27); }

  uint8_t getPackedCellStatus() const { return mBits[39]; }
  void setPackedCellStatus(uint8_t v) { mBits[39] = v ? true : false; }

  void setPacked(uint16_t id, uint16_t t, uint16_t en, uint16_t status)
  {
    mBits = uint64_t(id & 0x3fff) + (uint64_t(t & 0x1fff) << 14) + (uint64_t(en & 0xfff) << 27) + (uint64_t(status & 0x1) << 39);
  }

 private:
  std::bitset<40> mBits;

  ClassDefNV(Cell, 1);
};

std::ostream& operator<<(std::ostream& stream, const Cell& c);
} // namespace phos
} // namespace o2

#endif
