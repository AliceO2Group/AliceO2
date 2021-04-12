// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
// Bits 24-38: Energy (input/output in GeV/c^2, resolution 1 ADC count)
// Bits 14-23: Time (ns)
// Bits  0-13: Address: absID or TRU address

namespace o2
{
namespace phos
{

constexpr int kOffset = 1792;           // offset due to missing half of module 1: 56*32
constexpr int kNmaxCell = 14337;        //56*64*4 + 1 - kOffset
constexpr float kEnergyConv = 0.005;    //Energy digitization step
constexpr float kTimeAccuracy = 0.3e-9; //Time digitization step
constexpr float kTime0 = 150.e-9;       //-Minimal time to be digitized

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

  void setTRUId(short truId);
  short getTRUId() const;

  void setTime(float time);
  float getTime() const;

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

  uint16_t getPackedTime() const { return (getLong() >> 14) & 0x3ff; }
  void setPackedTime(uint16_t v) { mBits = (getLong() & 0xffff003fff) + (uint64_t(v & 0x3ff) << 14); }

  uint16_t getPackedEnergy() const { return (getLong() >> 24) & 0x7fff; }
  void setPackedEnergy(uint16_t v) { mBits = (getLong() & 0x8000ffffff) + (uint64_t(v & 0x7fff) << 24); }

  uint8_t getPackedCellStatus() const { return mBits[39]; }
  void setPackedCellStatus(uint8_t v) { mBits[39] = v ? true : false; }

  void setPacked(uint16_t id, uint16_t t, uint16_t en, uint16_t status)
  {
    mBits = uint64_t(id & 0x3fff) + (uint64_t(t & 0x3ff) << 14) + (uint64_t(en & 0x7fff) << 24) + (uint64_t(status & 0x1) << 39);
  }

 private:
  std::bitset<40> mBits;

  ClassDefNV(Cell, 1);
};

std::ostream& operator<<(std::ostream& stream, const Cell& c);
} // namespace phos
} // namespace o2

#endif
