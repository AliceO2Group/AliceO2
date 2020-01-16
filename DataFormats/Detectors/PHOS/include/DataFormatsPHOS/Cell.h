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

constexpr int kNmaxCell = 12545;        //56*64*3.5 + 1
constexpr float kEnergyConv = 0.005;    //Energy digitization step
constexpr float kTimeAccuracy = 0.3e-9; //Time digitization step
constexpr float kTime0 = 150.e-9;       //-Minimal time to be digitized

enum ChannelType_t {
  HIGH_GAIN, ///< High gain channel
  LOW_GAIN,  ///< Low gain channel
  TRU        ///< TRU channel
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

 private:
  std::bitset<40> mBits;

  ClassDefNV(Cell, 1);
};

std::ostream& operator<<(std::ostream& stream, const Cell& c);
} // namespace phos
} // namespace o2

#endif
