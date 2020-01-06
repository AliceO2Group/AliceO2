// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_CELL_H_
#define ALICEO2_EMCAL_CELL_H_

#include <bitset>
#include "Rtypes.h"
#include "DataFormatsEMCAL/Constants.h"

// Structure:
// Bits 38-39: Cell type: 00=Low Gain, 01=High Gain, 10=LED mon, 11=TRU
// Bits 24-37: Energy (input/output in GeV/c^2, resolution 1/16 ADC count)
// Bits 15-23: Time (ns)
// Bits  0-14: Tower ID

namespace o2
{
namespace emcal
{

class Cell
{
 public:
  Cell() = default;
  Cell(Short_t tower, Double_t energy, Double_t time, ChannelType_t ctype = ChannelType_t::LOW_GAIN);
  ~Cell() = default; // override

  void setTower(Short_t tower);
  Short_t getTower() const;

  void setTimeStamp(Double_t time);
  Short_t getTimeStamp() const;

  void setEnergyBits(Short_t ebits);
  Short_t getEnergyBits() const;

  void setEnergy(Double_t energy);
  Double_t getEnergy() const;

  void setType(ChannelType_t ctype);
  ChannelType_t getType() const;

  void setLowGain();
  Bool_t getLowGain() const;

  void setHighGain();
  Bool_t getHighGain() const;

  void setLEDMon();
  Bool_t getLEDMon() const;

  void setTRU();
  Bool_t getTRU() const;

  void setLong(ULong_t l);
  ULong_t getLong() const { return mBits.to_ulong(); }

  void PrintStream(std::ostream& stream) const;

 private:
  std::bitset<40> mBits;

  ClassDefNV(Cell, 1);
};

std::ostream& operator<<(std::ostream& stream, const Cell& c);
} // namespace emcal
} // namespace o2

#endif
