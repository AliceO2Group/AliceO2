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

// Structure:
// Bits 25-34: ADC
// Bits 16-24: Time (ns)
// Bits  0-16: Tower ID

namespace o2
{
namespace emcal
{
class Cell
{
 public:
  Cell() = default;
  Cell(Double_t amplitude, Double_t time, Short_t tower);
  Cell(const Cell& c) { mBits = c.mBits; }
  Cell& operator=(const Cell& c);
  ~Cell() = default; // override

  void setAmplitudeToADC(Double_t amplitude);
  void setADC(Short_t adc);
  Short_t getADC() const;

  void setTime(Double_t time);
  Short_t getTime() const;

  void setTower(Short_t tower);
  Short_t getTower() const;

  void setLong(ULong_t l);
  ULong_t getLong() const { return mBits.to_ulong(); }

  void PrintStream(std::ostream& stream) const;

 private:
  std::bitset<40> mBits;
};

std::ostream& operator<<(std::ostream& stream, const Cell& c);
} // namespace emcal
} // namespace o2

#endif
