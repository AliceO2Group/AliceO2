// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALBase/Cell.h"
#include <iostream>
#include <bitset>

using namespace o2::emcal;

ClassImp(Cell)

  Cell::Cell(Double_t amplitude, Double_t time, Short_t tower)
{
  setAmplitudeToADC(amplitude);
  setTime(time);
  setTower(tower);
}

Cell& Cell::operator=(const Cell& c)
{
  mBits = c.mBits;
  return *this;
}

void Cell::setAmplitudeToADC(Double_t amplitude)
{
  ULong_t a = 0;
  if (amplitude > 0x3ff * (constants::EMCAL_ADCENERGY))
    a = 0x3ff;
  else if (amplitude < 0)
    a = 0;
  else
    a = (ULong_t)(amplitude / (constants::EMCAL_ADCENERGY));

  a <<= 25;
  std::bitset<40> b1(a);
  std::bitset<40> b2(0x1ffffff); // (2^25 - 1)
  mBits = b1 + (mBits & b2);
}

void Cell::setADC(Short_t adc)
{
  if (adc > 0x3ff)
    adc = 0x3ff;
  else if (adc < 0)
    adc = 0;
  ULong_t a = (Ulong_t)adc;

  a <<= 25;
  std::bitset<40> b1(a);
  std::bitset<40> b2(0x1ffffff); // (2^25 - 1)
  mBits = b1 + (mBits & b2);
}

Short_t Cell::getADC() const
{
  ULong_t a = getLong();
  a >>= 25;
  a &= 0x3ff;
  return ((Short_t)a);
}

void Cell::setTime(Double_t time)
{
  ULong_t t = 0;
  if (time > 0x1ff)
    t = 0x1ff;
  else if (time < 0)
    t = 0;
  else
    t = (ULong_t)time;

  t <<= 16;
  std::bitset<40> b1(t);
  std::bitset<40> b2(0x07fe00ffff); // 0000011111111110000000001111111111111111
  mBits = b1 + (mBits & b2);
}

Short_t Cell::getTime() const
{
  ULong_t t = getLong();
  t >>= 16;
  t &= 0x1ff;
  return ((Short_t)t);
}

void Cell::setTower(Short_t tower)
{
  if (tower > 0x7fff || tower < 0)
    tower = 0x7fff;
  ULong_t t = (ULong_t)tower;

  std::bitset<40> b1(t);
  std::bitset<40> b2(0x07ffff0000); // 0000011111111111111111110000000000000000
  mBits = b1 + (mBits & b2);
}

Short_t Cell::getTower() const
{
  ULong_t t = getLong();
  t &= 0x7fff;
  return ((Short_t)t);
}

void Cell::setLong(ULong_t l)
{
  bitset<40> b(l);
  mBits = b;
}

void Cell::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell: Tower " << getTower() << ", Time " << getTime() << ", ADC " << getAmplitude() << ", Bits " << mBits;
}

std::ostream& operator<<(std::ostream& stream, const Cell& c)
{
  c.PrintStream(stream);
  return stream;
}
