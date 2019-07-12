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

 Cell::Cell(Short_t tower, Double_t energy, Double_t time, CellType ctype)
{
  setTower(tower);
  setTimeStamp(time);
  setEnergy(energy);
  setType(ctype);
}

Cell& Cell::operator=(const Cell& c)
{
  mBits = c.mBits;
  return *this;
}

void Cell::setTower(Short_t tower)
{
  if (tower > 0x7fff || tower < 0)
    tower = 0x7fff;
  ULong_t t = (ULong_t)tower;

  std::bitset<40> b1(t);
  std::bitset<40> b2(0xffffff8000); // 1111111111111111111111111000000000000000
  mBits = b1 + (mBits & b2);
}

Short_t Cell::getTower() const
{
  ULong_t t = getLong();
  t &= 0x7fff;
  return ((Short_t)t);
}

void Cell::setTimeStamp(Double_t time)
{
  ULong_t t = 0;
  if (time > 0x1ff)
    t = 0x1ff;
  else if (time < 0)
    t = 0;
  else
    t = (ULong_t)time;

  t <<= 15;
  std::bitset<40> b1(t);
  std::bitset<40> b2(0xffff007fff); // 1111111111111111000000000111111111111111
  mBits = b1 + (mBits & b2);
}

Short_t Cell::getTimeStamp() const
{
  ULong_t t = getLong();
  t >>= 15;
  t &= 0x1ff;
  return ((Short_t)t);
}

void Cell::setEnergyBits(Short_t ebits)
{
  if (ebits > 0x3fff)
    ebits = 0x3fff;
  else if (ebits < 0)
    ebits = 0;
  ULong_t a = (Ulong_t)ebits;

  a <<= 24;
  std::bitset<40> b1(a);
  std::bitset<40> b2(0xc00ffffff); // 1100000000000000111111111111111111111111
  mBits = b1 + (mBits & b2);
}

Short_t Cell::getEnergyBits() const
{
  ULong_t a = getLong();
  a >>= 24;
  a &= 0x3fff;
  return ((Short_t)a);
}

void Cell::setEnergy(Double_t energy)
{
  ULong_t a = 0;
  if (energy > 0x3fff * (constants::EMCAL_ADCENERGY / 16.0))
    a = 0x3fff;
  else if (energy < 0)
    a = 0;
  else
    a = (ULong_t)((energy / (constants::EMCAL_ADCENERGY)*16.0) + 0.5);

  a <<= 24;
  std::bitset<40> b1(a);
  std::bitset<40> b2(0xc00ffffff); // 1100000000000000111111111111111111111111
  mBits = b1 + (mBits & b2);
}

Short_t Cell::getEnergy() const
{
  return getEnergyBits() * (constants::EMCAL_ADCENERGY) / 16.0;
}

void Cell::setType(CellType ctype)
{
  if (ctype == CellType::kHighGain)
    setHighGain();
  else if (ctype == CellType::kLEDMon)
    setLEDMon();
  else if (ctype == CellTypa::kTRU)
    setTRU();
  else
    setLowGain();
}

UInt_t Cell::getType()
{
  if (getHighGain())
    return CellType::kHighGain;
  else if (getLEDMon())
    return CellType::kLEDMon;
  else if (getTRU())
    return CellType::kTRU;
  return CellType::kLowGain;
}

void Cell::setLowGain()
{
  std::bitset<40> b(0x3fffffffff); // 0011111111111111111111111111111111111111
  mBits = (mBits & b);
}

Bool_t Cell::getLowGain()
{
  ULong_t t = (getLong() >> 38);
  if (t)
    return false;
  return true;
}

void Cell::setHighGain()
{
  std::bitset<40> b1(0x4000000000); // 0100000000000000000000000000000000000000
  std::bitset<40> b2(0x3fffffffff); // 0011111111111111111111111111111111111111
  mBits = b1 + (mBits & b2);
}

Bool_t Cell::getHighGain()
{
  ULong_t t = (getLong() >> 38);
  if (t == 1)
    return true;
  return false;
}

void Cell::setLEDMon()
{
  std::bitset<40> b1(0x8000000000); // 1000000000000000000000000000000000000000
  std::bitset<40> b2(0x3fffffffff); // 0011111111111111111111111111111111111111
  mBits = b1 + (mBits & b2);
}

Bool_t Cell::getLEDMon()
{
  ULong_t t = (getLong() >> 38);
  if (t == 2)
    return true;
  return false;
}

void Cell::setTRU()
{
  std::bitset<40> b1(0xc000000000); // 1100000000000000000000000000000000000000
  std::bitset<40> b2(0x3fffffffff); // 0011111111111111111111111111111111111111
  mBits = b1 + (mBits & b2);
}

Bool_t Cell::getTRU()
{
  ULong_t t = (getLong() >> 38);
  if (t == 3)
    return true;
  return false;
}

void Cell::setLong(ULong_t l)
{
  bitset<40> b(l);
  mBits = b;
}

void Cell::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell: Type " << getType() << ", Energy " << getEnergy() << ", Time " << getTimeStamp() << ", Tower " << getTower() << ", Bits " << mBits;
}

std::ostream& operator<<(std::ostream& stream, const Cell& c)
{
  c.PrintStream(stream);
  return stream;
}
