// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsPHOS/Cell.h"
#include <iostream>
#include <bitset>

using namespace o2::phos;

// split 40 bits as following:
// 14 bits: address, normal cells. starting from NmaxCell=3.5*56*64+1=12 544 will be TRU cells (3 136 addresses)
// 10 bits: time
// 15 bits: Energy
// 1 bit:    High/low gain

Cell::Cell(short absId, float energy, float time, ChannelType_t ctype)
{
  if (ctype == ChannelType_t::TRU) {
    setTRUId(absId);
  } else {
    setAbsId(absId);
  }
  setTime(time);
  setEnergy(energy);
  setType(ctype);
}

void Cell::setAbsId(short absId)
{
  //14 bits available
  if (absId > kNmaxCell || absId < 0)
    absId = kNmaxCell;
  ULong_t t = (ULong_t)absId;

  ULong_t b = getLong() & 0xffffffc000; // 1111 1111 1111 1111 1111 1111 1100 0000 0000 0000
  mBits = b + t;
}
void Cell::setTRUId(short absId)
{
  //14 bits available
  absId += kNmaxCell + 1;
  //  if (absId > kNmaxCell || absId < 0)
  //    absId = kNmaxCell;
  ULong_t t = (ULong_t)absId;

  ULong_t b = getLong() & 0xffffffc000; // 1111 1111 1111 1111 1111 1111 1100 0000 0000 0000
  mBits = b + t;
}

short Cell::getAbsId() const
{
  ULong_t t = getLong() & 0x3fff; //14 bits
  short a = (short)t;
  if (a <= kNmaxCell)
    return a;
  else
    return 0;
}

short Cell::getTRUId() const
{
  ULong_t t = getLong() & 0x3fff; //14 bits
  short a = (short)t;
  if (a > kNmaxCell)
    return a - kNmaxCell - 1;
  else
    return 0;
}

void Cell::setTime(float time)
{
  //10 bits available for time
  ULong_t t = 0;
  //Convert time to long
  t = ULong_t((time + kTime0) / kTimeAccuracy);
  if (t > 0x3ff) {
    t = 0x3ff;
  } else {
    if (t < 0) {
      t = 0;
    }
  }

  t <<= 14;
  ULong_t b = getLong() & 0xffff003fff; // 1111 1111 1111 1111 0000 0000 0011 1111 1111 1111
  mBits = b + t;
}

float Cell::getTime() const
{
  ULong_t t = getLong();
  t >>= 14;
  t &= 0x3ff;
  //Convert back long to float

  return float(t * kTimeAccuracy) - kTime0;
}

void Cell::setEnergy(float energy)
{
  //15 bits
  ULong_t a = static_cast<ULong_t>(energy / kEnergyConv);
  a = a & 0x7FFF; //15 bits

  a <<= 24;
  ULong_t b = getLong() & 0x8000ffffff; // 1000 0000 0000 0000 1111 1111 1111 1111 1111 1111
  mBits = b + a;
}

float Cell::getEnergy() const
{
  ULong_t a = getLong();
  a >>= 24;
  a &= 0x7FFF;
  return float(a * kEnergyConv);
}

void Cell::setType(ChannelType_t ctype)
{
  switch (ctype) {
    case ChannelType_t::HIGH_GAIN:
      setHighGain();
      break;
    case ChannelType_t::LOW_GAIN:
      setLowGain();
      break;
    default:;
  };
}

ChannelType_t Cell::getType() const
{
  if (getHighGain())
    return ChannelType_t::HIGH_GAIN;
  else if (getTRU())
    return ChannelType_t::TRU;
  return ChannelType_t::LOW_GAIN;
}

void Cell::setLowGain()
{
  std::bitset<40> b(0x7fffffffff); // 0111111111111111111111111111111111111111
  mBits = (mBits & b);
}

Bool_t Cell::getLowGain() const
{
  ULong_t t = (getLong() >> 39);
  if (t)
    return false;
  return true;
}

void Cell::setHighGain()
{
  ULong_t b = getLong() & 0x7fffffffff; // 0111111111111111111111111111111111111111
  mBits = b + 0x8000000000;             // 1000000000000000000000000000000000000000
}

Bool_t Cell::getHighGain() const
{
  ULong_t t = (getLong() >> 39);
  if (t == 1)
    return true;
  return false;
}

Bool_t Cell::getTRU() const
{
  ULong_t t = getLong();
  t &= 0x3fff; //14 bits
  int a = (int)t;
  return (a > kNmaxCell); //TRU addresses
}

void Cell::setLong(ULong_t l)
{
  std::bitset<40> b(l);
  mBits = b;
}

void Cell::PrintStream(std::ostream& stream) const
{
  stream << "PHOS Cell: Type " << getType() << ", Energy " << getEnergy() << ", Time " << getTime() << ", absId " << getAbsId() << ", Bits " << mBits;
}

std::ostream& operator<<(std::ostream& stream, const Cell& c)
{
  c.PrintStream(stream);
  return stream;
}
