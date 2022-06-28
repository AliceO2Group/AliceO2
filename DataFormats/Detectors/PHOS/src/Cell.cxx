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

#include "DataFormatsPHOS/Cell.h"
#include <iostream>
#include <bitset>

using namespace o2::phos;

// split 40 bits as following:
// 14 bits: address, normal cells from 1 to NmaxCell=4*56*64-1792=14 336-1792=12544 will be
// TRU cells (3 136) addresses from 12545 to 15680
// 10 bits: time
// 15 bits: Energy
// 1 bit:    High/low gain

Cell::Cell(short absId, float energy, float time, ChannelType_t ctype)
{
  setAbsId(absId);
  setTime(time);
  setType(ctype);
  setEnergy(energy);
}

void Cell::setAbsId(short absId)
{
  // 14 bits available
  if (absId < kOffset) {
    absId = kOffset;
  }
  ULong_t t = (ULong_t)(absId - kOffset);

  ULong_t b = getLong() & 0xffffffc000; // 1111 1111 1111 1111 1111 1111 1100 0000 0000 0000
  mBits = b + t;
}
short Cell::getAbsId() const
{
  ULong_t t = getLong() & 0x3fff; // 14 bits
  short a = kOffset + (short)t;
  if (a <= kNmaxCell) { // readout cells
    return a;
  } else { // TRU cells
    return 0;
  }
}

short Cell::getTRUId() const
{
  ULong_t t = getLong() & 0x3fff; // 14 bits
  short a = kOffset + (short)t;
  return a;
}

void Cell::setTime(float time)
{
  // 13 bits available for time
  //  -Underflow
  //  -1500<t<-800 ns, 1.  ns binning =>  700 bin
  //   -800<t<-200 ns, 0.6 ns binning => 1000 bin
  //   -200<t< 200 ns, 0.2 ns binning => 2000 bin
  //    200<t< 800 ns, 0.6 ns binning => 1000 bin
  //    800<t<4290 ns, 1   ns binning => 3490 bin
  //  + overflow                total    8192

  ULong_t t = 0;
  // Convert time to long
  if (time < kTime0) {
    t = 0; // underflow
  } else {
    if (time < kTime1) {
      t = ULong_t(1.5 + (time - kTime0) / kTimeAccuracy1);
    } else {
      if (time < kTime2) {
        t = kTimeOffset1 + ULong_t(0.5 + (time - kTime1) / kTimeAccuracy2);
      } else {
        if (time < kTime3) {
          t = kTimeOffset2 + ULong_t(0.5 + (time - kTime2) / kTimeAccuracy3);
        } else {
          if (time < kTime4) {
            t = kTimeOffset3 + ULong_t(0.5 + (time - kTime3) / kTimeAccuracy4);
          } else {
            t = kTimeOffset4 + ULong_t(0.5 + (time - kTime4) / kTimeAccuracy5);
            if (t > 8191) {
              t = 8191; // overflow
            }
          }
        }
      }
    }
  }
  t <<= 14;
  ULong_t b = getLong() & 0xfff8003fff; // 1111 1111 1111 1000 0000 0000 0011 1111 1111 1111
  mBits = b + t;
}
float Cell::getTime() const
{
  ULong_t t = getLong();
  t >>= 14;
  t &= 0x1fff;
  // Convert back long to float
  if (t == 0) {
    return kTime0 - 1.e-9; // First bin- underflow
  }
  if (t < kTimeOffset1) {
    return float((t - 1) * kTimeAccuracy1) + kTime0; // First bin- underflow
  } else {
    if (t < kTimeOffset2) {
      return float((t - kTimeOffset1) * kTimeAccuracy2) + kTime1;
    } else {
      if (t < kTimeOffset3) {
        return float((t - kTimeOffset2) * kTimeAccuracy3) + kTime2;
      } else {
        if (t < kTimeOffset4) {
          return float((t - kTimeOffset3) * kTimeAccuracy4) + kTime3;
        } else {
          return float((t - kTimeOffset4) * kTimeAccuracy5) + kTime4;
        }
      }
    }
  }
}

void Cell::setEnergy(float amp)
{
  // 12 bits
  ULong_t a;
  if (getType() == HIGH_GAIN) {
    a = static_cast<ULong_t>(amp * 4);
  } else {
    a = static_cast<ULong_t>(amp);
  }
  a = a & 0xfff; // 12 bits

  a <<= 27;
  ULong_t b = getLong() & 0x8007ffffff; // 1000 0000 0000 0111 1111 1111 1111 1111 1111 1111
  mBits = b + a;
}

float Cell::getEnergy() const
{
  ULong_t a = getLong();
  a >>= 27;
  a &= 0xfff;
  if (getType() == HIGH_GAIN) {
    return float(0.25 * a);
  } else {
    return float(a);
  }
}

void Cell::setType(ChannelType_t ctype)
{
  switch (ctype) {
    case ChannelType_t::HIGH_GAIN:
    case ChannelType_t::TRU2x2:
      setHighGain();
      break;
    case ChannelType_t::LOW_GAIN:
    case ChannelType_t::TRU4x4:
      setLowGain();
      break;
    default:;
  };
}

ChannelType_t Cell::getType() const
{
  if (getTRU()) {
    if (getHighGain()) {
      return ChannelType_t::TRU2x2;
    } else {
      return ChannelType_t::TRU4x4;
    }
  } else {
    if (getHighGain()) {
      return ChannelType_t::HIGH_GAIN;
    } else {
      return ChannelType_t::LOW_GAIN;
    }
  }
}

void Cell::setLowGain()
{
  std::bitset<40> b(0x7fffffffff); // 0111111111111111111111111111111111111111
  mBits = (mBits & b);
}

Bool_t Cell::getLowGain() const
{
  ULong_t t = (getLong() >> 39);
  if (t) {
    return false;
  }
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
  if (t == 1) {
    return true;
  }
  return false;
}

Bool_t Cell::getTRU() const
{
  ULong_t t = getLong();
  t &= 0x3fff; // 14 bits
  int a = kOffset + (int)t;
  return (a > kNmaxCell); // TRU addresses
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
