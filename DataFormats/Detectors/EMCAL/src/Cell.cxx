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

#include "DataFormatsEMCAL/Constants.h"
#include "DataFormatsEMCAL/Cell.h"
#include <iostream>
#include <bitset>
#include <cmath>

using namespace o2::emcal;

const float TIME_SHIFT = 600.,
            TIME_RANGE = 1500.,
            TIME_RESOLUTION = TIME_RANGE / 2047.,
            ENERGY_TRUNCATION = 250.,
            ENERGY_RESOLUTION = ENERGY_TRUNCATION / 16383.;

Cell::Cell(short tower, float energy, float timestamp, ChannelType_t ctype) : mTowerID(tower), mEnergy(energy), mTimestamp(timestamp), mChannelType(ctype)
{
}

uint16_t Cell::getTowerIDEncoded() const
{
  return mTowerID;
}

uint16_t Cell::getTimeStampEncoded() const
{
  // truncate:
  auto timestamp = mTimestamp;
  const float TIME_MIN = -1. * TIME_SHIFT,
              TIME_MAX = TIME_RANGE - TIME_SHIFT;
  if (timestamp < TIME_MIN) {
    timestamp = TIME_MIN;
  } else if (timestamp > TIME_MAX) {
    timestamp = TIME_MAX;
  }
  return static_cast<uint16_t>(std::round((timestamp + TIME_SHIFT) / TIME_RESOLUTION));
}

uint16_t Cell::getEnergyEncoded() const
{
  double truncatedEnergy = mEnergy;
  if (truncatedEnergy < 0.) {
    truncatedEnergy = 0.;
  } else if (truncatedEnergy > ENERGY_TRUNCATION) {
    truncatedEnergy = ENERGY_TRUNCATION;
  }
  return static_cast<int16_t>(std::round(truncatedEnergy / ENERGY_RESOLUTION));
}

uint16_t Cell::getCellTypeEncoded() const
{
  return static_cast<uint16_t>(mChannelType);
}

void Cell::setEnergyEncoded(uint16_t energyBits)
{
  mEnergy = energyBits * ENERGY_RESOLUTION;
}

void Cell::setTimestampEncoded(uint16_t timestampBits)
{
  mTimestamp = (timestampBits * TIME_RESOLUTION) - TIME_SHIFT;
}

void Cell::setTowerIDEncoded(uint16_t towerIDBits)
{
  mTowerID = towerIDBits;
}

void Cell::setChannelTypeEncoded(uint16_t channelTypeBits)
{
  mChannelType = static_cast<ChannelType_t>(channelTypeBits);
}

void Cell::truncate()
{
  setEnergyEncoded(getEnergyEncoded());
  setTimestampEncoded(getTimeStampEncoded());
}

void Cell::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell: Type " << getType() << ", Energy " << getEnergy() << ", Time " << getTimeStamp() << ", Tower " << getTower();
}

std::ostream& operator<<(std::ostream& stream, const Cell& c)
{
  c.PrintStream(stream);
  return stream;
}
