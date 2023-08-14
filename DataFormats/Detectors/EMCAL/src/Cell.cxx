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
            TIME_RESOLUTION = TIME_RANGE / 2047.;
namespace EnergyEncoding
{
namespace v0
{
const float
  ENERGY_TRUNCATION = 250.,
  ENERGY_RESOLUTION = ENERGY_TRUNCATION / 16383.;
}

namespace v1
{
const float
  ENERGY_BITS = static_cast<float>(0x3FFF),
  HGLGTRANSITION = o2::emcal::constants::EMCAL_HGLGTRANSITION * o2::emcal::constants::EMCAL_ADCENERGY,
  ENERGY_TRUNCATION = 250.,
  ENERGY_RESOLUTION_LG = (ENERGY_TRUNCATION - HGLGTRANSITION) / ENERGY_BITS,
  ENERGY_RESOLUTION_HG = HGLGTRANSITION / ENERGY_BITS,
  ENERGY_RESOLUTION_TRU = ENERGY_TRUNCATION / ENERGY_BITS,
  ENERGY_RESOLUTION_LEDMON = ENERGY_TRUNCATION / ENERGY_BITS;
}
} // namespace EnergyEncoding

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
  return encodeEnergyV0(mEnergy);
}

uint16_t Cell::getCellTypeEncoded() const
{
  return static_cast<uint16_t>(mChannelType);
}

void Cell::setEnergyEncoded(uint16_t energyBits)
{
  mEnergy = decodeEnergyV0(energyBits);
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

uint16_t Cell::encodeEnergyV0(float energy)
{
  auto truncatedEnergy = energy;
  if (truncatedEnergy < 0.) {
    truncatedEnergy = 0.;
  } else if (truncatedEnergy > EnergyEncoding::v0::ENERGY_TRUNCATION) {
    truncatedEnergy = EnergyEncoding::v0::ENERGY_TRUNCATION;
  }
  return static_cast<int16_t>(std::round(truncatedEnergy / EnergyEncoding::v0::ENERGY_RESOLUTION));
}

uint16_t Cell::encodeEnergyV1(float energy, ChannelType_t celltype)
{
  double truncatedEnergy = energy;
  if (truncatedEnergy < 0.) {
    truncatedEnergy = 0.;
  } else if (truncatedEnergy > EnergyEncoding::v1::ENERGY_TRUNCATION) {
    truncatedEnergy = EnergyEncoding::v1::ENERGY_TRUNCATION;
  }
  float resolutionApplied = 0., energyOffset = 0.;
  switch (celltype) {
    case ChannelType_t::HIGH_GAIN: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_HG;
      break;
    }
    case ChannelType_t::LOW_GAIN: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_LG;
      energyOffset = EnergyEncoding::v1::HGLGTRANSITION;
      break;
    }
    case ChannelType_t::TRU: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_TRU;
      break;
    }
    case ChannelType_t::LEDMON: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_LEDMON;
      break;
    }
  }
  return static_cast<uint16_t>(std::round((truncatedEnergy - energyOffset) / resolutionApplied));
};

uint16_t Cell::V0toV1(uint16_t energyBits, ChannelType_t celltype)
{
  auto decodedEnergy = decodeEnergyV0(energyBits);
  return encodeEnergyV1(decodedEnergy, celltype);
}

float Cell::decodeEnergyV0(uint16_t energyBits)
{
  return static_cast<float>(energyBits) * EnergyEncoding::v0::ENERGY_RESOLUTION;
}

float Cell::decodeEnergyV1(uint16_t energyBits, ChannelType_t celltype)
{
  float resolutionApplied = 0.,
        energyOffset = 0.;
  switch (celltype) {
    case ChannelType_t::HIGH_GAIN: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_HG;
    }
    case ChannelType_t::LOW_GAIN: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_LG;
      energyOffset = EnergyEncoding::v1::HGLGTRANSITION;
    }
    case ChannelType_t::TRU: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_TRU;
    }
    case ChannelType_t::LEDMON: {
      resolutionApplied = EnergyEncoding::v1::ENERGY_RESOLUTION_LEDMON;
    }
  }
  return (static_cast<float>(energyBits) * resolutionApplied) + energyOffset;
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
