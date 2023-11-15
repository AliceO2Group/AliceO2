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

namespace TimeEncoding
{
const float TIME_SHIFT = 600.,
            TIME_RANGE = 1500.,
            TIME_RESOLUTION = TIME_RANGE / 2047.;

}
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

namespace v2
{
const float
  ENERGY_BITS = static_cast<float>(0x3FFF),
  SAFETYMARGIN = 0.2,
  HGLGTRANSITION = o2::emcal::constants::OVERFLOWCUT * o2::emcal::constants::EMCAL_ADCENERGY,
  OFFSET_LG = HGLGTRANSITION - SAFETYMARGIN,
  ENERGY_TRUNCATION = 250.,
  ENERGY_RESOLUTION_LG = (ENERGY_TRUNCATION - OFFSET_LG) / ENERGY_BITS,
  ENERGY_RESOLUTION_HG = HGLGTRANSITION / ENERGY_BITS,
  ENERGY_RESOLUTION_TRU = ENERGY_TRUNCATION / ENERGY_BITS,
  ENERGY_RESOLUTION_LEDMON = ENERGY_TRUNCATION / ENERGY_BITS;

}
} // namespace EnergyEncoding

namespace DecodingV0
{
struct __attribute__((packed)) CellDataPacked {
  uint16_t mTowerID : 15;   ///< bits 0-14   Tower ID
  uint16_t mTime : 11;      ///< bits 15-25: Time (signed, can become negative after calibration)
  uint16_t mEnergy : 14;    ///< bits 26-39: Energy
  uint16_t mCellStatus : 2; ///< bits 40-41: Cell status
  uint16_t mZerod : 6;      ///< bits 42-47: Zerod
};
} // namespace DecodingV0

Cell::Cell(short tower, float energy, float timestamp, ChannelType_t ctype) : mTowerID(tower), mEnergy(energy), mTimestamp(timestamp), mChannelType(ctype)
{
}

Cell::Cell(uint16_t towerBits, uint16_t energyBits, uint16_t timestampBits, uint16_t channelBits, EncoderVersion version)
{
  initialiseFromEncoded(towerBits, timestampBits, energyBits, channelBits, version);
}

uint16_t Cell::getTowerIDEncoded() const
{
  return mTowerID;
}

uint16_t Cell::getTimeStampEncoded() const
{
  return encodeTime(mTimestamp);
}

uint16_t Cell::getEnergyEncoded(EncoderVersion version) const
{
  uint16_t energyBits = 0;
  switch (version) {
    case EncoderVersion::EncodingV0:
      energyBits = encodeEnergyV0(mEnergy);
      break;

    case EncoderVersion::EncodingV1:
      energyBits = encodeEnergyV1(mEnergy, mChannelType);
      break;

    case EncoderVersion::EncodingV2:
      energyBits = encodeEnergyV2(mEnergy, mChannelType);
      break;
  }
  return energyBits;
}

uint16_t Cell::getCellTypeEncoded() const
{
  return static_cast<uint16_t>(mChannelType);
}

void Cell::setEnergyEncoded(uint16_t energyBits, uint16_t channelTypeBits, EncoderVersion version)
{
  switch (version) {
    case EncoderVersion::EncodingV0:
      mEnergy = decodeEnergyV0(energyBits);
      break;
    case EncoderVersion::EncodingV1:
      mEnergy = decodeEnergyV1(energyBits, static_cast<ChannelType_t>(channelTypeBits));
      break;
    case EncoderVersion::EncodingV2:
      mEnergy = decodeEnergyV2(energyBits, static_cast<ChannelType_t>(channelTypeBits));
      break;
  }
}

void Cell::setTimestampEncoded(uint16_t timestampBits)
{
  mTimestamp = decodeTime(timestampBits);
}

void Cell::setTowerIDEncoded(uint16_t towerIDBits)
{
  mTowerID = towerIDBits;
}

void Cell::setChannelTypeEncoded(uint16_t channelTypeBits)
{
  mChannelType = static_cast<ChannelType_t>(channelTypeBits);
}

void Cell::initializeFromPackedBitfieldV0(const char* bitfield)
{
  auto bitrepresentation = reinterpret_cast<const DecodingV0::CellDataPacked*>(bitfield);
  mEnergy = decodeEnergyV0(bitrepresentation->mEnergy);
  mTimestamp = decodeTime(bitrepresentation->mTime);
  mTowerID = bitrepresentation->mTowerID;
  mChannelType = static_cast<ChannelType_t>(bitrepresentation->mCellStatus);
}

float Cell::getEnergyFromPackedBitfieldV0(const char* bitfield)
{
  return decodeEnergyV0(reinterpret_cast<const DecodingV0::CellDataPacked*>(bitfield)->mEnergy);
}

float Cell::getTimeFromPackedBitfieldV0(const char* bitfield)
{
  return decodeTime(reinterpret_cast<const DecodingV0::CellDataPacked*>(bitfield)->mTime);
}

ChannelType_t Cell::getCellTypeFromPackedBitfieldV0(const char* bitfield)
{
  return static_cast<ChannelType_t>(reinterpret_cast<const DecodingV0::CellDataPacked*>(bitfield)->mCellStatus);
}

short Cell::getTowerFromPackedBitfieldV0(const char* bitfield)
{
  return reinterpret_cast<const DecodingV0::CellDataPacked*>(bitfield)->mTowerID;
}

void Cell::truncate(EncoderVersion version)
{
  setEnergyEncoded(getEnergyEncoded(version), getCellTypeEncoded(), version);
  setTimestampEncoded(getTimeStampEncoded());
}

uint16_t Cell::encodeTime(float timestamp)
{
  // truncate
  auto timestampTruncated = timestamp;
  const float TIME_MIN = -1. * TimeEncoding::TIME_SHIFT,
              TIME_MAX = TimeEncoding::TIME_RANGE - TimeEncoding::TIME_SHIFT;
  if (timestampTruncated < TIME_MIN) {
    timestampTruncated = TIME_MIN;
  } else if (timestampTruncated > TIME_MAX) {
    timestampTruncated = TIME_MAX;
  }
  return static_cast<uint16_t>(std::round((timestampTruncated + TimeEncoding::TIME_SHIFT) / TimeEncoding::TIME_RESOLUTION));
}

uint16_t Cell::encodeEnergyV0(float energy)
{
  auto truncatedEnergy = energy;
  if (truncatedEnergy < 0.) {
    truncatedEnergy = 0.;
  } else if (truncatedEnergy > EnergyEncoding::v0::ENERGY_TRUNCATION) {
    truncatedEnergy = EnergyEncoding::v0::ENERGY_TRUNCATION;
  }
  return static_cast<uint16_t>(std::round(truncatedEnergy / EnergyEncoding::v0::ENERGY_RESOLUTION));
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

uint16_t Cell::encodeEnergyV2(float energy, ChannelType_t celltype)
{
  double truncatedEnergy = energy;
  if (truncatedEnergy < 0.) {
    truncatedEnergy = 0.;
  } else if (truncatedEnergy > EnergyEncoding::v2::ENERGY_TRUNCATION) {
    truncatedEnergy = EnergyEncoding::v2::ENERGY_TRUNCATION;
  }
  float resolutionApplied = 0., energyOffset = 0.;
  switch (celltype) {
    case ChannelType_t::HIGH_GAIN: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_HG;
      break;
    }
    case ChannelType_t::LOW_GAIN: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_LG;
      energyOffset = EnergyEncoding::v2::OFFSET_LG;
      break;
    }
    case ChannelType_t::TRU: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_TRU;
      break;
    }
    case ChannelType_t::LEDMON: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_LEDMON;
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

uint16_t Cell::V0toV2(uint16_t energyBits, ChannelType_t celltype)
{
  auto decodedEnergy = decodeEnergyV0(energyBits);
  return encodeEnergyV2(decodedEnergy, celltype);
}

uint16_t Cell::V1toV2(uint16_t energyBits, ChannelType_t celltype)
{
  auto decodedEnergy = decodeEnergyV1(energyBits, celltype);
  return encodeEnergyV2(decodedEnergy, celltype);
}

float Cell::decodeTime(uint16_t timestampBits)
{
  return (static_cast<float>(timestampBits) * TimeEncoding::TIME_RESOLUTION) - TimeEncoding::TIME_SHIFT;
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
  return (static_cast<float>(energyBits) * resolutionApplied) + energyOffset;
}

float Cell::decodeEnergyV2(uint16_t energyBits, ChannelType_t celltype)
{
  float resolutionApplied = 0.,
        energyOffset = 0.;
  switch (celltype) {
    case ChannelType_t::HIGH_GAIN: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_HG;
      break;
    }
    case ChannelType_t::LOW_GAIN: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_LG;
      energyOffset = EnergyEncoding::v2::OFFSET_LG;
      break;
    }
    case ChannelType_t::TRU: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_TRU;
      break;
    }
    case ChannelType_t::LEDMON: {
      resolutionApplied = EnergyEncoding::v2::ENERGY_RESOLUTION_LEDMON;
      break;
    }
  }
  return (static_cast<float>(energyBits) * resolutionApplied) + energyOffset;
}

void Cell::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell: Type " << getType() << ", Energy " << getEnergy() << ", Time " << getTimeStamp() << ", Tower " << getTower();
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const Cell& c)
{
  c.PrintStream(stream);
  return stream;
}
