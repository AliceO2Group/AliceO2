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
#include "DataFormatsEMCAL/CellCompressed.h"
#include <iostream>
#include <bitset>
#include <cmath>

using namespace o2::emcal;

CellCompressed::CellCompressed()
{
  memset(mCellWords, 0, sizeof(uint16_t) * 3);
}

CellCompressed::CellCompressed(short tower, float energy, float time, ChannelType_t ctype, float chi2)
{
  memset(mCellWords, 0, sizeof(uint16_t) * 3);
  setTower(tower);
  setTimeStamp(time);
  setType(ctype); // type needs to be set before energy to allow for proper conversion
  setEnergy(energy);
  setChi2(chi2);
}

void CellCompressed::setTimeStamp(float timestamp)
{
  // truncate:
  const float TIME_MIN = -1. * TIME_SHIFT,
              TIME_MAX = TIME_RANGE - TIME_SHIFT;
  if (timestamp < TIME_MIN) {
    timestamp = TIME_MIN;
  } else if (timestamp > TIME_MAX) {
    timestamp = TIME_MAX;
  }
  getDataRepresentation()->mTime = static_cast<uint16_t>(std::round((timestamp + TIME_SHIFT) / TIME_RESOLUTION));
}

float CellCompressed::getTimeStamp() const
{
  return (static_cast<float>(getDataRepresentation()->mTime) * TIME_RESOLUTION) - TIME_SHIFT;
}

void CellCompressed::setEnergy(float energy)
{
  double truncatedEnergy = energy;
  if (truncatedEnergy < 0.) {
    truncatedEnergy = 0.;
  } else if (truncatedEnergy > ENERGY_TRUNCATION) {
    truncatedEnergy = ENERGY_TRUNCATION;
  }
  switch (getType()) {
    case ChannelType_t::HIGH_GAIN: {
      getDataRepresentation()->mEnergy = static_cast<uint16_t>(std::round(truncatedEnergy / ENERGY_RESOLUTION_HG));
      break;
    }
    case ChannelType_t::LOW_GAIN: {
      getDataRepresentation()->mEnergy = static_cast<uint16_t>(std::round(truncatedEnergy / ENERGY_RESOLUTION_LG));
      break;
    }
    case ChannelType_t::TRU: {
      getDataRepresentation()->mEnergy = static_cast<uint16_t>(std::round(truncatedEnergy / ENERGY_RESOLUTION_TRU));
      break;
    }
    case ChannelType_t::LEDMON: {
      getDataRepresentation()->mEnergy = static_cast<uint16_t>(std::round(truncatedEnergy / ENERGY_RESOLUTION_LEDMON));
      break;
    }
  }
}

float CellCompressed::getEnergy() const
{
  switch (getType()) {
    case ChannelType_t::HIGH_GAIN: {
      return static_cast<float>(getDataRepresentation()->mEnergy) * ENERGY_RESOLUTION_HG;
    }
    case ChannelType_t::LOW_GAIN: {
      return static_cast<float>(getDataRepresentation()->mEnergy) * ENERGY_RESOLUTION_LG;
    }
    case ChannelType_t::TRU: {
      return static_cast<float>(getDataRepresentation()->mEnergy) * ENERGY_RESOLUTION_TRU;
    }
    case ChannelType_t::LEDMON: {
      return static_cast<float>(getDataRepresentation()->mEnergy) * ENERGY_RESOLUTION_LEDMON;
    }
  }
}

void CellCompressed::setChi2(float chi2)
{
  if (chi2 < 0.) {
    chi2 = 0.;
  } else if (chi2 > CHI2_TRUNCATION) {
    chi2 = CHI2_TRUNCATION;
  }
  getDataRepresentation()->mChi2 = static_cast<uint16_t>(std::round(chi2 / CHI2_RESOLUTION));
}

float CellCompressed::getChi2() const
{
  return static_cast<float>(getDataRepresentation()->mChi2) * CHI2_RESOLUTION;
}

void CellCompressed::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell: Type " << getType() << ", Energy " << getEnergy() << ", Time " << getTimeStamp() << ", Tower " << getTower() << ", Chi2 " << getChi2();
}

std::ostream& operator<<(std::ostream& stream, const CellCompressed& c)
{
  c.PrintStream(stream);
  return stream;
}
