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

Cell::Cell()
{
  mtower = std::numeric_limits<short>::max();
  menergy = std::numeric_limits<float>::min();
  mtime = std::numeric_limits<float>::min();
  mchi2 = std::numeric_limits<float>::max();
  mtype = ChannelType_t::HIGH_GAIN;
}

Cell::Cell(short tower, float energy, float time, ChannelType_t type, float chi2)
{
  mtower = tower;
  menergy = energy;
  mtime = time;
  mtype = type;
  mchi2 = chi2;
  
}

void Cell::setAll(short tower, float energy, float time, ChannelType_t type, float chi2)
{
  mtower = tower;
  menergy = energy;
  mtime = time;
  mtype = type;
  mchi2 = chi2;
}

void Cell::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell: Type " << getType() << ", Energy " << getEnergy() << ", Time " << getTimeStamp() << ", Tower " << getTower() << ", Chi2 " << getChi2();
}

std::ostream& operator<<(std::ostream& stream, const Cell& c)
{
  c.PrintStream(stream);
  return stream;
}
