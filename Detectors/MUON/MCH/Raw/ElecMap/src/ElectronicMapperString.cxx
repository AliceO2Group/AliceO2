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

#include "MCHRawElecMap/Mapper.h"
#include "MapCRU.h"
#include "MapFEC.h"

namespace o2::mch::raw
{

std::string ElectronicMapperString::sCruMap = "";
std::string ElectronicMapperString::sFecMap = "";

template <>
Elec2DetMapper createElec2DetMapper<ElectronicMapperString>(uint64_t /*timestamp*/)
{
  MapFEC m(ElectronicMapperString::sFecMap);
  return m;
}

template <>
FeeLink2SolarMapper createFeeLink2SolarMapper<ElectronicMapperString>()
{
  MapCRU m(ElectronicMapperString::sCruMap);
  return m;
}

} // namespace o2::mch::raw
