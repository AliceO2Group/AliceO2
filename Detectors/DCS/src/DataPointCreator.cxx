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

#include "DetectorsDCS/DataPointCreator.h"

namespace
{
o2::dcs::DataPointCompositeObject createDPCOM(const std::string& alias, const uint64_t* val, uint32_t seconds, uint16_t msec, uint16_t flags, o2::dcs::DeliveryType dt)
{
  auto dpid = o2::dcs::DataPointIdentifier(alias, dt);
  auto dpval = o2::dcs::DataPointValue(
    flags,
    msec,
    seconds,
    val,
    dt);
  return o2::dcs::DataPointCompositeObject(dpid, dpval);
}
} // namespace

namespace o2::dcs
{
template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, double val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&val), seconds, msec, flags, DeliveryType::DPVAL_DOUBLE);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, float val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  float tmp[2];
  tmp[0] = val;
  tmp[1] = 0;
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&tmp[0]), seconds, msec, flags, DeliveryType::DPVAL_FLOAT);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, int32_t val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  int64_t tmp{val};
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&tmp), seconds, msec, flags, DeliveryType::DPVAL_INT);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, uint32_t val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  uint64_t tmp{val};
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&tmp), seconds, msec, flags, DeliveryType::DPVAL_UINT);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, long long val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  uint64_t tmp{static_cast<uint64_t>(val)};
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&tmp), seconds, msec, flags, DeliveryType::DPVAL_UINT);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, char val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&val), seconds, msec, flags, DeliveryType::DPVAL_CHAR);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, bool val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  uint64_t tmp{val};
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&tmp), seconds, msec, flags, DeliveryType::DPVAL_BOOL);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, std::string val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  constexpr int N{56};
  char str[N];
  strncpy(str, val.c_str(), N);
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&str[0]), seconds, msec, flags, DeliveryType::DPVAL_STRING);
}

} // namespace o2::dcs
