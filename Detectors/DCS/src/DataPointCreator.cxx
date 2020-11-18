// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&val), seconds, msec, flags, DeliveryType::RAW_DOUBLE);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, int32_t val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&val), seconds, msec, flags, DeliveryType::RAW_INT);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, uint32_t val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&val), seconds, msec, flags, DeliveryType::RAW_UINT);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, char val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&val), seconds, msec, flags, DeliveryType::RAW_CHAR);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, bool val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(&val), seconds, msec, flags, DeliveryType::RAW_BOOL);
}

template <>
DataPointCompositeObject createDataPointCompositeObject(const std::string& alias, std::string val, uint32_t seconds, uint16_t msec, uint16_t flags)
{
  return createDPCOM(alias, reinterpret_cast<const uint64_t*>(val.c_str()), seconds, msec, flags, DeliveryType::RAW_STRING);
}

} // namespace o2::dcs
