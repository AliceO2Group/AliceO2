// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsDCS/DataPointCompositeObject.h"

using namespace o2::dcs;

ClassImp(DataPointCompositeObject);

namespace o2::dcs
{
template<typename T, o2::dcs::DeliveryType dt>
T getValueImpl(const DataPointCompositeObject& dpcom) {
union Converter {
  uint64_t raw_data;
  T t_value;
};
  if (dpcom.id.get_type() != dt) {
    throw std::runtime_error("DPCOM is of unexpected type " + o2::dcs::show(dt));
  }
  Converter converter;
  converter.raw_data = dpcom.data.payload_pt1;
  return converter.t_value;
}

// only specialize the getValue function for the types we support :
//
// - double
// - uint32_t
// - int32_t
// - char 
// - bool

// template<>
// double getValueImpl<double,DeliveryType::RAW_DOUBLE>(const DataPointCompositeObject&);
//
template<>
double getValue(const DataPointCompositeObject& dpcom)
{
  return getValueImpl<double,DeliveryType::RAW_DOUBLE>(dpcom);
}

template <>
uint32_t getValue(const DataPointCompositeObject& dpcom)
{
  return getValueImpl<uint32_t,DeliveryType::RAW_UINT>(dpcom);
}

template <>
int32_t getValue(const DataPointCompositeObject& dpcom)
{
  return getValueImpl<int32_t,DeliveryType::RAW_INT>(dpcom);
}

template <>
char getValue(const DataPointCompositeObject& dpcom)
{
  return getValueImpl<char,DeliveryType::RAW_CHAR>(dpcom);
}

template <>
bool getValue(const DataPointCompositeObject& dpcom)
{
  return getValueImpl<bool,DeliveryType::RAW_BOOL>(dpcom);
}
} // namespace o2::dcs
