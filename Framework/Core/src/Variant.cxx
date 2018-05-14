// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/Variant.h"
#include <iostream>

namespace o2
{
namespace framework
{

std::ostream& operator<<(std::ostream& oss, Variant const& val)
{
  switch (val.type()) {
    case VariantType::Int:
      oss << val.get<int>();
      break;
    case VariantType::Int64:
      oss << val.get<int64_t>();
      break;
    case VariantType::Float:
      oss << val.get<float>();
      break;
    case VariantType::Double:
      oss << val.get<double>();
      break;
    case VariantType::String:
      oss << val.get<const char*>();
      break;
    case VariantType::Bool:
      oss << val.get<bool>();
      break;
    case VariantType::Empty:
      break;
    default:
      oss << "undefined";
      break;
  };
  return oss;
}

} // namespace framework
} // namespace o2
