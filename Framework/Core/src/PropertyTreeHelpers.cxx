// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PropertyTreeHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Variant.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/program_options/variables_map.hpp>

#include <vector>

namespace o2
{
namespace framework
{

void PropertyTreeHelpers::populate(std::vector<ConfigParamSpec> const& schema, boost::property_tree::ptree& pt, boost::program_options::variables_map const& vmap)
{
  for (auto& spec : schema) {
    switch (spec.type) {
      case VariantType::Int:
        pt.put(spec.name, vmap[spec.name].as<int>());
        break;
      case VariantType::Int64:
        pt.put(spec.name, vmap[spec.name].as<int64_t>());
        break;
      case VariantType::Float:
        pt.put(spec.name, vmap[spec.name].as<float>());
        break;
      case VariantType::Double:
        pt.put(spec.name, vmap[spec.name].as<double>());
        break;
      case VariantType::String:
        pt.put(spec.name, vmap[spec.name].as<std::string>());
        break;
      case VariantType::Bool:
        pt.put(spec.name, vmap[spec.name].as<bool>());
        break;
      case VariantType::Unknown:
      case VariantType::Empty:
      default:
        throw std::runtime_error("Unknown variant type");
    }
  }
}

} // namespace framework
} // namespace o2
