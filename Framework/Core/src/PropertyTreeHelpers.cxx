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
#include <string>

namespace o2::framework
{

void PropertyTreeHelpers::populate(std::vector<ConfigParamSpec> const& schema, boost::property_tree::ptree& pt, boost::program_options::variables_map const& vmap)
{
  for (auto& spec : schema) {
    // strip short version to get the correct key
    std::string key = spec.name.substr(0, spec.name.find(","));
    if (vmap.count(key) == 0) {
      continue;
    }
    try {
      switch (spec.type) {
        case VariantType::Int:
          pt.put(key, vmap[key].as<int>());
          break;
        case VariantType::Int64:
          pt.put(key, vmap[key].as<int64_t>());
          break;
        case VariantType::Float:
          pt.put(key, vmap[key].as<float>());
          break;
        case VariantType::Double:
          pt.put(key, vmap[key].as<double>());
          break;
        case VariantType::String:
          if (auto v = boost::any_cast<std::string>(&vmap[key].value())) {
            pt.put(key, *v);
          }
          break;
        case VariantType::Bool:
          pt.put(key, vmap[key].as<bool>());
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw std::runtime_error("Unknown variant type");
      }
    } catch (std::runtime_error& re) {
      throw re;
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("missing option: ") + key);
    }
  }
}

void PropertyTreeHelpers::populate(std::vector<ConfigParamSpec> const& schema, boost::property_tree::ptree& pt, boost::property_tree::ptree const& in)
{
  for (auto& spec : schema) {
    // strip short version to get the correct key
    std::string key = spec.name.substr(0, spec.name.find(","));
    auto it = in.get_child_optional(key);
    try {
      switch (spec.type) {
        case VariantType::Int:
          pt.put(key, it ? (*it).get_value<int>() : spec.defaultValue.get<int>());
          break;
        case VariantType::Int64:
          pt.put(key, it ? (*it).get_value<int64_t>() : spec.defaultValue.get<int64_t>());
          break;
        case VariantType::Float:
          pt.put(key, it ? (*it).get_value<float>() : spec.defaultValue.get<float>());
          break;
        case VariantType::Double:
          pt.put(key, it ? (*it).get_value<double>() : spec.defaultValue.get<double>());
          break;
        case VariantType::String:
          pt.put(key, it ? (*it).get_value<std::string>() : spec.defaultValue.get<char*>());
          break;
        case VariantType::Bool:
          pt.put(key, it ? (*it).get_value<bool>() : spec.defaultValue.get<bool>());
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw std::runtime_error("Unknown variant type");
      }
    } catch (std::runtime_error& re) {
      throw re;
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("missing option: ") + key);
    }
  }
}

} // namespace o2::framework
