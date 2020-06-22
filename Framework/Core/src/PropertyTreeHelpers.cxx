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

void PropertyTreeHelpers::populateDefaults(std::vector<ConfigParamSpec> const& schema,
                                           boost::property_tree::ptree& pt,
                                           boost::property_tree::ptree& provenance)
{
  for (auto& spec : schema) {
    std::string key = spec.name.substr(0, spec.name.find(","));
    try {
      if (spec.defaultValue.type() == VariantType::Empty) {
        continue;
      }
      switch (spec.type) {
        case VariantType::Int:
          pt.put(key, spec.defaultValue.get<int>());
          break;
        case VariantType::Int64:
          pt.put(key, spec.defaultValue.get<int64_t>());
          break;
        case VariantType::Float:
          pt.put(key, spec.defaultValue.get<float>());
          break;
        case VariantType::Double:
          pt.put(key, spec.defaultValue.get<double>());
          break;
        case VariantType::String:
          pt.put(key, spec.defaultValue.get<std::string>());
          break;
        case VariantType::Bool:
          pt.put(key, spec.defaultValue.get<bool>());
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw std::runtime_error("Unknown variant type");
      }
      provenance.put(key, "default");
    } catch (std::runtime_error& re) {
      throw re;
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("missing option: ") + key);
    }
  }
}

void PropertyTreeHelpers::populate(std::vector<ConfigParamSpec> const& schema,
                                   boost::property_tree::ptree& pt,
                                   boost::program_options::variables_map const& vmap,
                                   boost::property_tree::ptree& provenance)
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
      provenance.put(key, "fairmq");
    } catch (std::runtime_error& re) {
      throw re;
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("missing option: ") + key);
    }
  }
}

void PropertyTreeHelpers::populate(std::vector<ConfigParamSpec> const& schema,
                                   boost::property_tree::ptree& pt,
                                   boost::property_tree::ptree const& in,
                                   boost::property_tree::ptree& provenance,
                                   std::string const& provenanceLabel)
{
  for (auto& spec : schema) {
    // strip short version to get the correct key
    std::string key = spec.name.substr(0, spec.name.find(","));
    auto it = in.get_child_optional(key);
    if (!it) {
      continue;
    }
    try {
      switch (spec.type) {
        case VariantType::Int:
          pt.put(key, (*it).get_value<int>());
          break;
        case VariantType::Int64:
          pt.put(key, (*it).get_value<int64_t>());
          break;
        case VariantType::Float:
          pt.put(key, (*it).get_value<float>());
          break;
        case VariantType::Double:
          pt.put(key, (*it).get_value<double>());
          break;
        case VariantType::String:
          pt.put(key, (*it).get_value<std::string>());
          break;
        case VariantType::Bool:
          pt.put(key, (*it).get_value<bool>());
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw std::runtime_error("Unknown variant type");
      }
      provenance.put(key, provenanceLabel);
    } catch (std::runtime_error& re) {
      throw re;
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("missing option: ") + key);
    }
  }
}

namespace
{
void traverseRecursive(const boost::property_tree::ptree& parent,
                       const boost::property_tree::ptree::path_type& childPath,
                       const boost::property_tree::ptree& child,
                       PropertyTreeHelpers::WalkerFunction& method)
{
  using boost::property_tree::ptree;

  method(parent, childPath, child);
  for (ptree::const_iterator it = child.begin(); it != child.end(); ++it) {
    ptree::path_type curPath = childPath / ptree::path_type(it->first);
    traverseRecursive(parent, curPath, it->second, method);
  }
}
} // namespace

void PropertyTreeHelpers::traverse(const boost::property_tree::ptree& parent, PropertyTreeHelpers::WalkerFunction& method)
{
  traverseRecursive(parent, "", parent, method);
}

void PropertyTreeHelpers::merge(boost::property_tree::ptree& dest,
                                boost::property_tree::ptree const& source,
                                boost::property_tree::ptree::path_type const& mergePoint)
{
  using boost::property_tree::ptree;
  WalkerFunction merge = [&dest, &mergePoint](ptree const& parent, ptree::path_type childPath, ptree const& child) {
    dest.put(mergePoint / childPath, child.data());
  };
  traverse(source, merge);
}

} // namespace o2::framework
