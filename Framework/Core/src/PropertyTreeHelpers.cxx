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

#include "PropertyTreeHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/VariantStringHelpers.h"
#include "Framework/VariantPropertyTreeHelpers.h"
#include "Framework/RuntimeError.h"

#include <boost/program_options/variables_map.hpp>

#include <vector>
#include <string>

namespace o2::framework
{

void PropertyTreeHelpers::populateDefaults(std::vector<ConfigParamSpec> const& schema,
                                           boost::property_tree::ptree& pt,
                                           boost::property_tree::ptree& provenance)
{
  for (auto const& spec : schema) {
    std::string key = spec.name.substr(0, spec.name.find(','));
    try {
      if (spec.defaultValue.type() == VariantType::Empty) {
        continue;
      }
      switch (spec.type) {
        case VariantType::Int:
          pt.put(key, spec.defaultValue.get<int>());
          break;
        case VariantType::Int8:
          pt.put(key, spec.defaultValue.get<int8_t>());
          break;
        case VariantType::Int16:
          pt.put(key, spec.defaultValue.get<int16_t>());
          break;
        case VariantType::UInt8:
          pt.put(key, spec.defaultValue.get<uint8_t>());
          break;
        case VariantType::UInt16:
          pt.put(key, spec.defaultValue.get<uint16_t>());
          break;
        case VariantType::UInt32:
          pt.put(key, spec.defaultValue.get<uint32_t>());
          break;
        case VariantType::UInt64:
          pt.put(key, spec.defaultValue.get<uint64_t>());
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
        case VariantType::Dict:
          pt.put_child(key, boost::property_tree::ptree{});
          break;
        case VariantType::ArrayInt:
          pt.put_child(key, vectorToBranch(spec.defaultValue.get<int*>(), spec.defaultValue.size()));
          break;
        case VariantType::ArrayFloat:
          pt.put_child(key, vectorToBranch(spec.defaultValue.get<float*>(), spec.defaultValue.size()));
          break;
        case VariantType::ArrayDouble:
          pt.put_child(key, vectorToBranch(spec.defaultValue.get<double*>(), spec.defaultValue.size()));
          break;
        case VariantType::ArrayBool:
          pt.put_child(key, vectorToBranch(spec.defaultValue.get<bool*>(), spec.defaultValue.size()));
          break;
        case VariantType::ArrayString:
          pt.put_child(key, vectorToBranch(spec.defaultValue.get<std::string*>(), spec.defaultValue.size()));
          break;
        case VariantType::Array2DInt:
          pt.put_child(key, array2DToBranch(spec.defaultValue.get<Array2D<int>>()));
          break;
        case VariantType::Array2DFloat:
          pt.put_child(key, array2DToBranch(spec.defaultValue.get<Array2D<float>>()));
          break;
        case VariantType::Array2DDouble:
          pt.put_child(key, array2DToBranch(spec.defaultValue.get<Array2D<double>>()));
          break;
        case VariantType::LabeledArrayInt:
          pt.put_child(key, labeledArrayToBranch(spec.defaultValue.get<LabeledArray<int>>()));
          break;
        case VariantType::LabeledArrayFloat:
          pt.put_child(key, labeledArrayToBranch(spec.defaultValue.get<LabeledArray<float>>()));
          break;
        case VariantType::LabeledArrayDouble:
          pt.put_child(key, labeledArrayToBranch(spec.defaultValue.get<LabeledArray<double>>()));
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw runtime_error_f("Unknown variant type", spec.type);
      }
      provenance.put(key, "default");
    } catch (std::runtime_error& re) {
      throw;
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
  for (auto const& spec : schema) {
    // strip short version to get the correct key
    std::string key = spec.name.substr(0, spec.name.find(','));
    if (vmap.count(key) == 0) {
      continue;
    }
    try {
      switch (spec.type) {
        case VariantType::Int:
          pt.put(key, vmap[key].as<int>());
          break;
        case VariantType::Int8:
          pt.put(key, vmap[key].as<int8_t>());
          break;
        case VariantType::Int16:
          pt.put(key, vmap[key].as<int16_t>());
          break;
        case VariantType::UInt8:
          pt.put(key, vmap[key].as<uint8_t>());
          break;
        case VariantType::UInt16:
          pt.put(key, vmap[key].as<uint16_t>());
          break;
        case VariantType::UInt32:
          pt.put(key, vmap[key].as<uint32_t>());
          break;
        case VariantType::UInt64:
          pt.put(key, vmap[key].as<uint64_t>());
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
          if (auto const* v = boost::any_cast<std::string>(&vmap[key].value())) {
            pt.put(key, *v);
          }
          break;
        case VariantType::Bool:
          pt.put(key, vmap[key].as<bool>());
          break;
        case VariantType::ArrayInt:
          pt.put_child(key, vectorToBranch<int>(stringToVector<int>(vmap[key].as<std::string>())));
          break;
        case VariantType::ArrayFloat:
          pt.put_child(key, vectorToBranch<float>(stringToVector<float>(vmap[key].as<std::string>())));
          break;
        case VariantType::ArrayDouble:
          pt.put_child(key, vectorToBranch<double>(stringToVector<double>(vmap[key].as<std::string>())));
          break;
        case VariantType::ArrayBool:
          //          pt.put_child(key, vectorToBranch<bool>(stringToVector<bool>(vmap[key].as<std::string>())));
          break;
        case VariantType::ArrayString:
          pt.put_child(key, vectorToBranch<std::string>(stringToVector<std::string>(vmap[key].as<std::string>())));
          break;
        case VariantType::Array2DInt:
          pt.put_child(key, array2DToBranch<int>(stringToArray2D<int>(vmap[key].as<std::string>())));
          break;
        case VariantType::Array2DFloat:
          pt.put_child(key, array2DToBranch<float>(stringToArray2D<float>(vmap[key].as<std::string>())));
          break;
        case VariantType::Array2DDouble:
          pt.put_child(key, array2DToBranch<double>(stringToArray2D<double>(vmap[key].as<std::string>())));
          break;
        case VariantType::Dict:
          pt.put_child(key, vmap[key].as<boost::property_tree::ptree>());
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw runtime_error("Unknown variant type");
      }
      provenance.put(key, "fairmq");
    } catch (std::runtime_error& re) {
      throw;
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("missing option: ") + key);
    }
  }
}

template <typename T>
auto replaceLabels(LabeledArray<T>& input, LabeledArray<T>&& spec)
{
  if (!input.getLabelsCols().empty() || !input.getLabelsRows().empty()) {
    return false;
  }
  if (spec.getLabelsCols().empty() == false) {
    input.replaceLabelsCols(spec.getLabelsCols());
  }
  if (spec.getLabelsRows().empty() == false) {
    input.replaceLabelsRows(spec.getLabelsRows());
  }
  return true;
}

void PropertyTreeHelpers::populate(std::vector<ConfigParamSpec> const& schema,
                                   boost::property_tree::ptree& pt,
                                   boost::property_tree::ptree const& in,
                                   boost::property_tree::ptree& provenance,
                                   std::string const& provenanceLabel)
{
  for (auto const& spec : schema) {
    // strip short version to get the correct key
    std::string key = spec.name.substr(0, spec.name.find(','));
    auto it = in.get_child_optional(key);
    if (!it) {
      it = in.get_child_optional(boost::property_tree::path(key, '/'));
    }
    if (!it) {
      continue;
    }
    try {
      switch (spec.type) {
        case VariantType::Int:
          pt.put(key, (*it).get_value<int>());
          break;
        case VariantType::Int8:
          pt.put(key, (*it).get_value<int8_t>());
          break;
        case VariantType::Int16:
          pt.put(key, (*it).get_value<int16_t>());
          break;
        case VariantType::UInt8:
          pt.put(key, (*it).get_value<uint8_t>());
          break;
        case VariantType::UInt16:
          pt.put(key, (*it).get_value<uint16_t>());
          break;
        case VariantType::UInt32:
          pt.put(key, (*it).get_value<uint32_t>());
          break;
        case VariantType::UInt64:
          pt.put(key, (*it).get_value<uint64_t>());
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
        case VariantType::Dict:
        case VariantType::ArrayInt:
        case VariantType::ArrayFloat:
        case VariantType::ArrayDouble:
        case VariantType::ArrayBool:
        case VariantType::ArrayString:
        case VariantType::Array2DInt:
        case VariantType::Array2DFloat:
        case VariantType::Array2DDouble:
          pt.put_child(key, *it);
          break;
        case VariantType::LabeledArrayInt: {
          auto v = labeledArrayFromBranch<int>(it.value());
          if (!replaceLabels(v, spec.defaultValue.get<LabeledArray<int>>())) {
            pt.put_child(key, *it);
          } else {
            pt.put_child(key, labeledArrayToBranch(std::move(v)));
          }
        }; break;
        case VariantType::LabeledArrayFloat: {
          auto v = labeledArrayFromBranch<float>(it.value());
          if (!replaceLabels(v, spec.defaultValue.get<LabeledArray<float>>())) {
            pt.put_child(key, *it);
          } else {
            pt.put_child(key, labeledArrayToBranch(std::move(v)));
          }
        }; break;
        case VariantType::LabeledArrayDouble: {
          auto v = labeledArrayFromBranch<double>(it.value());
          if (!replaceLabels(v, spec.defaultValue.get<LabeledArray<double>>())) {
            pt.put_child(key, *it);
          } else {
            pt.put_child(key, labeledArrayToBranch(std::move(v)));
          }
        }; break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw std::runtime_error("Unknown variant type");
      }
      provenance.put(key, provenanceLabel);
    } catch (std::runtime_error& re) {
      throw;
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
                       PropertyTreeHelpers::WalkerFunction<boost::property_tree::ptree>& method)
{
  using boost::property_tree::ptree;

  method(parent, childPath, child);
  for (ptree::const_iterator it = child.begin(); it != child.end(); ++it) {
    ptree::path_type curPath = childPath / ptree::path_type(it->first);
    traverseRecursive(parent, curPath, it->second, method);
  }
}
} // namespace

template <>
void PropertyTreeHelpers::traverse<boost::property_tree::ptree>(const boost::property_tree::ptree& parent, PropertyTreeHelpers::WalkerFunction<boost::property_tree::ptree>& method)
{
  traverseRecursive(parent, "", parent, method);
}

} // namespace o2::framework
