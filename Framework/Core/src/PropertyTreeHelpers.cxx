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
#include <boost/lexical_cast.hpp>

#include <vector>
#include <string>
#include <regex>

namespace o2::framework
{

void PropertyTreeHelpers::populateDefaults(std::vector<ConfigParamSpec> const& schema,
                                           boost::property_tree::ptree& pt,
                                           boost::property_tree::ptree& provenance)
{
  auto addBranch = [&](std::string const& key, auto* values, size_t size) {
    boost::property_tree::ptree branch;
    for (auto i = 0u; i < size; ++i) {
      boost::property_tree::ptree leaf;
      leaf.put("", values[i]);
      branch.push_back(std::make_pair("", leaf));
    }
    pt.put_child(key, branch);
  };

  auto addSubTree = [&](std::string const& key, auto const& m) {
    boost::property_tree::ptree subtree;
    for (auto i = 0u; i < m.rows; ++i) {
      boost::property_tree::ptree branch;
      for (auto j = 0u; j < m.cols; ++j) {
        boost::property_tree::ptree leaf;
        leaf.put("", m(i, j));
        branch.push_back(std::make_pair("", leaf));
      }
      subtree.push_back(std::make_pair("", branch));
    }
    pt.put_child(key, subtree);
  };

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
        case VariantType::ArrayInt:
          addBranch(key, spec.defaultValue.get<int*>(), spec.defaultValue.size());
          break;
        case VariantType::ArrayFloat:
          addBranch(key, spec.defaultValue.get<float*>(), spec.defaultValue.size());
          break;
        case VariantType::ArrayDouble:
          addBranch(key, spec.defaultValue.get<double*>(), spec.defaultValue.size());
          break;
        case VariantType::ArrayBool:
          addBranch(key, spec.defaultValue.get<bool*>(), spec.defaultValue.size());
          break;
        case VariantType::ArrayString:
          addBranch(key, spec.defaultValue.get<std::string*>(), spec.defaultValue.size());
          break;
        case VariantType::MatrixInt:
          addSubTree(key, spec.defaultValue.get<Array2D<int>>());
          break;
        case VariantType::MatrixFloat:
          addSubTree(key, spec.defaultValue.get<Array2D<float>>());
          break;
        case VariantType::MatrixDouble:
          addSubTree(key, spec.defaultValue.get<Array2D<double>>());
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw std::runtime_error("Unknown variant type");
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

namespace
{
template <typename T>
std::vector<T> toVector(std::string const& input)
{
  std::vector<T> result;
  //check if the array string has correct array type symbol
  assert(input[0] == variant_array_symbol<T>::symbol);
  std::regex nmatch(R"((?:(?!=,)|(?!=\[))[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?(?=,|\]))");
  auto end = std::sregex_iterator();
  auto values = std::sregex_iterator(input.begin(), input.end(), nmatch);
  for (auto& v = values; v != end; ++v) {
    result.push_back(boost::lexical_cast<T>(v->str()));
  }
  return result;
}

template <>
std::vector<std::string> toVector(std::string const& input)
{
  std::vector<std::string> result;
  //check if the array string has correct array type symbol
  assert(input[0] == variant_array_symbol<std::string>::symbol);
  std::regex smatch(R"((?:(?!=,)|(?!=\[))\w+(?=,|\]))");
  auto end = std::sregex_iterator();
  auto values = std::sregex_iterator(input.begin(), input.end(), smatch);
  for (auto v = values; v != end; ++v) {
    result.push_back(v->str());
  }
  return result;
}

template <typename T>
Array2D<T> toMatrix(std::string const& input)
{
  std::vector<T> cache;
  assert(input[0] == variant_array_symbol<T>::symbol);
  std::regex mrows(R"(\[[^\[\]]+\])");
  std::regex marray(R"((?:(?!=,)|(?!=\[))[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?(?=,|\]))");
  auto end = std::sregex_iterator();
  auto rows = std::sregex_iterator(input.begin(), input.end(), mrows);
  uint32_t nrows = 0;
  uint32_t ncols = 0;
  bool first = true;
  for (auto& row = rows; row != end; ++row) {
    auto str = row->str();
    auto values = std::sregex_iterator(str.begin(), str.end(), marray);
    if (first) {
      ncols = 0;
    }
    for (auto& v = values; v != end; ++v) {
      cache.push_back(boost::lexical_cast<T>(v->str()));
      if (first) {
        ++ncols;
      }
    }
    if (first) {
      first = false;
    }
    ++nrows;
  }
  return Array2D<T>{cache, nrows, ncols};
}
} // namespace

void PropertyTreeHelpers::populate(std::vector<ConfigParamSpec> const& schema,
                                   boost::property_tree::ptree& pt,
                                   boost::program_options::variables_map const& vmap,
                                   boost::property_tree::ptree& provenance)
{
  auto addBranch = [&](std::string const& key, auto vector) {
    boost::property_tree::ptree branch;
    for (auto i = 0u; i < vector.size(); ++i) {
      boost::property_tree::ptree leaf;
      leaf.put("", vector[i]);
      branch.push_back(std::make_pair("", leaf));
    }
    pt.put_child(key, branch);
  };

  auto addSubTree = [&](std::string const& key, auto m) {
    boost::property_tree::ptree subtree;
    for (auto i = 0u; i < m.rows; ++i) {
      boost::property_tree::ptree branch;
      for (auto j = 0u; j < m.cols; ++j) {
        boost::property_tree::ptree leaf;
        leaf.put("", m(i, j));
        branch.push_back(std::make_pair("", leaf));
      }
      subtree.push_back(std::make_pair("", branch));
    }
    pt.put_child(key, subtree);
  };

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
        case VariantType::ArrayInt: {
          auto v = toVector<int>(vmap[key].as<std::string>());
          addBranch(key, v);
        };
          break;
        case VariantType::ArrayFloat: {
          auto v = toVector<float>(vmap[key].as<std::string>());
          addBranch(key, v);
        };
          break;
        case VariantType::ArrayDouble: {
          auto v = toVector<double>(vmap[key].as<std::string>());
          addBranch(key, v);
        };
          break;
        case VariantType::ArrayBool: {
          auto v = toVector<bool>(vmap[key].as<std::string>());
          addBranch(key, v);
        };
          break;
        case VariantType::ArrayString: {
          auto v = toVector<std::string>(vmap[key].as<std::string>());
          addBranch(key, v);
        };
          break;
        case VariantType::MatrixInt: {
          auto m = toMatrix<int>(vmap[key].as<std::string>());
          addSubTree(key, m);
        };
          break;
        case VariantType::MatrixFloat: {
          auto m = toMatrix<float>(vmap[key].as<std::string>());
          addSubTree(key, m);
        };
          break;
        case VariantType::MatrixDouble: {
          auto m = toMatrix<double>(vmap[key].as<std::string>());
          addSubTree(key, m);
        };
          break;
        case VariantType::Unknown:
        case VariantType::Empty:
        default:
          throw std::runtime_error("Unknown variant type");
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
        case VariantType::ArrayInt:
        case VariantType::ArrayFloat:
        case VariantType::ArrayDouble:
        case VariantType::ArrayBool:
        case VariantType::ArrayString:
        case VariantType::MatrixInt:
        case VariantType::MatrixFloat:
        case VariantType::MatrixDouble:
          pt.put_child(key, *it);
          break;
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

} // namespace o2::framework
