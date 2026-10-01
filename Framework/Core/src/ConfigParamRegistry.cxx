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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/VariantPropertyTreeHelpers.h"
#include "Framework/Array2D.h"

#include <type_traits>

namespace o2::framework
{

ConfigParamRegistry::ConfigParamRegistry(std::unique_ptr<ConfigParamStore> store)
  : mStore{std::move(store)}
{
}

bool ConfigParamRegistry::isSet(const char* key) const
{
  return mStore->store().count(key);
}

bool ConfigParamRegistry::hasOption(const char* key) const
{
  return mStore->store().get_child_optional(key).is_initialized();
}

bool ConfigParamRegistry::isDefault(const char* key) const
{
  return mStore->store().count(key) > 0 && mStore->provenance(key) != "default";
}

namespace
{
template <SimpleConfigValueType T>
T getImpl(boost::property_tree::ptree const& tree, const char* key)
{
  return tree.get<T>(key);
}

template <StringConfigValueType T>
T getImpl(boost::property_tree::ptree const& tree, const char* key)
{
  return tree.get<std::string>(key);
}

template <typename T>
  requires VectorConfigValueType<T>
auto getImpl(boost::property_tree::ptree const& tree, const char* key)
{
  return o2::framework::vectorFromBranch<typename T::value_type>(tree.get_child(key));
}

template <Array2DLike T>
auto getImpl(boost::property_tree::ptree& tree, const char* key)
{
  return array2DFromBranch<typename T::element_t>(tree.get_child(key));
}

template <LabeledArrayLike T>
auto getImpl(boost::property_tree::ptree& tree, const char* key)
{
  return labeledArrayFromBranch<typename T::element_t>(tree.get_child(key));
}
} // namespace

template <ConfigValueType T>
T ConfigParamRegistry::get(const char* key) const
{
  try {
    return getImpl<T>(this->mStore->store(), key);
  } catch (std::exception& e) {
    throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
  } catch (...) {
    throw std::invalid_argument(std::string("error parsing option: ") + key);
  }
}

void ConfigParamRegistry::override(const char* key, ConfigValueType auto const& val) const
{
  using T = std::decay_t<decltype(val)>;
  try {
    auto& store = mStore->store();
    if constexpr (SimpleConfigValueType<T> || StringConfigValueType<T>) {
      store.put(key, val);
    } else if constexpr (VectorConfigValueType<T>) {
      // arrays are stored as subtrees, which have to be replaced as a whole
      store.put_child(key, vectorToBranch(val.data(), val.size()));
    } else if constexpr (Array2DLike<T>) {
      store.put_child(key, array2DToBranch(T(val)));
    } else if constexpr (LabeledArrayLike<T>) {
      store.put_child(key, labeledArrayToBranch(T(val)));
    }
  } catch (std::exception& e) {
    throw std::invalid_argument(std::string("failed to store an option: ") + key + " (" + e.what() + ")");
  } catch (...) {
    throw std::invalid_argument(std::string("failed to store an option: ") + key);
  }
}

// Load extra parameters discovered while we process data
void ConfigParamRegistry::loadExtra(std::vector<ConfigParamSpec>& extras)
{
  mStore->load(extras);
}

[[nodiscard]] std::vector<ConfigParamSpec> const& ConfigParamRegistry::specs() const
{
  return mStore->specs();
}

template int8_t ConfigParamRegistry::get<int8_t>(const char* key) const;
template short ConfigParamRegistry::get<short>(const char* key) const;
template int ConfigParamRegistry::get<int>(const char* key) const;
template long ConfigParamRegistry::get<long>(const char* key) const;
template long long ConfigParamRegistry::get<long long>(const char* key) const;
template uint8_t ConfigParamRegistry::get<uint8_t>(const char* key) const;
template uint16_t ConfigParamRegistry::get<uint16_t>(const char* key) const;
template unsigned long ConfigParamRegistry::get<unsigned long>(const char* key) const;
template unsigned long long ConfigParamRegistry::get<unsigned long long>(const char* key) const;
template unsigned int ConfigParamRegistry::get<unsigned int>(const char* key) const;
template LabeledArray<std::string> ConfigParamRegistry::get<LabeledArray<std::string>>(const char* key) const;
template LabeledArray<double> ConfigParamRegistry::get<LabeledArray<double>>(const char* key) const;
template LabeledArray<float> ConfigParamRegistry::get<LabeledArray<float>>(const char* key) const;
template LabeledArray<int> ConfigParamRegistry::get<LabeledArray<int>>(const char* key) const;
template Array2D<std::string> ConfigParamRegistry::get<Array2D<std::string>>(const char* key) const;
template Array2D<double> ConfigParamRegistry::get<Array2D<double>>(const char* key) const;
template Array2D<float> ConfigParamRegistry::get<Array2D<float>>(const char* key) const;
template Array2D<int> ConfigParamRegistry::get<Array2D<int>>(const char* key) const;
template std::vector<std::string> ConfigParamRegistry::get<std::vector<std::string>>(const char* key) const;
template std::vector<double> ConfigParamRegistry::get<std::vector<double>>(const char* key) const;
template std::vector<float> ConfigParamRegistry::get<std::vector<float>>(const char* key) const;
template std::vector<int> ConfigParamRegistry::get<std::vector<int>>(const char* key) const;
template float ConfigParamRegistry::get<float>(const char* key) const;
template double ConfigParamRegistry::get<double>(const char* key) const;
template std::string ConfigParamRegistry::get<std::string>(const char* key) const;
template bool ConfigParamRegistry::get<bool>(const char* key) const;

template void ConfigParamRegistry::override(const char* key, int8_t const&) const;
template void ConfigParamRegistry::override(const char* key, int16_t const&) const;
template void ConfigParamRegistry::override(const char* key, int32_t const&) const;
template void ConfigParamRegistry::override(const char* key, int64_t const&) const;
template void ConfigParamRegistry::override(const char* key, uint8_t const&) const;
template void ConfigParamRegistry::override(const char* key, uint16_t const&) const;
template void ConfigParamRegistry::override(const char* key, uint32_t const&) const;
template void ConfigParamRegistry::override(const char* key, uint64_t const&) const;
template void ConfigParamRegistry::override(const char* key, float const&) const;
template void ConfigParamRegistry::override(const char* key, double const&) const;
template void ConfigParamRegistry::override(const char* key, std::string const&) const;
template void ConfigParamRegistry::override(const char* key, bool const&) const;

template void ConfigParamRegistry::override(char const* key, LabeledArray<std::string> const&) const;
template void ConfigParamRegistry::override(char const* key, LabeledArray<double> const&) const;
template void ConfigParamRegistry::override(char const* key, LabeledArray<float> const&) const;
template void ConfigParamRegistry::override(char const* key, LabeledArray<int> const&) const;
template void ConfigParamRegistry::override(char const* key, Array2D<std::string> const&) const;
template void ConfigParamRegistry::override(char const* key, Array2D<double> const&) const;
template void ConfigParamRegistry::override(char const* key, Array2D<float> const&) const;
template void ConfigParamRegistry::override(char const* key, Array2D<int> const&) const;
template void ConfigParamRegistry::override(char const* key, std::vector<std::string> const&) const;
template void ConfigParamRegistry::override(char const* key, std::vector<double> const&) const;
template void ConfigParamRegistry::override(char const* key, std::vector<float> const&) const;
template void ConfigParamRegistry::override(char const* key, std::vector<int> const&) const;
} // namespace o2::framework
