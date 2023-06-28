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
#ifndef O2_FRAMEWORK_CONFIGPARAMREGISTRY_H_
#define O2_FRAMEWORK_CONFIGPARAMREGISTRY_H_

#include "Framework/ParamRetriever.h"
#include "Framework/ConfigParamStore.h"
#include "Framework/Traits.h"
#include "Framework/VariantPropertyTreeHelpers.h"

#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <string>
#include <cassert>

namespace
{
template <typename T>
constexpr auto isSimpleType()
{
  return std::is_same_v<T, int> ||
         std::is_same_v<T, int8_t> ||
         std::is_same_v<T, int16_t> ||
         std::is_same_v<T, uint8_t> ||
         std::is_same_v<T, uint16_t> ||
         std::is_same_v<T, uint32_t> ||
         std::is_same_v<T, uint64_t> ||
         std::is_same_v<T, int64_t> ||
         std::is_same_v<T, long> ||
         std::is_same_v<T, float> ||
         std::is_same_v<T, double> ||
         std::is_same_v<T, bool>;
}
} // namespace

namespace o2::framework
{
class ConfigParamStore;

/// This provides unified access to the parameters specified in the workflow
/// specification.
/// The ParamRetriever is a concrete implementation of the registry which
/// will actually get the options. For example it could get them from the
/// FairMQ ProgOptions plugin or (to run "device-less", e.g. in batch simulation
/// jobs).
class ConfigParamRegistry
{
 public:
  ConfigParamRegistry(std::unique_ptr<ConfigParamStore> store)
    : mStore{std::move(store)}
  {
  }

  bool isSet(const char* key) const
  {
    return mStore->store().count(key);
  }

  bool hasOption(const char* key) const
  {
    return mStore->store().get_child_optional(key).is_initialized();
  }

  bool isDefault(const char* key) const
  {
    return mStore->store().count(key) > 0 && mStore->provenance(key) != "default";
  }

  [[nodiscard]] std::vector<ConfigParamSpec> const& specs() const
  {
    return mStore->specs();
  }

  template <typename T>
  T get(const char* key) const
  {
    assert(mStore.get());
    try {
      if constexpr (isSimpleType<T>()) {
        return mStore->store().get<T>(key);
      } else if constexpr (std::is_same_v<T, std::string>) {
        return mStore->store().get<std::string>(key);
      } else if constexpr (std::is_same_v<T, std::string_view>) {
        return std::string_view{mStore->store().get<std::string>(key)};
      } else if constexpr (is_base_of_template_v<std::vector, T>) {
        return vectorFromBranch<typename T::value_type>(mStore->store().get_child(key));
      } else if constexpr (is_base_of_template_v<o2::framework::Array2D, T>) {
        return array2DFromBranch<typename T::element_t>(mStore->store().get_child(key));
      } else if constexpr (is_base_of_template_v<o2::framework::LabeledArray, T>) {
        return labeledArrayFromBranch<typename T::element_t>(mStore->store().get_child(key));
      } else if constexpr (std::is_same_v<T, boost::property_tree::ptree>) {
        return mStore->store().get_child(key);
      } else if constexpr (std::is_constructible_v<T, boost::property_tree::ptree>) {
        return T{mStore->store().get_child(key)};
      } else if constexpr (std::is_constructible_v<T, boost::property_tree::ptree> == false) {
        static_assert(std::is_constructible_v<T, boost::property_tree::ptree> == false,
                      "Not a basic type and no constructor from ptree provided");
      }
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("error parsing option: ") + key);
    }
    throw std::invalid_argument(std::string("bad type for option: ") + key);
  }

  template <typename T>
  void override(const char* key, const T& val) const
  {
    assert(mStore.get());
    try {
      mStore->store().put(key, val);
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("failed to store an option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("failed to store an option: ") + key);
    }
  }

 private:
  std::unique_ptr<ConfigParamStore> mStore;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CONFIGPARAMREGISTRY_H_
