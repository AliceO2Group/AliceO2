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

#include "Framework/ConfigParamStore.h"
#include <boost/property_tree/ptree.hpp>

#include <concepts>
#include <cstdint>
#include <memory>
#include <string>

template <typename T>
concept SimpleConfigValueType = std::same_as<T, int> ||
                                std::same_as<T, int8_t> ||
                                std::same_as<T, int16_t> ||
                                std::same_as<T, uint8_t> ||
                                std::same_as<T, uint16_t> ||
                                std::same_as<T, unsigned int> ||
                                std::same_as<T, unsigned long> ||
                                std::same_as<T, unsigned long long> ||
                                std::same_as<T, long> ||
                                std::same_as<T, long long> ||
                                std::same_as<T, float> ||
                                std::same_as<T, double> ||
                                std::same_as<T, bool>;

template <typename T>
concept VectorConfigValueType = std::same_as<T, std::vector<int>> ||
                                std::same_as<T, std::vector<float>> ||
                                std::same_as<T, std::vector<double>> ||
                                std::same_as<T, std::vector<std::string>> ||
                                std::same_as<T, std::vector<bool>>;

template <typename T>
concept StringConfigValueType = std::same_as<T, std::string>;

template <typename T>
concept PtreeConfigValueType = std::same_as<T, boost::property_tree::ptree> || std::constructible_from<T, boost::property_tree::ptree>;

template <typename T>
concept Array2DLike = requires(T& t) { t.is_array_2d(); };

template <typename T>
concept LabeledArrayLike = requires(T& t) { t.is_labeled_array(); };

template <typename T>
concept ConfigValueType = SimpleConfigValueType<T> || StringConfigValueType<T> || VectorConfigValueType<T> || Array2DLike<T> || LabeledArrayLike<T>;

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
  ConfigParamRegistry(std::unique_ptr<ConfigParamStore> store);

  bool isSet(const char* key) const;

  bool hasOption(const char* key) const;

  bool isDefault(const char* key) const;

  [[nodiscard]] std::vector<ConfigParamSpec> const& specs() const;

  template <ConfigValueType T>
  T get(const char* key) const;

  template <typename T>
  T get(const char* key) const;

  void override(const char* key, ConfigValueType auto const& val) const;

  // Remove an option (a leaf or a whole subtree) from the active store, no-op if it is missing
  void erase(const char* key) const;

  // Load extra parameters discovered while we process data
  void loadExtra(std::vector<ConfigParamSpec>& extras);

 private:
  std::unique_ptr<ConfigParamStore> mStore;
};

template <typename T>
T ConfigParamRegistry::get(const char* key) const
{
  try {
    return T{mStore->store().get_child(key)};
  } catch (std::exception& e) {
    throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
  } catch (...) {
    throw std::invalid_argument(std::string("error parsing option: ") + key);
  }
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_CONFIGPARAMREGISTRY_H_
