// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfig_genrtti.inc
/// \author David Rohr

// Provides static inline qConfigGenRTTI() in the namespace it is included in.
// Requires <algorithm>, <limits>, <optional>, <stdexcept>, <string>, <tuple>, <vector>, "qconfig_rtti.h", "qconfig_internal_helpers.h" to be included before (outside of the namespace), and the config structs to be visible unqualified.

#ifndef QCONFIG_GENRTTI_INC
#define QCONFIG_GENRTTI_INC

template <typename T>
static inline T qConfigRTTICheckedCast(size_t v, const char* what)
{
  if (v > std::numeric_limits<T>::max()) {
    throw std::runtime_error(std::string("qConfig RTTI: ") + what + " " + std::to_string(v) + " exceeds the range of the entry");
  }
  return static_cast<T>(v);
}

static inline std::optional<size_t> qConfigRTTIStructIndex(std::vector<std::string>& configStructs, const char* name, bool filter) // Index of name in configStructs, appended if not filtering, nullopt if filtered out
{
  if (auto it = std::find(configStructs.begin(), configStructs.end(), name); it != configStructs.end()) {
    return it - configStructs.begin();
  }
  if (filter) {
    return std::nullopt;
  }
  configStructs.emplace_back(name);
  return configStructs.size() - 1;
}

template <typename T, typename... Args>
static inline void qConfigRTTIEntry(std::vector<::qConfig::qConfigRTTIentry>& retVal, ::qConfig::qConfigRTTIentry entry, size_t arraySize, size_t configStructIndex, std::string defaultValue, const char* help, Args&&... args) // Complete the entry, check the narrow members for overflow
{
  auto settings = ::qConfig::qConfigSettingsParser<T>::qConfigGetSettings(args...);
  entry.size = qConfigRTTICheckedCast<decltype(entry.size)>(sizeof(T), "size");
  entry.arraySize = qConfigRTTICheckedCast<decltype(entry.arraySize)>(arraySize, "arraySize");
  entry.configStructIndex = qConfigRTTICheckedCast<decltype(entry.configStructIndex)>(configStructIndex, "configStructIndex");
  entry.defaultValue = std::move(defaultValue);
  entry.description = help;
  if (settings.checkMin) {
    entry.min = ::qConfig::print_type(settings.min);
  }
  if (settings.checkMax) {
    entry.max = ::qConfig::print_type(settings.max);
  }
  if (settings.message) {
    entry.message = settings.message;
  }
  retVal.emplace_back(std::move(entry));
}

static inline std::tuple<std::vector<::qConfig::qConfigRTTIentry>, std::vector<std::string>> qConfigGenRTTI(const std::vector<std::string>* configStructs) // Entries and list of config structs, for the given config structs (all if nullptr)
{
  using namespace ::qConfig; // For the settings helpers (min, max, ...) and print_type used in GPUSettingsList.h / qconfig_internal.h
  std::vector<qConfigRTTIentry> retVal;
  std::vector<std::string> retStructs;
  if (configStructs) {
    retStructs = *configStructs;
  }
#define QCONFIG_GEN_RTTI
#include "qconfig_internal.h"
#undef QCONFIG_GEN_RTTI
#undef QCONFIG_RTTI
  return {std::move(retVal), std::move(retStructs)};
}

#endif // QCONFIG_GENRTTI_INC
