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

#include "Framework/CCDBParamSpec.h"
#include <fmt/format.h>
#include <algorithm>

namespace o2::framework
{

ConfigParamSpec ccdbPathSpec(std::string const& path)
{
  return ConfigParamSpec{"ccdb-path", VariantType::String, path, {fmt::format("Path in CCDB ({})", path)}, ConfigParamKind::kGeneric};
}

ConfigParamSpec ccdbRunDependent(bool defaultValue)
{
  return ConfigParamSpec{"ccdb-run-dependent", VariantType::Bool, defaultValue, {"Give object for specific run number"}, ConfigParamKind::kGeneric};
}

ConfigParamSpec ccdbMetadataSpec(std::string const& key, std::string const& defaultValue)
{
  return ConfigParamSpec{fmt::format("ccdb-metadata-{}", key),
                         VariantType::String,
                         defaultValue,
                         {fmt::format("CCDB metadata {}", key)},
                         ConfigParamKind::kGeneric};
}

std::vector<ConfigParamSpec> ccdbParamSpec(std::string const& path, std::vector<CCDBMetadata> metadata)
{
  return ccdbParamSpec(path, false, metadata);
}

std::vector<ConfigParamSpec> ccdbParamSpec(std::string const& path, bool runDependent, std::vector<CCDBMetadata> metadata)
{
  // Add here CCDB objecs which should be considered run dependent
  static std::vector<std::string> runDependentObjects = {"GLO/GRP"};
  if (std::any_of(runDependentObjects.begin(), runDependentObjects.end(), [&path](std::string const& s) { return path == s; })) {
    runDependent = true;
  }
  std::vector<ConfigParamSpec> result{ccdbPathSpec(path), ccdbRunDependent(runDependent)};
  for (auto& [key, value] : metadata) {
    result.push_back(ccdbMetadataSpec(key, value));
  }
  return result;
}

ConfigParamSpec startTimeParamSpec(int64_t t)
{
  return ConfigParamSpec{"start-value-enumeration", VariantType::Int64, t, {fmt::format("start time for enumeration ({})", t)}, ConfigParamKind::kGeneric};
}

} // namespace o2::framework
