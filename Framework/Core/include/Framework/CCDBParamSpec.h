// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ConfigParamSpec.h"
#include <fmt/format.h>

namespace o2::framework
{

ConfigParamSpec ccdbParamSpec(std::string const& path)
{
  return ConfigParamSpec{"ccdb-path", VariantType::String, path, {fmt::format("Path in CCDB ({})", path)}, ConfigParamKind::kGeneric};
}

ConfigParamSpec startTimeParamSpec(int64_t t)
{
  return ConfigParamSpec{"start-value-enumeration", VariantType::Int64, t, {fmt::format("start time for enumeration ({})", t)}, ConfigParamKind::kGeneric};
}

} // namespace o2::framework
