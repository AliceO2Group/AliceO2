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
#ifndef O2_FRAMEWORK_CCDBPARAMSPEC_H_
#define O2_FRAMEWORK_CCDBPARAMSPEC_H_
#include "Framework/ConfigParamSpec.h"
#include <vector>
#include <string>

namespace o2::framework
{

struct CCDBMetadata {
  std::string key;
  std::string value;
};

ConfigParamSpec ccdbPathSpec(std::string const& path);
ConfigParamSpec ccdbRunDependent(bool defaultValue = true);

std::vector<ConfigParamSpec> ccdbParamSpec(std::string const& path, bool runDependent, std::vector<CCDBMetadata> metadata = {}, int64_t qrate = 0);
/// Helper to create an InputSpec which will read from a CCDB
/// Notice that those input specs have some convetions for their metadata:
///
/// `ccdb-path`: is the path in CCDB for the entry
/// `ccdb-run-dependent`: is a boolean flag to indicate if the entry is run dependent
/// `ccdb-metadata-<key>`: is a list of metadata to be added to the query, where key is the metadata key
std::vector<ConfigParamSpec> ccdbParamSpec(std::string const& path, std::vector<CCDBMetadata> metadata = {}, int64_t qrate = 0);
ConfigParamSpec startTimeParamSpec(int64_t t);

} // namespace o2::framework
#endif
