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
#ifndef O2_FRAMEWORK_CORE_DATAPROCESSORINFO_H_
#define O2_FRAMEWORK_CORE_DATAPROCESSORINFO_H_

#include "Framework/ConfigParamSpec.h"

#include <string>
#include <vector>

namespace o2
{

namespace framework
{

/// Runtime metadata about a data processor.  Used to hold information like the
/// actual executable name or the options passed to it.
struct DataProcessorInfo {
  /// Name of the associated DataProcessorSpec
  std::string name;
  /// The executable name of the program which holds the DataProcessorSpec
  std::string executable;
  /// The argument passed on the command line for this DataProcessorSpec
  std::vector<std::string> cmdLineArgs;
  /// The workflow options which are available for the associated DataProcessorSpec
  std::vector<ConfigParamSpec> workflowOptions;
  /// The channels for a given dataprocessor
  std::vector<std::string> channels;
};

} // namespace framework
} // namespace o2

#endif // O2_FRAMEWORK_CORE_DATAPROCESSORINFO_H_
