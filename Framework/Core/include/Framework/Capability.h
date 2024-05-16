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

#ifndef O2_FRAMEWORK_CAPABILITY_H_
#define O2_FRAMEWORK_CAPABILITY_H_

#include <string_view>
#include <functional>

namespace o2::framework
{

struct ConfigParamRegistry;

/// A plugin which is able to discover more option then the ones
/// provided on the command line. The actual loading is in two parts,
/// the first one decides if the options are enough to actually perform
/// the discovery, the second part will do the discovery itself.
///
/// Its a good idea to have the Loader part in a standalone library to
/// minimize dependency on unneed thir party code, e.g. ROOT.
struct Capability {
  // Wether or not this capability is required.
  // FIXME: for now let's pass the arguments to get the metadata to work
  std::string name;
  std::function<bool(ConfigParamRegistry&, int argc, char** argv)> checkIfNeeded;
  char const* requiredPlugin;
};

struct CapabilityPlugin {
  virtual Capability* create() = 0;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CAPABILITY_H_
