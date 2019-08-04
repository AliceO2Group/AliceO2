// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CONFIG_CONTEXT_H
#define FRAMEWORK_CONFIG_CONTEXT_H

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ServiceRegistry.h"

namespace o2
{
namespace framework
{

/// This is the context class for information which are available at
/// (re)configuration of the topology. It's automatically filled by the data
/// processing layer and passed to the user `defineDataProcessing` function.
class ConfigContext
{
 public:
  ConfigContext(ConfigParamRegistry& options) : mOptions{options} {}

  ConfigParamRegistry& options() const { return mOptions; }

 private:
  ConfigParamRegistry& mOptions;
};

} // namespace framework
} // namespace o2

#endif
