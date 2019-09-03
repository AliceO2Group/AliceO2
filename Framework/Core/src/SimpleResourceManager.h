// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_SIMPLERESOURCEMANAGER_H
#define FRAMEWORK_SIMPLERESOURCEMANAGER_H

#include "ResourceManager.h"

namespace o2
{
namespace framework
{

/// A resource manager with infinite resources at its disposal.
/// This is a trivial implementation which can be used to do
/// laptop deploys.
class SimpleResourceManager : public ResourceManager
{
 public:
  /// @a initialPort is the first port which can be used
  ///              by this trivial resource manager.
  /// @a maxPorts is the maximum number of ports starting from
  ///             initialPort that this resource manager can allocate.
  SimpleResourceManager(unsigned short initialPort, unsigned short maxPorts = 1000)
    : mInitialPort{initialPort},
      mMaxPorts{maxPorts}
  {
  }
  std::vector<ComputingResource> getAvailableResources() override;

 private:
  int mInitialPort;
  int mMaxPorts;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_SIMPLERESOURCEMANAGER_H
