// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_SIMPLERESOURCEMANAGER_H_
#define O2_FRAMEWORK_SIMPLERESOURCEMANAGER_H_

#include "ResourceManager.h"

namespace o2::framework
{

/// A resource manager with infinite resources at its disposal.
/// This is a trivial implementation which can be used to do
/// laptop deploys.
class SimpleResourceManager : public ResourceManager
{
 public:
  /// @a initialResources the precomputed list of available resources
  SimpleResourceManager(std::vector<ComputingResource> intialResources)
    : mResources{intialResources}
  {
  }
  /// Get the available resources for a device to run on
  std::vector<ComputingOffer> getAvailableOffers() override;

  /// Notify that we have accepted a given resource and that it
  /// should not be reoffered
  void notifyAcceptedOffer(ComputingOffer const& accepted) override;

 private:
  std::vector<ComputingResource> mResources;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SIMPLERESOURCEMANAGER_H_
