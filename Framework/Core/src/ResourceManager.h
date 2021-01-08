// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_RESOURCEMANAGER_H_
#define O2_FRAMEWORK_RESOURCEMANAGER_H_

#include "Framework/ComputingResource.h"
#include <vector>

namespace o2::framework
{

class ResourceManager
{
 public:
  virtual std::vector<ComputingOffer> getAvailableOffers() = 0;
  virtual void notifyAcceptedOffer(ComputingOffer const&) = 0;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_RESOURCEMANAGER_H_
