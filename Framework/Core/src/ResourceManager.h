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
