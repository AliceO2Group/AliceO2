// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_RESOURCEMANAGER_H
#define FRAMEWORK_RESOURCEMANAGER_H

#include "ComputingResource.h"
#include <vector>

namespace o2
{
namespace framework
{

class ResourceManager
{
 public:
  virtual std::vector<ComputingResource> getAvailableResources() = 0;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_RESOURCEMANAGER_H
