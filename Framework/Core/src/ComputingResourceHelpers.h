// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_COMPUTINGRESOURCEHELPERS_H_
#define O2_FRAMEWORK_COMPUTINGRESOURCEHELPERS_H_

#include "Framework/ComputingResource.h"

#include <string>
#include <vector>

namespace o2::framework
{
struct ComputingResourceHelpers {
  /// This will create a ComputingResource which matches what offered by localhost.
  /// Notice that the port range will always be [22000, 23000) since in any case we will
  /// use ipc:// in place of tcp://
  static ComputingResource getLocalhostResource();

  /// Parse a string which contains resources specified in the following format:
  ///
  /// <hostname>:<cpu cores>:<memory in MB>:<start port>:<last port>
  static std::vector<ComputingResource> parseResources(std::string const& resourceString);
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_COMPUTINGRESOURCEHELPERS_H_
