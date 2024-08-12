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
//
#ifndef O2_FRAMEWORK_DEFAULTHELPERS_H_
#define O2_FRAMEWORK_DEFAULTHELPERS_H_

namespace o2::framework
{
enum struct DeploymentMode;

struct DefaultsHelpers {
  static DeploymentMode deploymentMode();
  /// @true if running online
  static bool onlineDeploymentMode();
  /// get max number of timeslices in the queue
  static unsigned int pipelineLength();
};
} // namespace o2::framework

#endif
