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

#ifndef O2_FRAMEWORK_DRIVERCONFIG_H_
#define O2_FRAMEWORK_DRIVERCONFIG_H_

namespace o2::framework
{
// Configuation for the driver, also available to
// the children.
struct DriverConfig {
  /// Whether the driver was started in batch mode or not.
  bool batch = true;
  /// Whether or not the driver has an active GUI we need to
  /// feed with data.
  bool driverHasGUI = false;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_DRIVERCONFIG_H_
