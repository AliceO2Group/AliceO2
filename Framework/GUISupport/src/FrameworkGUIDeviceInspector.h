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

/// Helper to display information about a device

namespace o2
{
namespace framework
{

struct DeviceSpec;
struct DeviceControl;
struct DeviceInfo;
struct DeviceMetricsInfo;
struct DataProcessorInfo;

namespace gui
{

void displayDeviceInspector(DeviceSpec const& spec, DeviceInfo const& info, DeviceMetricsInfo const& metrics, DataProcessorInfo const& metadata, DeviceControl& control);

} // namespace gui
} // namespace framework
} // namespace o2
