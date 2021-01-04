// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

class ImVec2;

namespace o2
{
namespace framework
{
class DeviceMetricsInfo;
class DeviceInfo;

namespace gui
{

/// View of the DataRelayer metrics for a given DeviceInfo
void displayDataRelayer(DeviceMetricsInfo const& metrics, DeviceInfo const& info, ImVec2 const& size);

} // namespace gui
} // namespace framework
} // namespace o2
