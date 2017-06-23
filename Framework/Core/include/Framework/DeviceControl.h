// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEVICECONTROL_H
#define FRAMEWORK_DEVICECONTROL_H

#include <map>
#include <string>

namespace o2 {
namespace framework {

// Controller state for the Device. This is useful for both GUI and
// batch operations of the system. Whenever something external to the
// device wants to modify it, it should be registered here and it will
// be acted on in the subsequent state update.
struct DeviceControl {
  bool stopped; // whether the device should start in STOP
  bool quiet; // wether we should be capturing device output.
  char logFilter[256] = {0};  // Lines in the log should match this to be displayed
  char logStartTrigger[256] = {0}; // Start printing log with the last occurence of this
  char logStopTrigger[256] = {0}; // Stop producing log with the first occurrence of this after the start
  std::map<std::string, std::string> options; // Where the GUI should store the options it wants.
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DEVICECONTROL_H
