// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_TEXTCONTROLSERVICE_H_
#define O2_FRAMEWORK_TEXTCONTROLSERVICE_H_

#include "Framework/ControlService.h"
#include <string>
#include <regex>

namespace o2::framework
{

class ServiceRegistry;
class DeviceState;

/// A service that data processors can use to talk to control and ask for
/// their own state change or others.
class TextControlService : public ControlService
{
 public:
  TextControlService(ServiceRegistry& registry, DeviceState& deviceState);
  /// Tell the control that I am ready to quit. This will be
  /// done by printing (only once)
  ///
  /// CONTROL_ACTION: READY_TO_QUIT_ME
  ///
  /// or
  ///
  /// CONTROL_ACTION: READY_TO_QUIT_ALL
  ///
  /// depending on the value of \param all.
  ///
  /// It's up to the driver to actually react on that and terminate the
  /// child.
  void readyToQuit(QuitRequest all = QuitRequest::Me) final;

  void endOfStream() final;

 private:
  bool mOnce = false;
  ServiceRegistry& mRegistry;
  DeviceState& mDeviceState;
};

bool parseControl(std::string const& s, std::smatch& match);

} // namespace o2::framework
#endif // O2_FRAMEWORK_TEXTCONTROLSERVICE_H_
