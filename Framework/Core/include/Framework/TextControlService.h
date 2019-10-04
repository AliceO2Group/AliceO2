// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_TEXTCONTROLSERVICE_H
#define FRAMEWORK_TEXTCONTROLSERVICE_H

#include "Framework/ControlService.h"
#include <string>
#include <regex>

namespace o2
{
namespace framework
{

/// A service that data processors can use to talk to control and ask for
/// their own state change or others.
class TextControlService : public ControlService
{
 public:
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
  void readyToQuit(bool all = false) final;

 private:
  bool mOnce = false;
};

bool parseControl(std::string const& s, std::smatch& match);

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TEXTCONTROLSERVICE_H
