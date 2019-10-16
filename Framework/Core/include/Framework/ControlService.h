// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CONTROLSERVICE_H_
#define O2_FRAMEWORK_CONTROLSERVICE_H_

namespace o2::framework
{

/// Kind of request we want to issue to control
enum struct QuitRequest {
  /// Only quit this data processor
  Me = 0,
  /// Quit all data processor, regardless of their state
  All = 1,
};

/// A service that data processors can use to talk to control and ask for their
/// own state change or others.
class ControlService
{
 public:
  /// Compatibility with old API.
  void readyToQuit(bool all) { this->readyToQuit(all ? QuitRequest::All : QuitRequest::Me); }
  /// Signal control that we are potentially ready to quit some / all
  /// dataprocessor.
  virtual void readyToQuit(QuitRequest kind) = 0;
  /// Signal that we are done with the current stream
  virtual void endOfStream() = 0;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_ROOTFILESERVICE_H_
