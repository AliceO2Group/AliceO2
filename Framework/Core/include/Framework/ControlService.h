// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CONTROLSERVICE_H
#define FRAMEWORK_CONTROLSERVICE_H

namespace o2
{
namespace framework
{

// A service that data processors can use to talk to control and ask for
// their own state change or others.
class ControlService
{
 public:
  virtual void readyToQuit(bool all = false) = 0; // Tell the control that I am ready to quit
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_ROOTFILESERVICE_H
