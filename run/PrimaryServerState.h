// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_PRIMARYSERVERSTATE_H
#define O2_PRIMARYSERVERSTATE_H

namespace o2
{

/// enum to represent state of the O2Sim event/primary server
enum class O2PrimaryServerState {
  Initializing = 2,
  ReadyToServe = 3,
  WaitingEvent = 4,
  Idle = 5,
  Stopped = 6
};

} // namespace o2

#endif //O2_PRIMARYSERVERSTATE_H
