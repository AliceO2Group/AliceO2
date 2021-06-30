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

#ifndef O2_PRIMARYSERVERSTATE_H
#define O2_PRIMARYSERVERSTATE_H

namespace o2
{

/// enum to represent state of the O2Sim event/primary server
enum class O2PrimaryServerState {
  Initializing = 0,
  ReadyToServe = 1,
  WaitingEvent = 2,
  Idle = 3,
  Stopped = 4
};
static const char* PrimStateToString[5] = {"INIT", "SERVING", "WAITEVENT", "IDLE", "STOPPED"};

/// enum class for type of info request
enum class O2PrimaryServerInfoRequest {
  Status = 1,
  Config = 2
};

/// Struct to be used as payload when making a request
/// to the primary server
struct PrimaryChunkRequest {
  int workerid = -1;
  int workerpid = -1;
  int requestid = -1;
};

/// Struct to be used as header payload when replying to
/// a worker request.
struct PrimaryChunkAnswer {
  O2PrimaryServerState serverstate;
  bool payload_attached; // whether real payload follows (or server has no work at this moment)
};

} // namespace o2

#endif //O2_PRIMARYSERVERSTATE_H
