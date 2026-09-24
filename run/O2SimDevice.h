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

/// @author Sandro Wenzel

#ifndef ALICEO2_DEVICES_SIMDEVICE_H_
#define ALICEO2_DEVICES_SIMDEVICE_H_

#include <memory>
#include <string>
#include <fairmq/Device.h>
#include <FairRunSim.h>
#include <TStopwatch.h>
#include "PrimaryServerState.h"

class TVirtualMC;

namespace o2::steer
{
class O2MCApplication;
}

// a helper for logging with worker index prefixed
void doLogInfo(int workerID, std::string const& message);

namespace o2
{
namespace devices
{

// device representing a simulation worker
class O2SimDevice final : public fair::mq::Device
{
 public:
  O2SimDevice() = default;
  O2SimDevice(o2::steer::O2MCApplication* vmcapp, TVirtualMC* vmc) : mVMCApp{vmcapp}, mVMC{vmc} {}

  /// Default destructor
  ~O2SimDevice() final;

 protected:
  /// Overloads the InitTask() method of fair::mq::Device
  void InitTask() final;

 public:
  void lateInit();

  // initializes the simulation classes; queries the configuration on a given channel
  static bool initSim(fair::mq::Channel& channel, std::unique_ptr<FairRunSim>& simptr);

  bool isWorkAvailable(fair::mq::Channel& statuschannel, int workerID = -1);

  bool Kernel(int workerID, fair::mq::Channel& requestchannel, fair::mq::Channel& dataoutchannel, fair::mq::Channel* statuschannel = nullptr);

 protected:
  /// Overloads the ConditionalRun() method of fair::mq::Device
  bool ConditionalRun() final;

  void PostRun() final;

 private:
  TStopwatch mTimer;                             //!
  o2::steer::O2MCApplication* mVMCApp = nullptr; //!
  TVirtualMC* mVMC = nullptr;                    //!
  std::unique_ptr<FairRunSim> mSimRun;           //!
};

} // namespace devices
} // namespace o2

#endif
