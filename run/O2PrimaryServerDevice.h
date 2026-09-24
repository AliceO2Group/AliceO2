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

#ifndef O2_DEVICES_PRIMSERVDEVICE_H_
#define O2_DEVICES_PRIMSERVDEVICE_H_

#include <fairmq/Device.h>
#include <DetectorsBase/Stack.h>
#include <Generators/PrimaryGenerator.h>
#include <SimConfig/SimConfig.h>
#include <SimulationDataFormat/DigitizationContext.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <TRandom3.h>
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include "PrimaryServerState.h"

namespace o2
{
namespace devices
{

class O2PrimaryServerDevice final : public fair::mq::Device
{
 public:
  /// constructor
  O2PrimaryServerDevice();

  /// Default destructor
  ~O2PrimaryServerDevice() final;

 protected:
  void initGenerator();

  // function generating one event
  void generateEvent(/*bool changeState = false*/);

  // launches a thread that listens for status/config/shutdown requests from outside asynchronously
  void launchInfoThread();

  void InitTask() final;

  // function for intermediate/on-the-fly reinitializations
  bool ReInit(o2::conf::SimReconfigData const& reconfig);

  // method reacting to requests to get the simulation configuration
  bool HandleConfigRequest(fair::mq::Channel& channel);

  bool ConditionalRun() override;

  void PostRun() override;

  bool HandleRequest(fair::mq::MessagePtr& request, int /*index*/, fair::mq::Channel& channel);

  void stateTransition(O2PrimaryServerState to, const char* message);

  void waitForControlInput();

 private:
  o2::conf::SimConfig mSimConfig = o2::conf::SimConfig::Instance(); // local sim config object
  o2::eventgen::PrimaryGenerator* mPrimGen = nullptr;               // the current primary generator
  o2::dataformats::MCEventHeader mEventHeader;
  o2::data::Stack* mStack = nullptr; // the stack which is filled (pointer since constructor to be called only init method)
  int mChunkGranularity = 500;       // how many primaries to send to a worker
  int mPartCounter = 0;
  bool mNeedNewEvent = true;
  int mMaxEvents = 2;
  ULong_t mInitialSeed = 0;
  bool mUseFixedChunkSeed = false;
  ULong_t mFixedChunkSeed = 0;
  int mPipeToDriver = -1; // handle for direct piper to driver (to communicate meta info)
  int mEventCounter = 0;

  std::thread mGeneratorThread; //! a thread used to concurrently init the particle generator
                                //  or to generate events
  std::thread mControlThread;   //! a thread used to wait for control commands

  // Keeps various generators instantiated in memory
  // useful when running simulation as a service (when generators
  // change between batches). Also takes care of resource management of Primary generators via unique ptr
  // TODO: some care needs to be taken (or the user warned) that the caching is based on generator name
  //       and that parameter-based reconfiguration is not yet implemented (for which we would need to hash all
  //       configuration parameters as well)
  std::map<std::string, std::unique_ptr<o2::eventgen::PrimaryGenerator>> mPrimGeneratorCache;

  std::atomic<O2PrimaryServerState> mState{O2PrimaryServerState::Initializing};
  std::atomic<int> mWaitingControlInput{0};
  std::atomic<bool> mInfoThreadStopped{false};

  bool mAsService = false;

  // a dedicate (on-the-fly channel) for control messages
  fair::mq::Channel mControlChannel;

  // some information specific to use case when we have a collision context
  o2::steer::DigitizationContext* mCollissionContext = nullptr; //!
  std::unordered_map<int, int> mEventID_to_CollID;              //!
  std::string mEmbeddIntoPrefix;                                //! sim prefix of background events

  TRandom3 mSeedGenerator; //! specific random generator for seed generation for work chunks
};

} // namespace devices
} // namespace o2

#endif
