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
#include <fairmq/Message.h>
#include <fairmq/Device.h>
#include <fairmq/Parts.h>
#include <fairlogger/Logger.h>
#include "../macro/o2sim.C"
#include "TVirtualMC.h"
#include "TMessage.h"
#include <DetectorsBase/Stack.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <TRandom.h>
#include <SimConfig/SimConfig.h>
#include <cstring>
#include "PrimaryServerState.h"

namespace o2
{
namespace devices
{

class TMessageWrapper : public TMessage
{
 public:
  TMessageWrapper(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
  ~TMessageWrapper() override = default;
};

// device representing a simulation worker
class O2SimDevice final : public fair::mq::Device
{
 public:
  O2SimDevice() = default;
  O2SimDevice(o2::steer::O2MCApplication* vmcapp, TVirtualMC* vmc) : mVMCApp{vmcapp}, mVMC{vmc} {}

  /// Default destructor
  ~O2SimDevice() final
  {
    FairSystemInfo sysinfo;
    o2::utils::ShmManager::Instance().release();
    LOG(info) << "Shutting down O2SimDevice";
    LOG(info) << "TIME-STAMP " << mTimer.RealTime() << "\t";
    LOG(info) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " " << sysinfo.GetMaxMemory() << " MB\n";
  }

 protected:
  /// Overloads the InitTask() method of fair::mq::Device
  void InitTask() final
  {
    // in the initialization phase we will init the simulation
    // NOTE: In a fair::mq::Device this is better done here (instead of outside) since
    // we have to setup simulation + worker in the same thread (due to many threadlocal variables
    // in the simulation) ... at least as long fair::mq::Device is not spawning workers on the master thread
    initSim(fChannels.at("o2sim-primserv-info").at(0), mSimRun);

    // set the vmc and app pointers
    mVMC = TVirtualMC::GetMC();
    mVMCApp = static_cast<o2::steer::O2MCApplication*>(TVirtualMCApplication::Instance());
    lateInit();
  }

  static void CustomCleanup(void* data, void* hint) { delete static_cast<std::string*>(hint); }

 public:
  void lateInit()
  {
    // late init
    mVMCApp->initLate();
  }

  // should go into a helper
  // this function queries the sim config data and initializes the SimConfig singleton
  // returns true if successful / false if not
  static bool querySimConfig(fair::mq::Channel& channel)
  {
    // auto text = new std::string("configrequest");
    // std::unique_ptr<fair::mq::Message> request(channel.NewMessage(const_cast<char*>(text->c_str()),
    //                                                           text->length(), CustomCleanup, text));
    std::unique_ptr<fair::mq::Message> request(channel.NewSimpleMessage(O2PrimaryServerInfoRequest::Config));
    std::unique_ptr<fair::mq::Message> reply(channel.NewMessage());

    int timeoutinMS = 60000; // wait for 60s max --> should be fast reply
    if (channel.Send(request, timeoutinMS) > 0) {
      LOG(info) << "Waiting for configuration answer ";
      if (channel.Receive(reply, timeoutinMS) > 0) {
        LOG(info) << "Configuration answer received, containing " << reply->GetSize() << " bytes ";

        // the answer is a TMessage containing the simulation Configuration
        auto message = std::make_unique<o2::devices::TMessageWrapper>(reply->GetData(), reply->GetSize());
        auto config = static_cast<o2::conf::SimConfigData*>(message.get()->ReadObjectAny(message.get()->GetClass()));
        if (!config) {
          return false;
        }

        LOG(info) << "COMMUNICATED ENGINE " << config->mMCEngine;

        auto& conf = o2::conf::SimConfig::Instance();
        conf.resetFromConfigData(*config);
        FairLogger::GetLogger()->SetLogVerbosityLevel(conf.getLogVerbosity().c_str());
        delete config;
      } else {
        LOG(error) << "No configuration received within " << timeoutinMS << "ms\n";
        return false;
      }
    } else {
      LOG(error) << "Could not send configuration request within " << timeoutinMS << "ms\n";
      return false;
    }
    return true;
  }

  // initializes the simulation classes; queries the configuration on a given channel
  static bool initSim(fair::mq::Channel& channel, std::unique_ptr<FairRunSim>& simptr)
  {
    if (!querySimConfig(channel)) {
      return false;
    }

    LOG(info) << "Setting up the simulation ...";
    simptr = std::move(std::unique_ptr<FairRunSim>(o2sim_init(true)));
    FairSystemInfo sysinfo;

    // to finish initialization (trigger further cross section table building etc) -- which especially
    // G4 is doing at the first ProcessRun
    // The goal is to have everything setup before we fork
    TVirtualMC::GetMC()->ProcessRun(0);

    LOG(info) << "MEM-STAMP END OF SIM INIT" << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
              << sysinfo.GetMaxMemory() << " MB\n";

    return true;
  }

  bool isWorkAvailable(fair::mq::Channel& statuschannel, int workerID = -1)
  {
    std::stringstream str;
    str << "[W" << workerID << "]";
    auto workerStr = str.str();

    int timeoutinMS = 2000; // wait for 2s max
    bool reprobe = true;
    while (reprobe) {
      reprobe = false;
      int i = -1;
      fair::mq::MessagePtr request(statuschannel.NewSimpleMessage(O2PrimaryServerInfoRequest::Status));
      fair::mq::MessagePtr reply(statuschannel.NewSimpleMessage(i));
      auto sendcode = statuschannel.Send(request, timeoutinMS);
      if (sendcode > 0) {
        LOG(info) << workerStr << " Waiting for status answer ";
        auto code = statuschannel.Receive(reply, timeoutinMS);
        if (code > 0) {
          int state(*((int*)(reply->GetData())));
          if (state == (int)o2::O2PrimaryServerState::ReadyToServe) {
            LOG(info) << workerStr << " SERVER IS SERVING";
            return true;
          } else if (state == (int)o2::O2PrimaryServerState::Initializing) {
            LOG(info) << workerStr << " SERVER IS STILL INITIALIZING";
            reprobe = true;
            sleep(1);
          } else if (state == (int)o2::O2PrimaryServerState::WaitingEvent) {
            LOG(info) << workerStr << " SERVER IS WAITING FOR EVENT";
            reprobe = true;
            sleep(1);
          } else if (state == (int)o2::O2PrimaryServerState::Idle) {
            LOG(info) << workerStr << " SERVER IS IDLE";
            return false;
          } else {
            LOG(info) << workerStr << " SERVER STATE UNKNOWN OR STOPPED";
          }
        } else {
          LOG(error) << workerStr << " STATUS REQUEST UNSUCCESSFUL";
        }
      }
    }
    return false;
  }

  bool Kernel(int workerID, fair::mq::Channel& requestchannel, fair::mq::Channel& dataoutchannel, fair::mq::Channel* statuschannel = nullptr)
  {
    static int counter = 0;

    fair::mq::MessagePtr request(requestchannel.NewSimpleMessage(PrimaryChunkRequest{workerID, -1, counter++})); // <-- don't need content; channel means -> give primaries
    fair::mq::Parts reply;

    mVMCApp->setSimDataChannel(&dataoutchannel);

    // we log info with workerID prepended
    auto workerStr = [workerID]() {
      std::stringstream str;
      str << "[W" << workerID << "]";
      return str.str();
    };

    LOG(info) << workerStr() << " Requesting work chunk";
    int timeoutinMS = 2000;
    auto sendcode = requestchannel.Send(request, timeoutinMS);
    if (sendcode > 0) {
      LOG(info) << workerStr() << " Waiting for answer";
      // asking for primary generation

      auto code = requestchannel.Receive(reply);
      if (code > 0) {
        LOG(info) << workerStr() << " Primary chunk received";
        auto rawmessage = std::move(reply.At(0));
        auto header = *(o2::PrimaryChunkAnswer*)(rawmessage->GetData());
        if (!header.payload_attached) {
          LOG(info) << "No payload; Server in state " << PrimStateToString[(int)header.serverstate];
          // if no payload attached we inspect the server state, to see what to do
          if (header.serverstate == O2PrimaryServerState::Initializing || header.serverstate == O2PrimaryServerState::WaitingEvent) {
            sleep(1); // back-off and retry
            return true;
          }
          return false;
        } else {
          auto payload = std::move(reply.At(1));
          // wrap incoming bytes as a TMessageWrapper which offers "adoption" of a buffer
          auto message = new TMessageWrapper(payload->GetData(), payload->GetSize());
          auto chunk = static_cast<o2::data::PrimaryChunk*>(message->ReadObjectAny(message->GetClass()));

          bool goon = true;
          // no particles and eventID == -1 --> indication for no more work
          if (chunk->mParticles.size() == 0 && chunk->mSubEventInfo.eventID == -1) {
            LOG(info) << workerStr() << " No particles in reply : quitting kernel";
            goon = false;
          }

          if (goon) {
            mVMCApp->setPrimaries(chunk->mParticles);

            auto info = chunk->mSubEventInfo;
            mVMCApp->setSubEventInfo(&info);

            LOG(info) << workerStr() << " Processing " << chunk->mParticles.size() << " primary particles "
                      << "for event " << info.eventID << "/" << info.maxEvents << " "
                      << "part " << info.part << "/" << info.nparts;
            LOG(info) << "Setting seed for this sub-event to " << chunk->mSubEventInfo.seed;
            gRandom->SetSeed(chunk->mSubEventInfo.seed);

            // Process one event
            auto& conf = o2::conf::SimConfig::Instance();
            if (strcmp(conf.getMCEngine().c_str(), "TGeant4") == 0 || strcmp(conf.getMCEngine().c_str(), "O2TrivialMCEngine") == 0) {
              // this is preferred and necessary for Geant4
              // since repeated "ProcessRun" might have significant overheads
              mVMC->ProcessEvent();
            } else {
              // for Geant3 calling ProcessEvent is not enough
              // as some hooks are not called
              mVMC->ProcessRun(1);
            }

            FairSystemInfo sysinfo;
            LOG(info) << workerStr() << " TIME-STAMP " << mTimer.RealTime() << "\t";
            mTimer.Continue();
            LOG(info) << workerStr() << " MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
                      << sysinfo.GetMaxMemory() << " MB\n";
          }
          delete message;
          delete chunk;
        }
      } else {
        LOG(info) << workerStr() << " No primary answer received from server (within timeout). Return code " << code;
      }
    } else {
      LOG(info) << workerStr() << " Requesting work from server not possible. Return code " << sendcode;
      return false;
    }
    return true;
  }

 protected:
  /// Overloads the ConditionalRun() method of fair::mq::Device
  bool ConditionalRun() final
  {
    return Kernel(-1, fChannels.at("primary-get").at(0), fChannels.at("simdata").at(0));
  }

  void PostRun() final { LOG(info) << "Shutting down "; }

 private:
  TStopwatch mTimer;                             //!
  o2::steer::O2MCApplication* mVMCApp = nullptr; //!
  TVirtualMC* mVMC = nullptr;                    //!
  std::unique_ptr<FairRunSim> mSimRun;           //!
};

} // namespace devices
} // namespace o2

#endif
