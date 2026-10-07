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

#include "O2SimDevice.h"
#include "../macro/o2sim.C"
#include "TVirtualMC.h"
#include <fairmq/Message.h>
#include <fairmq/Parts.h>
#include <fairlogger/Logger.h>
#include <DetectorsBase/Stack.h>
#include <DetectorsBase/VMCSeederService.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <TRandom.h>
#include <SimConfig/SimConfig.h>
#include <cstring>

void doLogInfo(int workerID, std::string const& message)
{
  LOG(info) << "[W" << workerID << "] " << message;
}

namespace o2
{
namespace devices
{

O2SimDevice::~O2SimDevice()
{
  FairSystemInfo sysinfo;
  o2::utils::ShmManager::Instance().release();
  LOG(info) << "Shutting down O2SimDevice";
  LOG(info) << "TIME-STAMP " << mTimer.RealTime() << "\t";
  LOG(info) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " " << sysinfo.GetMaxMemory() << " MB\n";
}

void O2SimDevice::InitTask()
{
  // in the initialization phase we will init the simulation
  // NOTE: In a fair::mq::Device this is better done here (instead of outside) since
  // we have to setup simulation + worker in the same thread (due to many threadlocal variables
  // in the simulation) ... at least as long fair::mq::Device is not spawning workers on the master thread
  initSim(GetChannels().at("o2sim-primserv-info").at(0), mSimRun);

  // set the vmc and app pointers
  mVMC = TVirtualMC::GetMC();
  mVMCApp = static_cast<o2::steer::O2MCApplication*>(TVirtualMCApplication::Instance());
  lateInit();
}

void O2SimDevice::lateInit()
{
  // late init
  mVMCApp->initLate();
}

bool O2SimDevice::initSim(fair::mq::Channel& channel, std::unique_ptr<FairRunSim>& simptr)
{
  if (!o2::querySimConfig(channel)) {
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

bool O2SimDevice::isWorkAvailable(fair::mq::Channel& statuschannel, int workerID)
{
  std::stringstream str;
  str << "[W" << workerID << "]";
  auto workerStr = str.str();

  int timeoutinMS = 2000; // wait for 2s max
  bool reprobe = true;
  while (reprobe) {
    reprobe = false;
    int i = -1;
    fair::mq::MessagePtr request(statuschannel.NewSimpleMessage((int)O2PrimaryServerInfoRequest::Status));
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

bool O2SimDevice::Kernel(int workerID, fair::mq::Channel& requestchannel, fair::mq::Channel& dataoutchannel, fair::mq::Channel* statuschannel)
{
  static int counter = 0;
  bool reproducibleSim = true;
  if (getenv("O2_DISABLE_REPRODUCIBLE_SIM")) {
    reproducibleSim = false;
  }

  // Mainly for debugging reasons, we allow to transport
  // a specific event + eventpart. This allows to reproduce and debug bugs faster, once
  // we know in which precise chunk they occur. The expected format for the environment variable
  // is "eventnum:partid".
  auto eventselection = getenv("O2SIM_RESTRICT_EVENTPART");
  int focus_on_event = -1;
  int focus_on_part = -1;
  if (eventselection) {
    auto splitString = [](const std::string& str) {
      std::pair<std::string, std::string> parts;
      size_t pos = str.find(':');
      if (pos != std::string::npos) {
        parts.first = str.substr(0, pos);
        parts.second = str.substr(pos + 1);
      }
      return parts;
    };
    auto p = splitString(eventselection);
    focus_on_event = std::atoi(p.first.c_str());
    focus_on_part = std::atoi(p.second.c_str());
  }

  fair::mq::MessagePtr request(requestchannel.NewSimpleMessage(PrimaryChunkRequest{workerID, -1, counter++})); // <-- don't need content; channel means -> give primaries
  fair::mq::Parts reply;

  mVMCApp->setSimDataChannel(&dataoutchannel);

  // we log info with workerID prepended
  auto workerStr = [workerID]() {
    std::stringstream str;
    str << "[W" << workerID << "]";
    return str.str();
  };

  doLogInfo(workerID, "Requesting work chunk");
  int timeoutinMS = 2000;
  auto sendcode = requestchannel.Send(request, timeoutinMS);
  if (sendcode > 0) {
    doLogInfo(workerID, "Waiting for answer");
    // asking for primary generation

    auto code = requestchannel.Receive(reply);
    if (code > 0) {
      doLogInfo(workerID, "Primary chunk received");
      auto rawmessage = std::move(reply.At(0));
      auto header = *(o2::PrimaryChunkAnswer*)(rawmessage->GetData());
      if (!header.payload_attached) {
        doLogInfo(workerID, "No payload; Server in stage " + std::string(PrimStateToString[(int)header.serverstate]));
        // if no payload attached we inspect the server state, to see what to do
        if (header.serverstate == O2PrimaryServerState::Initializing || header.serverstate == O2PrimaryServerState::WaitingEvent) {
          sleep(1); // back-off and retry
          return true;
        }
        // we need to decide what to do when the server is idle ---> if this happens immediately after a new batch request it means that the server might just lag a bit behind
        return false;
      } else {
        auto payload = std::move(reply.At(1));
        // wrap incoming bytes as a TMessageWrapper which offers "adoption" of a buffer
        auto message = new TMessageWrapper(payload->GetData(), payload->GetSize());
        auto chunk = static_cast<o2::data::PrimaryChunk*>(message->ReadObjectAny(message->GetClass()));

        bool goon = true;
        // no particles and eventID == -1 --> indication for no more work
        if (chunk->mParticles.size() == 0 && chunk->mSubEventInfo.eventID == -1) {
          doLogInfo(workerID, "No particles in reply : quitting kernel");
          goon = false;
        }

        if (goon) {

          auto info = chunk->mSubEventInfo;
          LOG(info) << workerStr() << " Processing " << chunk->mParticles.size() << " primary particles "
                    << "for event " << info.eventID << "/" << info.maxEvents << " "
                    << "part " << info.part << "/" << info.nparts;

          if (eventselection == nullptr || (focus_on_event == info.eventID && focus_on_part == info.part)) {
            mVMCApp->setPrimaries(chunk->mParticles);
          } else {
            // nothing to transport here
            mVMCApp->setPrimaries(std::vector<TParticle>{});
            LOG(info) << workerStr() << " This chunk will be skipped";
          }

          mVMCApp->setSubEventInfo(&info);

          if (reproducibleSim) {
            LOG(info) << workerStr() << " Setting seed for this sub-event to " << chunk->mSubEventInfo.seed;
            gRandom->SetSeed(chunk->mSubEventInfo.seed);
            o2::base::VMCSeederService::instance().setSeed();
          }

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

bool O2SimDevice::ConditionalRun()
{
  return Kernel(-1, GetChannels().at("primary-get").at(0), GetChannels().at("simdata").at(0));
}

void O2SimDevice::PostRun() { LOG(info) << "Shutting down "; }

} // namespace devices
} // namespace o2
