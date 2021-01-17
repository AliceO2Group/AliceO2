// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Sandro Wenzel

#ifndef ALICEO2_DEVICES_SIMDEVICE_H_
#define ALICEO2_DEVICES_SIMDEVICE_H_

#include <memory>
#include "FairMQMessage.h"
#include <FairMQDevice.h>
#include <FairLogger.h>
#include "../macro/o2sim.C"
#include "TVirtualMC.h"
#include "TMessage.h"
#include <SimulationDataFormat/Stack.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <TRandom.h>
#include <SimConfig/SimConfig.h>
#include <cstring>

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
class O2SimDevice final : public FairMQDevice
{
 public:
  O2SimDevice() = default;
  O2SimDevice(o2::steer::O2MCApplication* vmcapp, TVirtualMC* vmc) : mVMCApp{vmcapp}, mVMC{vmc} {}

  /// Default destructor
  ~O2SimDevice() final
  {
    FairSystemInfo sysinfo;
    o2::utils::ShmManager::Instance().release();
    LOG(INFO) << "Shutting down O2SimDevice";
    LOG(INFO) << "TIME-STAMP " << mTimer.RealTime() << "\t";
    LOG(INFO) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " " << sysinfo.GetMaxMemory() << " MB\n";
  }

 protected:
  /// Overloads the InitTask() method of FairMQDevice
  void InitTask() final
  {
    // in the initialization phase we will init the simulation
    // NOTE: In a FairMQDevice this is better done here (instead of outside) since
    // we have to setup simulation + worker in the same thread (due to many threadlocal variables
    // in the simulation) ... at least as long FairMQDevice is not spawning workers on the master thread
    initSim(fChannels.at("primary-get").at(0), mSimRun);

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
  static bool querySimConfig(FairMQChannel& channel)
  {
    auto text = new std::string("configrequest");
    std::unique_ptr<FairMQMessage> request(channel.NewMessage(const_cast<char*>(text->c_str()),
                                                              text->length(), CustomCleanup, text));
    std::unique_ptr<FairMQMessage> reply(channel.NewMessage());

    int timeoutinMS = 100000; // wait for 100s max
    if (channel.Send(request, timeoutinMS) > 0) {
      LOG(INFO) << "Waiting for configuration answer ";
      if (channel.Receive(reply, timeoutinMS) > 0) {
        LOG(INFO) << "Configuration answer received, containing " << reply->GetSize() << " bytes ";

        // the answer is a TMessage containing the simulation Configuration
        auto message = std::make_unique<o2::devices::TMessageWrapper>(reply->GetData(), reply->GetSize());
        auto config = static_cast<o2::conf::SimConfigData*>(message.get()->ReadObjectAny(message.get()->GetClass()));
        if (!config) {
          return false;
        }

        LOG(INFO) << "COMMUNICATED ENGINE " << config->mMCEngine;

        auto& conf = o2::conf::SimConfig::Instance();
        conf.resetFromConfigData(*config);
        FairLogger::GetLogger()->SetLogVerbosityLevel(conf.getLogVerbosity().c_str());
        delete config;
      } else {
        LOG(ERROR) << "No configuration received within " << timeoutinMS << "ms\n";
        return false;
      }
    } else {
      LOG(ERROR) << "Could not send configuration request within " << timeoutinMS << "ms\n";
      return false;
    }
    return true;
  }

  // initializes the simulation classes; queries the configuration on a given channel
  static bool initSim(FairMQChannel& channel, std::unique_ptr<FairRunSim>& simptr)
  {
    if (!querySimConfig(channel)) {
      return false;
    }

    LOG(INFO) << "Setting up the simulation ...";
    simptr = std::move(std::unique_ptr<FairRunSim>(o2sim_init(true)));
    FairSystemInfo sysinfo;

    // to finish initialization (trigger further cross section table building etc) -- which especially
    // G4 is doing at the first ProcessRun
    // The goal is to have everything setup before we fork
    TVirtualMC::GetMC()->ProcessRun(0);

    LOG(INFO) << "MEM-STAMP END OF SIM INIT" << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
              << sysinfo.GetMaxMemory() << " MB\n";

    return true;
  }

  bool Kernel(FairMQChannel& requestchannel, FairMQChannel& dataoutchannel)
  {
    auto text = new std::string("primrequest");

    // create message object with a pointer to the data buffer,
    // its size,
    // custom deletion function (called when transfer is done),
    // and pointer to the object managing the data buffer
    FairMQMessagePtr request(requestchannel.NewMessage(const_cast<char*>(text->c_str()), // data
                                                       text->length(),                   // size
                                                       CustomCleanup,
                                                       text));
    FairMQMessagePtr reply(dataoutchannel.NewMessage());

    mVMCApp->setSimDataChannel(&dataoutchannel);

    LOG(INFO) << "Requesting work ";
    int timeoutinMS = 1000000; // wait for 1000s max -- we should have a more robust solution
                               // this should be mostly driven by the time to setup the particle generator in the server
    auto sendcode = requestchannel.Send(request, timeoutinMS);
    if (sendcode > 0) {
      LOG(INFO) << "Waiting for answer ";
      // asking for primary generation

      int trial = 0;
      int code = -1;
      do {
        code = requestchannel.Receive(reply, timeoutinMS);
        trial++;
        if (code > 0) {
          LOG(INFO) << "Answer received, containing " << reply->GetSize() << " bytes ";

          // wrap incoming bytes as a TMessageWrapper which offers "adoption" of a buffer
          auto message = new TMessageWrapper(reply->GetData(), reply->GetSize());
          auto chunk = static_cast<o2::data::PrimaryChunk*>(message->ReadObjectAny(message->GetClass()));

          mVMCApp->setPrimaries(chunk->mParticles);

          auto info = chunk->mSubEventInfo;
          mVMCApp->setSubEventInfo(&info);

          LOG(INFO) << "Processing " << chunk->mParticles.size() << " primary particles "
                    << "for event " << info.eventID << "/" << info.maxEvents << " "
                    << "part " << info.part << "/" << info.nparts;
          gRandom->SetSeed(chunk->mSubEventInfo.seed);

          // Process one event
          auto& conf = o2::conf::SimConfig::Instance();
          if (strcmp(conf.getMCEngine().c_str(), "TGeant4") == 0) {
            // this is preferred and necessary for Geant4
            // since repeated "ProcessRun" might have significant overheads
            mVMC->ProcessEvent();
          } else {
            // for Geant3 calling ProcessEvent is not enough
            // as some hooks are not called
            mVMC->ProcessRun(1);
          }

          FairSystemInfo sysinfo;
          LOG(INFO) << "TIME-STAMP " << mTimer.RealTime() << "\t";
          mTimer.Continue();
          LOG(INFO) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
                    << sysinfo.GetMaxMemory() << " MB\n";
          delete message;
          delete chunk;
        } else {
          LOG(INFO) << " No answer reveived from server. Return code " << code;
        }
      } while (code == -1 && trial <= 2);
    } else {
      LOG(INFO) << " Requesting work from server not possible. Return code " << sendcode;
      return false;
    }
    return true;
  }

 protected:
  /// Overloads the ConditionalRun() method of FairMQDevice
  bool ConditionalRun() final
  {
    return Kernel(fChannels.at("primary-get").at(0), fChannels.at("simdata").at(0));
  }

  void PostRun() final { LOG(INFO) << "Shutting down "; }

 private:
  TStopwatch mTimer;                             //!
  o2::steer::O2MCApplication* mVMCApp = nullptr; //!
  TVirtualMC* mVMC = nullptr;                    //!
  std::unique_ptr<FairRunSim> mSimRun;           //!
};

} // namespace devices
} // namespace o2

#endif
