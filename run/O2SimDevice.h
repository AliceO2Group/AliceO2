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
class O2SimDevice : public FairMQDevice
{
 public:
  /// Default constructor
  O2SimDevice() = default;

  /// Default destructor
  ~O2SimDevice() final
  {
    FairSystemInfo sysinfo;
    LOG(INFO) << "TIME-STAMP " << mTimer.RealTime() << "\t";
    mTimer.Continue();
    LOG(INFO) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " " << sysinfo.GetMaxMemory() << " MB\n";
  }

 protected:
  /// Overloads the InitTask() method of FairMQDevice
  void InitTask() final
  {
    // nothing to do here (unless we could already send a message)
  }

  bool initializeWorker()
  {
    static int counter = 0;
    // try at most 3 times
    counter++;
    if (counter >= 3) {
      return false;
    }

    auto text = new std::string("configrequest");

    // create message object with a pointer to the data buffer,
    // its size,
    // custom deletion function (called when transfer is done),
    // and pointer to the object managing the data buffer
    FairMQMessagePtr request(NewMessage(const_cast<char*>(text->c_str()),                                               // data
                                        text->length(),                                                                 // size
                                        [](void* /*data*/, void* object) { delete static_cast<std::string*>(object); }, // deletion callback
                                        text));                                                                         // object that manages the data

    // FairMQMessagePtr request(NewSimpleMessage(&c));
    FairMQMessagePtr reply(NewMessage());

    // ask for the configuration object
    // LOG(INFO) << "Asking for configuration with message " << c;
    int timeoutinMS = 100000; // wait for 100s max
    if (Send(request, "primary-get", 0, timeoutinMS) >= 0) {
      LOG(INFO) << "Waiting for configuration answer " << FairLogger::endl;
      if (Receive(reply, "primary-get", 0, timeoutinMS) > 0) {
        LOG(INFO) << "Configuration answer received, containing " << reply->GetSize() << " bytes " << FairLogger::endl;

        // the answer is a TMessage containing the Simulation Configuration
        auto message = std::make_unique<TMessageWrapper>(reply->GetData(), reply->GetSize());
        auto config = static_cast<o2::conf::SimConfigData*>(message.get()->ReadObjectAny(message.get()->GetClass()));

        if (config) {
          LOG(INFO) << "COMMUNICATED ENGINE " << config->mMCEngine;
        }

        auto& conf = o2::conf::SimConfig::Instance();
        conf.resetFromConfigData(*config);
      } else {
        LOG(WARNING) << "No configuration received within " << timeoutinMS << "ms\n";
        return true;
      }
    } else {
      LOG(WARNING) << "Could not send configuration request within " << timeoutinMS << "ms\n";
      return true;
    }

    LOG(INFO) << "Init SIM device " << FairLogger::endl;
    o2sim_init(true);
    FairSystemInfo sysinfo;
    LOG(INFO) << "TIME-STAMP " << mTimer.RealTime() << "\t";
    mTimer.Continue();
    LOG(INFO) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
              << sysinfo.GetMaxMemory() << " MB\n";
    mIsInitialized = true;
    return true;
  }

  /// Overloads the ConditionalRun() method of FairMQDevice
  bool ConditionalRun() final
  {
    // we contact the server a first time to retrieve the configuration
    // and to setup the simulation engine
    if (!mIsInitialized) {
      return initializeWorker();
    }

    auto text = new std::string("primrequest");

    // create message object with a pointer to the data buffer,
    // its size,
    // custom deletion function (called when transfer is done),
    // and pointer to the object managing the data buffer
    FairMQMessagePtr request(NewMessage(const_cast<char*>(text->c_str()), // data
                                        text->length(),                   // size
                                        [](void* /*data*/, void* object) { delete static_cast<std::string*>(object); },
                                        text));
    FairMQMessagePtr reply(NewMessage());

    // we get the O2MCApplication and inject the primaries before each run
    auto app = static_cast<o2::steer::O2MCApplication*>(TVirtualMCApplication::Instance());
    app->setSimDataChannel(&fChannels.at("simdata").at(0));

    LOG(INFO) << "Trying to Send  " << FairLogger::endl;
    int timeoutinMS = 2000; // wait for 2s max
    if (Send(request, "primary-get", 0, timeoutinMS) >= 0) {
      LOG(INFO) << "Waiting for answer " << FairLogger::endl;
      // asking for primary generation

      if (Receive(reply, "primary-get", 0, timeoutinMS) > 0) {
        LOG(INFO) << "Answer received, containing " << reply->GetSize() << " bytes " << FairLogger::endl;

        // wrap incoming bytes as a TMessageWrapper which offers "adoption" of a buffer
        auto message = new TMessageWrapper(reply->GetData(), reply->GetSize());
        LOG(INFO) << "message class " << message->GetClass() << FairLogger::endl;
        auto chunk = static_cast<o2::Data::PrimaryChunk*>(message->ReadObjectAny(message->GetClass()));

        app->setPrimaries(chunk->mParticles);

        auto info = chunk->mEventIDs[0];
        app->setSubEventInfo(info);

        LOG(INFO) << "Processing " << chunk->mParticles.size() << FairLogger::endl;
        gRandom->SetSeed(chunk->mEventIDs[0].seed);

        TVirtualMC::GetMC()->ProcessRun(1);
        FairSystemInfo sysinfo;
        LOG(INFO) << "TIME-STAMP " << mTimer.RealTime() << "\t";
        mTimer.Continue();
        LOG(INFO) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
                  << sysinfo.GetMaxMemory() << " MB\n";
        delete message;
      } else {
        return false;
      }
    } else {
      return false;
    }
    // need to solve the problem of writing
    return true;
  }

  void PostRun() final { LOG(INFO) << "Shutting down " << FairLogger::endl; }
 private:
  std::string mInChannelName = "";
  std::string mOutChannelName = "";
  TStopwatch mTimer;
  bool mIsInitialized = false;
};

} // namespace devices
} // namespace o2

#endif
