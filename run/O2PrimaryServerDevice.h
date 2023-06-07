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
#include <fairmq/TransportFactory.h>
#include <FairPrimaryGenerator.h>
#include <Generators/GeneratorFactory.h>
#include <fairmq/Message.h>
#include <DetectorsBase/Stack.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/DigitizationContext.h>
#include <TMessage.h>
#include <TClass.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <Generators/GeneratorFromFile.h>
#include <Generators/PrimaryGenerator.h>
#include <SimConfig/SimConfig.h>
#include <CommonUtils/ConfigurableParam.h>
#include <CommonUtils/RngHelper.h>
#include <DetectorsBase/SimFieldUtils.h>
#include <Field/MagneticField.h>
#include <TGeoGlobalMagField.h>
#include <typeinfo>
#include <thread>
#include <TROOT.h>
#include <TStopwatch.h>
#include <fstream>
#include <iostream>
#include <atomic>
#include "PrimaryServerState.h"
#include "SimPublishChannelHelper.h"
#include <chrono>
#include <CCDB/BasicCCDBManager.h>

namespace o2
{
namespace devices
{

class O2PrimaryServerDevice final : public fair::mq::Device
{
 public:
  /// constructor
  O2PrimaryServerDevice()
  {
    mUseFixedChunkSeed = getenv("ALICEO2_O2SIM_SUBEVENTSEED") && atoi(getenv("ALICEO2_O2SIM_SUBEVENTSEED"));
    if (mUseFixedChunkSeed) {
      mFixedChunkSeed = atol(getenv("ALICEO2_O2SIM_SUBEVENTSEED"));
    }
  }

  /// Default destructor
  ~O2PrimaryServerDevice() final
  {
    try {
      if (mGeneratorThread.joinable()) {
        mGeneratorThread.join();
      }
      if (mControlThread.joinable()) {
        mControlThread.join();
      }
    } catch (...) {
    }
  }

 protected:
  void initGenerator()
  {
    TStopwatch timer;
    timer.Start();
    const auto& conf = mSimConfig;
    auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
    ccdbmgr.setURL(conf.getConfigData().mCCDBUrl);
    ccdbmgr.setTimestamp(conf.getTimestamp());

    // init magnetic field as it might be needed by the generator
    if (TGeoGlobalMagField::Instance()->GetField() == nullptr) {
      TGeoGlobalMagField::Instance()->SetField(o2::base::SimFieldUtils::createMagField());
      TGeoGlobalMagField::Instance()->Lock();
    }

    // look if we find a cached instances of Pythia8 or external generators in order to avoid
    // (long) initialization times.
    // This is evidently a bit weak, as generators might need reconfiguration (to be treated later).
    // For now, we'd like to allow for fast switches between say a pythia8 instance and reading from kinematics
    // to continue an already started simulation.
    //
    // Not using cached instances for external kinematics since these might change input filenames etc.
    // and are in any case quickly setup.
    mPrimGen = nullptr;
    if (conf.getGenerator().compare("extkin") != 0 || conf.getGenerator().compare("extkinO2") != 0) {
      auto iter = mPrimGeneratorCache.find(conf.getGenerator());
      if (iter != mPrimGeneratorCache.end()) {
        mPrimGen = iter->second;
        LOG(info) << "Found cached generator for " << conf.getGenerator();
      }
    }

    if (mPrimGen == nullptr) {
      mPrimGen = new o2::eventgen::PrimaryGenerator;
      o2::eventgen::GeneratorFactory::setPrimaryGenerator(conf, mPrimGen);

      // setup vertexing
      auto vtxMode = conf.getVertexMode();
      using o2::conf::VertexMode;
      if (vtxMode == VertexMode::kNoVertex || vtxMode == VertexMode::kDiamondParam) {
        mPrimGen->setVertexMode(vtxMode);
      } else if (vtxMode == VertexMode::kCCDB) {
        // we need to fetch the CCDB object
        mPrimGen->setVertexMode(vtxMode, ccdbmgr.getForTimeStamp<o2::dataformats::MeanVertexObject>("GLO/Calib/MeanVertex", conf.getTimestamp()));
      } else {
        LOG(fatal) << "Unsupported vertex mode";
      }

      auto embedinto_filename = conf.getEmbedIntoFileName();
      if (!embedinto_filename.empty()) {
        mPrimGen->embedInto(embedinto_filename);
      }

      mPrimGen->Init();

      mPrimGeneratorCache[conf.getGenerator()] = mPrimGen;
    }
    mPrimGen->SetEvent(&mEventHeader);

    // A good moment to couple to collision context
    auto collContextFileName = mSimConfig.getConfigData().mFromCollisionContext;
    if (collContextFileName.size() > 0) {
      LOG(info) << "Simulation has collission context";
      mCollissionContext = o2::steer::DigitizationContext::loadFromFile(collContextFileName);
      if (mCollissionContext) {
        const auto& vertices = mCollissionContext->getInteractionVertices();
        LOG(info) << "We found " << vertices.size() << " vertices included ";

        // initialize the eventID to collID mapping
        const auto source = mCollissionContext->findSimPrefix(mSimConfig.getOutPrefix());
        if (source == -1) {
          LOG(fatal) << "Wrong simulation prefix";
        }
        mEventID_to_CollID.clear();
        mEventID_to_CollID = mCollissionContext->getCollisionIndicesForSource(source);
      }
    }

    LOG(info) << "Generator initialization took " << timer.CpuTime() << "s";
    if (mMaxEvents > 0) {
      generateEvent(); // generate a first event
    }
  }

  // function generating one event
  void generateEvent(/*bool changeState = false*/)
  {
    bool changeState = false;
    LOG(info) << "Event generation started ";
    if (changeState) {
      stateTransition(O2PrimaryServerState::WaitingEvent, "GENEVENT");
    }
    TStopwatch timer;
    timer.Start();
    try {
      mStack->Reset();
      // see if we the vertex comes from the collision context
      if (mCollissionContext) {
        const auto& vertices = mCollissionContext->getInteractionVertices();
        if (vertices.size() > 0) {
          auto collisionindex = mEventID_to_CollID.at(mEventCounter);
          auto& vertex = vertices.at(collisionindex);
          LOG(info) << "Setting vertex " << vertex << " for event " << mEventCounter << " for prefix " << mSimConfig.getOutPrefix();
          mPrimGen->setExternalVertexForNextEvent(vertex.X(), vertex.Y(), vertex.Z());
        }
      }
      mPrimGen->GenerateEvent(mStack);
    } catch (std::exception const& e) {
      LOG(error) << " Exception occurred during event gen ";
    }
    timer.Stop();
    LOG(info) << "Event generation took " << timer.CpuTime() << "s"
              << " and produced " << mStack->getPrimaries().size() << " primaries ";
    if (changeState) {
      stateTransition(O2PrimaryServerState::ReadyToServe, "GENEVENT");
    }
  }

  // launches a thread that listens for status requests from outside asynchronously
  void launchInfoThread()
  {
    static std::vector<std::thread> threads;
    LOG(info) << "LAUNCHING STATUS THREAD";
    auto lambda = [this]() {
      while (mState != O2PrimaryServerState::Stopped) {
        auto& channel = fChannels.at("o2sim-primserv-info").at(0);
        if (!channel.IsValid()) {
          LOG(error) << "channel primserv-info not valid";
        }
        std::unique_ptr<fair::mq::Message> request(channel.NewSimpleMessage(-1));
        int timeout = 100; // 100ms --> so as not to block and allow for proper termination of this thread
        if (channel.Receive(request, timeout) > 0) {
          LOG(info) << "INFO REQUEST RECEIVED";
          if (*(int*)(request->GetData()) == (int)O2PrimaryServerInfoRequest::Status) {
            LOG(info) << "Received status request";
            // request needs to be a simple enum of type O2PrimaryServerInfoRequest
            std::unique_ptr<fair::mq::Message> reply(channel.NewSimpleMessage((int)mState.load()));
            if (channel.Send(reply) > 0) {
              LOG(info) << "Send status successful";
            }
          } else if (*(int*)request->GetData() == (int)O2PrimaryServerInfoRequest::Config) {
            HandleConfigRequest(channel);
          } else {
            LOG(fatal) << "UNKNOWN REQUEST";
            std::unique_ptr<fair::mq::Message> reply(channel.NewSimpleMessage(404));
            channel.Send(reply);
          }
        }
      }
      mInfoThreadStopped = true;
    };
    threads.push_back(std::thread(lambda));
    threads.back().detach();
  }

  void InitTask() final
  {
    // fatal without core dump
    fair::Logger::OnFatal([] { throw fair::FatalException("Fatal error occured. Exiting without core dump..."); });

    o2::simpubsub::publishMessage(fChannels["primary-notifications"].at(0), "SERVER : INITIALIZING");

    stateTransition(O2PrimaryServerState::Initializing, "INITTASK");
    LOG(info) << "Init Server device ";

    // init sim config
    auto& vm = GetConfig()->GetVarMap();
    auto& conf = o2::conf::SimConfig::Instance();
    if (vm.count("isRun5")) {
      conf.setRun5();
    }
    conf.resetFromParsedMap(vm);

    // update the parameters from an INI/JSON file, if given (overrides code-based version)
    o2::conf::ConfigurableParam::updateFromFile(conf.getConfigFile());
    // update the parameters from stuff given at command line (overrides file-based version)
    o2::conf::ConfigurableParam::updateFromString(conf.getKeyValueString());

    // customize the level of log output
    FairLogger::GetLogger()->SetLogScreenLevel(conf.getLogSeverity().c_str());
    FairLogger::GetLogger()->SetLogVerbosityLevel(conf.getLogVerbosity().c_str());

    // from now on mSimConfig should be used within this process
    mSimConfig = conf;

    mStack = new o2::data::Stack();
    mStack->setExternalMode(true);

    // MC ENGINE
    LOG(info) << "ENGINE SET TO " << vm["mcEngine"].as<std::string>();
    // CHUNK SIZE
    mChunkGranularity = vm["chunkSize"].as<unsigned int>();
    LOG(info) << "CHUNK SIZE SET TO " << mChunkGranularity;

    // initial initial seed --> we should store this somewhere
    mInitialSeed = vm["seed"].as<ULong_t>();
    mInitialSeed = o2::utils::RngHelper::setGRandomSeed(mInitialSeed);
    LOG(info) << "RNG INITIAL SEED " << mInitialSeed;

    mMaxEvents = conf.getNEvents();

    // need to make ROOT thread-safe since we use ROOT services in all places
    ROOT::EnableThreadSafety();

    launchInfoThread();

    // launch initialization of particle generator asynchronously
    // so that we reach the RUNNING state of the server quickly
    // and do not block here
    mGeneratorThread = std::thread(&O2PrimaryServerDevice::initGenerator, this);
    if (mGeneratorThread.joinable()) {
      mGeneratorThread.join();
    }

    // init pipe
    auto pipeenv = getenv("ALICE_O2SIMSERVERTODRIVER_PIPE");
    if (pipeenv) {
      mPipeToDriver = atoi(pipeenv);
      LOG(info) << "ASSIGNED PIPE HANDLE " << mPipeToDriver;
    } else {
      LOG(info) << "DID NOT FIND ENVIRONMENT VARIABLE TO INIT PIPE";
    }

    mAsService = vm["asservice"].as<bool>();

    if (mMaxEvents <= 0) {
      if (mAsService) {
        stateTransition(O2PrimaryServerState::Idle, "INITTASK");
      }
    } else {
      stateTransition(O2PrimaryServerState::ReadyToServe, "INITTASK");
    }

    // feedback to driver that we are done initializing
    if (mPipeToDriver != -1) {
      int message = -111; // special code meaning end of initialization
      if (write(mPipeToDriver, &message, sizeof(int))) {
      }
    }
  }

  // function for intermediate/on-the-fly reinitializations
  bool ReInit(o2::conf::SimReconfigData const& reconfig)
  {
    LOG(info) << "ReInit Server device ";

    if (reconfig.stop) {
      return false;
    }

    // mSimConfig.getConfigData().mKeyValueTokens=reconfig.keyValueTokens;
    // Think about this:
    // update the parameters from an INI/JSON file, if given (overrides code-based version)
    o2::conf::ConfigurableParam::updateFromFile(reconfig.configFile);
    // update the parameters from stuff given at command line (overrides file-based version)
    o2::conf::ConfigurableParam::updateFromString(reconfig.keyValueTokens);

    // initial initial seed --> we should store this somewhere
    mInitialSeed = reconfig.startSeed;
    mInitialSeed = o2::utils::RngHelper::setGRandomSeed(mInitialSeed);
    LOG(info) << "RNG INITIAL SEED " << mInitialSeed;

    mMaxEvents = reconfig.nEvents;

    // updating the simconfig member with new information especially concerning the generators
    // TODO: put this into utility function?
    mSimConfig.getConfigData().mGenerator = reconfig.generator;
    mSimConfig.getConfigData().mTrigger = reconfig.trigger;
    mSimConfig.getConfigData().mExtKinFileName = reconfig.extKinfileName;

    mEventCounter = 0;
    mPartCounter = 0;
    mNeedNewEvent = true;
    // reinit generator and start generation of a new event
    if (mGeneratorThread.joinable()) {
      mGeneratorThread.join();
    }
    mGeneratorThread = std::thread(&O2PrimaryServerDevice::initGenerator, this);
    // initGenerator();
    if (mGeneratorThread.joinable()) {
      mGeneratorThread.join();
    }

    return true;
  }

  // method reacting to requests to get the simulation configuration
  bool HandleConfigRequest(fair::mq::Channel& channel)
  {
    LOG(info) << "Received config request";
    // just sending the simulation configuration to anyone that wants it
    const auto& confdata = mSimConfig.getConfigData();

    TMessage* tmsg = new TMessage(kMESS_OBJECT);
    tmsg->WriteObjectAny((void*)&confdata, TClass::GetClass(typeid(confdata)));

    auto free_tmessage = [](void* data, void* hint) { delete static_cast<TMessage*>(hint); };

    std::unique_ptr<fair::mq::Message> message(
      fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

    // send answer
    if (channel.Send(message) > 0) {
      LOG(info) << "config reply send ";
      return true;
    }
    return true;
  }

  bool ConditionalRun() override
  {
    // we might come here in IDLE mode
    if (mState == O2PrimaryServerState::Idle) {
      if (mWaitingControlInput.load() == 0) {
        if (mControlThread.joinable()) {
          mControlThread.join();
        }
        mControlThread = std::thread(&O2PrimaryServerDevice::waitForControlInput, this);
      }
    }

    auto& channel = fChannels.at("primary-get").at(0);
    PrimaryChunkRequest requestpayload;
    std::unique_ptr<fair::mq::Message> request(channel.NewSimpleMessage(requestpayload));
    auto bytes = channel.Receive(request);
    if (bytes < 0) {
      LOG(error) << "Some error/interrupt occurred on socket during receive";
      if (NewStatePending()) { // new state is typically pending if (term) signal was received
        WaitForNextState();
        // ask ourselves for termination of this loop
        stateTransition(O2PrimaryServerState::Stopped, "CONDRUN");
      }
      return false;
    }

    TStopwatch timer;
    timer.Start();
    auto& r = *((PrimaryChunkRequest*)(request->GetData()));
    LOG(debug) << "PARTICLE REQUEST IN STATE " << PrimStateToString[(int)mState.load()] << " from " << r.workerid << ":" << r.requestid;

    auto prestate = mState.load();
    auto more = HandleRequest(request, 0, channel);
    if (!more) {
      if (mAsService) {
        if (prestate == O2PrimaryServerState::ReadyToServe || prestate == O2PrimaryServerState::WaitingEvent) {
          stateTransition(O2PrimaryServerState::Idle, "CONDRUN");
        }
      } else {
        stateTransition(O2PrimaryServerState::Stopped, "CONDRUN");
      }
    }
    timer.Stop();
    auto time = timer.CpuTime();
    LOG(debug) << "COND-RUN TOOK " << time << " s";
    return mState != O2PrimaryServerState::Stopped;
  }

  void PostRun() override
  {
    while (!mInfoThreadStopped) {
      LOG(info) << "Waiting info thread";
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(100ms);
    }
  }

  bool HandleRequest(fair::mq::MessagePtr& request, int /*index*/, fair::mq::Channel& channel)
  {
    // LOG(debug) << "GOT A REQUEST WITH SIZE " << request->GetSize();
    // std::string requeststring(static_cast<char*>(request->GetData()), request->GetSize());
    // LOG(info) << "NORMAL REQUEST STRING " << requeststring;
    bool workavailable = true;
    if (mEventCounter >= mMaxEvents && mNeedNewEvent) {
      workavailable = false;
    }
    if (!(mState == O2PrimaryServerState::ReadyToServe || mState == O2PrimaryServerState::WaitingEvent)) {
      // send a zero answer
      workavailable = false;
    }

    PrimaryChunkAnswer header{mState, workavailable};
    fair::mq::Parts reply;
    std::unique_ptr<fair::mq::Message> headermsg(channel.NewSimpleMessage(header));
    reply.AddPart(std::move(headermsg));

    LOG(debug) << "Received request for work " << mEventCounter << " " << mMaxEvents << " " << mNeedNewEvent << " available " << workavailable;
    if (mNeedNewEvent) {
      // we need a newly generated event now
      if (mGeneratorThread.joinable()) {
        try {
          mGeneratorThread.join();
        } catch (std::exception const& e) {
          LOG(warn) << "Exception during thread join ..ignoring";
        }
      }
      mNeedNewEvent = false;
      mPartCounter = 0;
      mEventCounter++;
    }

    auto& prims = mStack->getPrimaries();
    auto numberofparts = (int)std::ceil(prims.size() / (1. * mChunkGranularity));
    // number of parts should be at least 1 (even if empty)
    numberofparts = std::max(1, numberofparts);

    LOG(debug) << "Have " << prims.size() << " " << numberofparts;

    o2::data::PrimaryChunk m;
    o2::data::SubEventInfo i;
    i.eventID = workavailable ? mEventCounter : -1;
    i.maxEvents = mMaxEvents;
    i.part = mPartCounter + 1;
    i.nparts = numberofparts;

    i.seed = mUseFixedChunkSeed ? mFixedChunkSeed : mEventCounter + mInitialSeed;
    i.index = m.mParticles.size();
    i.mMCEventHeader = mEventHeader;
    m.mSubEventInfo = i;

    if (workavailable) {
      int endindex = prims.size() - mPartCounter * mChunkGranularity;
      int startindex = prims.size() - (mPartCounter + 1) * mChunkGranularity;
      LOG(debug) << "indices " << startindex << " " << endindex;

      if (startindex < 0) {
        startindex = 0;
      }
      if (endindex < 0) {
        endindex = 0;
      }

      for (int index = startindex; index < endindex; ++index) {
        m.mParticles.emplace_back(prims[index]);
      }

      LOG(info) << "Sending " << m.mParticles.size() << " particles";
      LOG(info) << "treating ev " << mEventCounter << " part " << i.part << " out of " << i.nparts;

      // feedback to driver if new event started
      if (mPipeToDriver != -1 && i.part == 1 && workavailable) {
        if (write(mPipeToDriver, &mEventCounter, sizeof(mEventCounter))) {
        }
      }

      mPartCounter++;
      if (mPartCounter == numberofparts) {
        mNeedNewEvent = true;
        // start generation of a new event
        if (mEventCounter < mMaxEvents) {
          mGeneratorThread = std::thread(&O2PrimaryServerDevice::generateEvent, this);
        }
      }

      TMessage* tmsg = new TMessage(kMESS_OBJECT);
      tmsg->WriteObjectAny((void*)&m, TClass::GetClass("o2::data::PrimaryChunk"));

      auto free_tmessage = [](void* data, void* hint) { delete static_cast<TMessage*>(hint); };

      std::unique_ptr<fair::mq::Message> message(channel.NewMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

      reply.AddPart(std::move(message));
    }

    // send answer
    TStopwatch timer;
    timer.Start();
    auto code = Send(reply, "primary-get", 0, 5000); // we introduce timeout in order not to block other requests
    timer.Stop();
    auto time = timer.CpuTime();
    if (code > 0) {
      LOG(debug) << "Reply send in " << time << "s";
      return workavailable;
    } else {
      LOG(warn) << "Sending process had problems. Return code : " << code << " time " << time << "s";
    }
    return false; // -> error should not get here
  }

  void stateTransition(O2PrimaryServerState to, const char* message)
  {
    LOG(info) << message << " CHANGING STATE TO " << PrimStateToString[(int)to];
    mState = to;
  }

  void waitForControlInput()
  {
    mWaitingControlInput.store(1);
    stateTransition(O2PrimaryServerState::Idle, "CONTROL");

    o2::simpubsub::publishMessage(fChannels["primary-notifications"].at(0), o2::simpubsub::simStatusString("PRIMSERVER", "STATUS", "AWAITING INPUT"));
    // this means we are idling

    auto factory = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
    auto channel = fair::mq::Channel{"o2sim-control", "sub", factory};
    auto controlsocketname = getenv("ALICE_O2SIMCONTROL");
    channel.Connect(std::string(controlsocketname));
    channel.Validate();
    std::unique_ptr<fair::mq::Message> reply(channel.NewMessage());

    bool ok = false;

    LOG(info) << "WAITING FOR CONTROL INPUT";
    if (channel.Receive(reply) > 0) {
      stateTransition(O2PrimaryServerState::Initializing, "CONTROL");
      auto data = reply->GetData();
      auto size = reply->GetSize();

      std::string command(reinterpret_cast<char const*>(data), size);
      LOG(info) << "message: " << command;

      o2::conf::SimReconfigData reconfig;
      o2::conf::parseSimReconfigFromString(command, reconfig);
      LOG(info) << "Processing " << reconfig.nEvents << " new events";
      try {
        LOG(info) << "REINIT START";
        ok = ReInit(reconfig);
        LOG(info) << "REINIT DONE";
      } catch (std::exception e) {
        LOG(info) << "Exception during reinit";
      }
    } else {
      LOG(info) << "NOTHING RECEIVED";
    }
    if (ok) {
      stateTransition(O2PrimaryServerState::ReadyToServe, "CONTROL");
    } else {
      stateTransition(O2PrimaryServerState::Stopped, "CONTROL");
    }
    mWaitingControlInput.store(0);
  }

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
  // change between batches)
  // TODO: some care needs to be taken (or the user warned) that the caching is based on generator name
  //       and that parameter-based reconfiguration is not yet implemented (for which we would need to hash all
  //       configuration parameters as well)
  std::map<std::string, o2::eventgen::PrimaryGenerator*> mPrimGeneratorCache;

  std::atomic<O2PrimaryServerState> mState{O2PrimaryServerState::Initializing};
  std::atomic<int> mWaitingControlInput{0};
  std::atomic<bool> mInfoThreadStopped{false};

  bool mAsService = false;

  // some information specific to use case when we have a collision context
  o2::steer::DigitizationContext* mCollissionContext = nullptr; //!
  std::unordered_map<int, int> mEventID_to_CollID;              //!
};

} // namespace devices
} // namespace o2

#endif
