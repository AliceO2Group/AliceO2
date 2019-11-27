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

#ifndef O2_DEVICES_PRIMSERVDEVICE_H_
#define O2_DEVICES_PRIMSERVDEVICE_H_

#include <FairMQDevice.h>
#include <FairPrimaryGenerator.h>
#include <Generators/GeneratorFactory.h>
#include <FairMQMessage.h>
#include <SimulationDataFormat/Stack.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <TMessage.h>
#include <TClass.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <Generators/GeneratorFromFile.h>
#include <Generators/PrimaryGenerator.h>
#include <SimConfig/SimConfig.h>
#include <SimConfig/ConfigurableParam.h>
#include <CommonUtils/RngHelper.h>
#include <typeinfo>
#include <thread>
#include <TROOT.h>
#include <TStopwatch.h>

namespace o2
{
namespace devices
{

class O2PrimaryServerDevice : public FairMQDevice
{
 public:
  /// Default constructor
  O2PrimaryServerDevice()
  {
    mStack.setExternalMode(true);
  }

  /// Default destructor
  ~O2PrimaryServerDevice() final
  {
    if (mGeneratorThread.joinable()) {
      mGeneratorThread.join();
    }
  }

 protected:
  void initGenerator()
  {
    TStopwatch timer;
    timer.Start();
    auto& conf = o2::conf::SimConfig::Instance();
    o2::conf::ConfigurableParam::updateFromString(conf.getKeyValueString());
    o2::eventgen::GeneratorFactory::setPrimaryGenerator(conf, &mPrimGen);
    mPrimGen.SetEvent(&mEventHeader);

    auto embedinto_filename = conf.getEmbedIntoFileName();
    if (!embedinto_filename.empty()) {
      mPrimGen.embedInto(embedinto_filename);
    }
    mPrimGen.Init();
    LOG(INFO) << "Generator initialization took " << timer.CpuTime() << "s";
    generateEvent(); // generate a first event
  }

  // function generating one event
  void generateEvent()
  {
    TStopwatch timer;
    timer.Start();
    mStack.Reset();
    mPrimGen.GenerateEvent(&mStack);
    timer.Stop();
    LOG(INFO) << "Event generation took " << timer.CpuTime() << "s";
  }

  void InitTask() final
  {
    LOG(INFO) << "Init Server device ";

    // init sim config
    auto& conf = o2::conf::SimConfig::Instance();
    auto& vm = GetConfig()->GetVarMap();
    conf.resetFromParsedMap(vm);
    // output varmap
    for (auto& keyvalue : vm) {
      LOG(INFO) << "///// " << keyvalue.first << " " << keyvalue.second.value().type().name();
    }
    // MC ENGINE
    LOG(INFO) << "ENGINE SET TO " << vm["mcEngine"].as<std::string>();
    // CHUNK SIZE
    mChunkGranularity = vm["chunkSize"].as<unsigned int>();
    LOG(INFO) << "CHUNK SIZE SET TO " << mChunkGranularity;

    // initial initial seed --> we should store this somewhere
    mInitialSeed = vm["seed"].as<int>();
    mInitialSeed = o2::utils::RngHelper::setGRandomSeed(mInitialSeed);
    LOG(INFO) << "RNG INITIAL SEED " << mInitialSeed;

    mMaxEvents = conf.getNEvents();

    // need to make ROOT thread-safe since we use ROOT services in all places
    ROOT::EnableThreadSafety();

    // lunch initialization of particle generator asynchronously
    // so that we reach the RUNNING state of the server quickly
    // and do not block here
    mGeneratorThread = std::thread(&O2PrimaryServerDevice::initGenerator, this);

    // init pipe
    auto pipeenv = getenv("ALICE_O2SIMSERVERTODRIVER_PIPE");
    if (pipeenv) {
      mPipeToDriver = atoi(pipeenv);
      LOG(INFO) << "ASSIGNED PIPE HANDLE " << mPipeToDriver;
    } else {
      LOG(INFO) << "DID NOT FIND ENVIRONMENT VARIABLE TO INIT PIPE";
    }
  }

  // method reacting to requests to get the simulation configuration
  bool HandleConfigRequest(FairMQMessagePtr& request)
  {
    LOG(INFO) << "received config request";
    // just sending the simulation configuration to anyone that wants it
    auto& conf = o2::conf::SimConfig::Instance();
    const auto& confdata = conf.getConfigData();

    TMessage* tmsg = new TMessage(kMESS_OBJECT);
    tmsg->WriteObjectAny((void*)&confdata, TClass::GetClass(typeid(confdata)));

    auto free_tmessage = [](void* data, void* hint) { delete static_cast<TMessage*>(hint); };

    std::unique_ptr<FairMQMessage> message(
      fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

    // send answer
    if (Send(message, "primary-get", 0) > 0) {
      LOG(INFO) << "config reply send ";
      return true;
    }
    return true;
  }

  bool ConditionalRun() override
  {
    auto& channel = fChannels.at("primary-get").at(0);
    std::unique_ptr<FairMQMessage> request(channel.NewMessage());
    auto bytes = channel.Receive(request);
    if (bytes < 0) {
      LOG(ERROR) << "Some error occurred on socket during receive";
      return true; // keep going
    }
    return HandleRequest(request, 0);
  }

  /// Overloads the ConditionalRun() method of FairMQDevice
  bool HandleRequest(FairMQMessagePtr& request, int /*index*/)
  {
    LOG(INFO) << "GOT A REQUEST WITH SIZE " << request->GetSize();
    std::string requeststring(static_cast<char*>(request->GetData()), request->GetSize());

    if (requeststring.compare("configrequest") == 0) {
      return HandleConfigRequest(request);
    }

    else if (requeststring.compare("primrequest") != 0) {
      LOG(INFO) << "unknown request\n";
      return true;
    }

    static int counter = 0;
    if (counter >= mMaxEvents && mNeedNewEvent) {
      return false;
    }

    LOG(INFO) << "Received request for work ";
    if (mNeedNewEvent) {
      // we need a newly generated event now
      if (mGeneratorThread.joinable()) {
        mGeneratorThread.join();
      }
      mNeedNewEvent = false;
      mPartCounter = 0;
      counter++;
    }

    auto& prims = mStack.getPrimaries();
    auto numberofparts = (int)std::ceil(prims.size() / (1. * mChunkGranularity));
    // number of parts should be at least 1 (even if empty)
    numberofparts = std::max(1, numberofparts);

    o2::data::PrimaryChunk m;
    o2::data::SubEventInfo i;
    i.eventID = counter;
    i.maxEvents = mMaxEvents;
    i.part = mPartCounter + 1;
    i.nparts = numberofparts;
    i.seed = counter + mInitialSeed;
    i.index = m.mParticles.size();
    i.mMCEventHeader = mEventHeader;
    m.mSubEventInfo = i;

    int endindex = prims.size() - mPartCounter * mChunkGranularity;
    int startindex = prims.size() - (mPartCounter + 1) * mChunkGranularity;
    if (startindex < 0) {
      startindex = 0;
    }
    if (endindex < 0) {
      endindex = 0;
    }

    for (int index = startindex; index < endindex; ++index) {
      m.mParticles.emplace_back(prims[index]);
    }

    LOG(INFO) << "Sending " << m.mParticles.size() << " particles";
    LOG(INFO) << "treating ev " << counter << " part " << i.part << " out of " << i.nparts;

    // feedback to driver if new event started
    if (mPipeToDriver != -1 && i.part == 1) {
      if (write(mPipeToDriver, &counter, sizeof(counter))) {
      }
    }

    mPartCounter++;
    if (mPartCounter == numberofparts) {
      mNeedNewEvent = true;
      // start generation of a new event
      mGeneratorThread = std::thread(&O2PrimaryServerDevice::generateEvent, this);
    }

    TMessage* tmsg = new TMessage(kMESS_OBJECT);
    tmsg->WriteObjectAny((void*)&m, TClass::GetClass("o2::data::PrimaryChunk"));

    auto free_tmessage = [](void* data, void* hint) { delete static_cast<TMessage*>(hint); };

    std::unique_ptr<FairMQMessage> message(
      fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

    // send answer
    TStopwatch timer;
    timer.Start();
    auto code = Send(message, "primary-get", 0, 5000); // we introduce timeout in order not to block other requests
    timer.Stop();
    auto time = timer.CpuTime();
    if (code > 0) {
      LOG(INFO) << "Reply send in " << time << "s";
      return true;
    } else {
      LOG(WARN) << "Sending process had problems. Return code : " << code << " time " << time << "s";
    }
    return true;
  }

 private:
  std::string mOutChannelName = "";
  o2::eventgen::PrimaryGenerator mPrimGen;
  o2::dataformats::MCEventHeader mEventHeader;
  o2::data::Stack mStack;      // the stack which is filled
  int mChunkGranularity = 500; // how many primaries to send to a worker
  int mLastPosition = 0;       // last position in stack vector
  int mPartCounter = 0;
  bool mNeedNewEvent = true;
  int mMaxEvents = 2;
  int mInitialSeed = -1;
  int mPipeToDriver = -1; // handle for direct piper to driver (to communicate meta info)

  std::thread mGeneratorThread; //! a thread used to concurrently init the particle generator
                                //  or to generate events
};

} // namespace devices
} // namespace o2

#endif
