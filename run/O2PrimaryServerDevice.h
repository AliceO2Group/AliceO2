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
#include <FairMCEventHeader.h>
#include <TMessage.h>
#include <TClass.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <Generators/GeneratorFromFile.h>
#include <SimConfig/SimConfig.h>
#include <typeinfo>
#include <thread>
#include <TROOT.h>
#include <fcntl.h>

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
    OnData("primary-get", &O2PrimaryServerDevice::HandleRequest);
  }

  /// Default destructor
  ~O2PrimaryServerDevice() final = default;

 protected:
  void initGenerator()
  {
    auto& conf = o2::conf::SimConfig::Instance();
    o2::eventgen::GeneratorFactory::setPrimaryGenerator(conf, &mPrimGen);
    mPrimGen.SetEvent(&mEventHeader);
    mPrimGen.Init();
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
    if (mInitialSeed == -1) {
      mInitialSeed = getRandomSeed();
    }
    LOG(INFO) << "INITIAL SEED " << mInitialSeed;
    // set seed here ... in order to influence already event generation
    gRandom->SetSeed(mInitialSeed);

    mMaxEvents = conf.getNEvents();

    // need to make ROOT thread-safe since we use ROOT services in all places
    ROOT::EnableThreadSafety();

    // lunch initialization of particle generator asynchronously
    // so that we reach the RUNNING state of the server quickly
    // and do not block here
    mGeneratorInitThread = std::thread(&O2PrimaryServerDevice::initGenerator, this);
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
    if (Send(message, "primary-get") > 0) {
      LOG(INFO) << "config reply send ";
      return true;
    }
    return true;
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

    // we only need the initialized generator at this moment
    if (mGeneratorInitThread.joinable()) {
      mGeneratorInitThread.join();
    }

    static int counter = 0;
    if (counter >= mMaxEvents && mNeedNewEvent) {
      return false;
    }

    LOG(INFO) << "Received request for work ";
    if (mNeedNewEvent) {
      mStack.Reset();
      mPrimGen.GenerateEvent(&mStack);
      mNeedNewEvent = false;
      mPartCounter = 0;
      counter++;
    }

    auto& prims = mStack.getPrimaries();
    auto numberofparts = (int)std::ceil(prims.size() / (1. * mChunkGranularity));

    o2::Data::PrimaryChunk m;
    o2::Data::SubEventInfo i;
    i.eventID = counter;
    i.maxEvents = mMaxEvents;
    i.part = mPartCounter + 1;
    i.nparts = numberofparts;
    i.seed = counter + mInitialSeed;
    i.index = m.mParticles.size();
    m.mEventIDs.emplace_back(i);

    //auto startoffset = (mPartCounter + 1) * mChunkGranularity;
    //auto endoffset = startindex + mChunkGranularity;
    //auto startiter = prims.begin() + mPartCounter * mChunkGranularity;
    //auto enditer = endindex < prims.size() ? startiter + mChunkGranularity : prims.end();
    //auto startiter = startoffset < prims.size() ? prims.rbegin() - startoffset : prims.begin();
    //auto remaining = prims.size() - (mPartCounter + 1) * mChunkGranularity;
    //auto enditer = startiter + (remaining > mChunkGranularity)? mChunkGranularity : remaining;
    int endindex = prims.size() - mPartCounter * mChunkGranularity;
    int startindex = prims.size() - (mPartCounter + 1) * mChunkGranularity;
    if (startindex < 0) {
      startindex = 0;
    }
    if (endindex < 0) {
      endindex = 0;
    }

    // std::copy(startiter, enditer, std::back_inserter(m.mParticles));
    for (int index = startindex; index < endindex; ++index) {
      m.mParticles.emplace_back(prims[index]);
    }

    LOG(WARNING) << "Sending " << m.mParticles.size() << " particles\n";
    LOG(WARNING) << "treating ev " << counter << " part " << i.part << " out of " << i.nparts << "\n";

    mPartCounter++;
    if (mPartCounter == numberofparts) {
      mNeedNewEvent = true;
    }

    TMessage* tmsg = new TMessage(kMESS_OBJECT);
    tmsg->WriteObjectAny((void*)&m, TClass::GetClass("o2::Data::PrimaryChunk"));

    auto free_tmessage = [](void* data, void* hint) { delete static_cast<TMessage*>(hint); };

    std::unique_ptr<FairMQMessage> message(
      fTransportFactory->CreateMessage(tmsg->Buffer(), tmsg->BufferSize(), free_tmessage, tmsg));

    // send answer
    if (Send(message, "primary-get") > 0) {
      LOG(INFO) << "reply send ";
      return true;
    }
    return true;
  }

 private:
  // helper function to get truly random seed
  int getRandomSeed() const
  {
    int randomDataHandle = open("/dev/urandom", O_RDONLY);
    if (randomDataHandle < 0) {
      // something went wrong
    } else {
      int seed;
      auto result = read(randomDataHandle, &seed, sizeof(seed));
      if (result < 0) {
        // something went wrong
      }
      close(randomDataHandle);
      return seed;
    }
    return 0;
  }

  std::string mOutChannelName = "";
  FairPrimaryGenerator mPrimGen;
  FairMCEventHeader mEventHeader;
  o2::Data::Stack mStack;      // the stack which is filled
  int mChunkGranularity = 500; // how many primaries to send to a worker
  int mLastPosition = 0;       // last position in stack vector
  int mPartCounter = 0;
  bool mNeedNewEvent = true;
  int mMaxEvents = 2;
  int mInitialSeed = -1;

  std::thread mGeneratorInitThread; //! a thread used to concurrently init the particle generator
};

} // namespace devices
} // namespace o2

#endif
