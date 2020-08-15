// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DetectorsRaw/RawFileReader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DetectorsRaw/HBFUtils.h"

#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include "RawFileReaderWorkflow.h" // not installed
#include <TStopwatch.h>
#include <fairmq/FairMQDevice.h>

#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <string>
#include <climits>

using namespace o2::raw;

namespace o2f = o2::framework;
namespace o2h = o2::header;

class RawReaderSpecs : public o2f::Task
{
 public:
  explicit RawReaderSpecs(const std::string& config, int loop = 1, uint32_t delay_us = 0,
                          uint32_t errmap = 0xffffffff, uint32_t minTF = 0, uint32_t maxTF = 0xffffffff, bool partPerSP = true, bool cache = false,
                          size_t spSize = 1024L * 1024L, size_t buffSize = 5 * 1024UL,
                          const std::string& rawChannelName = "")
    : mLoop(loop < 0 ? INT_MAX : (loop < 1 ? 1 : loop)), mDelayUSec(delay_us), mMinTFID(minTF), mMaxTFID(maxTF), mPartPerSP(partPerSP), mReader(std::make_unique<o2::raw::RawFileReader>(config, 0, buffSize)), mRawChannelName(rawChannelName)
  {
    mReader->setCheckErrors(errmap);
    mReader->setMaxTFToRead(maxTF);
    mReader->setNominalSPageSize(spSize);
    mReader->setCacheData(cache);
    LOG(INFO) << "Will preprocess files with buffer size of " << buffSize << " bytes";
    LOG(INFO) << "Number of loops over whole data requested: " << mLoop;
    for (int i = NTimers; i--;) {
      mTimer[i].Stop();
      mTimer[i].Reset();
    }
  }

  void init(o2f::InitContext& ic) final
  {
    assert(mReader);
    mTimer[TimerInit].Start();
    mReader->init();
    mTimer[TimerInit].Stop();
    if (mMaxTFID >= mReader->getNTimeFrames()) {
      mMaxTFID = mReader->getNTimeFrames() ? mReader->getNTimeFrames() - 1 : 0;
    }
  }

  void run(o2f::ProcessingContext& ctx) final
  {
    assert(mReader);
    if (mDone) {
      return;
    }
    auto tTotStart = mTimer[TimerTotal].CpuTime(), tIOStart = mTimer[TimerIO].CpuTime();
    mTimer[TimerTotal].Start(false);
    auto device = ctx.services().get<o2f::RawDeviceService>().device();
    assert(device);

    auto findOutputChannel = [&ctx, this](RawFileReader::LinkData& link, size_t timeslice) {
      if (!this->mRawChannelName.empty()) {
        link.fairMQChannel = this->mRawChannelName;
        return true;
      }
      auto outputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs;
      for (auto& oroute : outputRoutes) {
        LOG(DEBUG) << "comparing with matcher to route " << oroute.matcher << " TSlice:" << oroute.timeslice;
        if (o2f::DataSpecUtils::match(oroute.matcher, link.origin, link.description, link.subspec) && ((timeslice % oroute.maxTimeslices) == oroute.timeslice)) {
          link.fairMQChannel = oroute.channel;
          LOG(DEBUG) << "picking the route:" << o2f::DataSpecUtils::describe(oroute.matcher) << " channel " << oroute.channel;
          return true;
        }
      }
      LOGF(ERROR, "Failed to find output channel for %s/%s/0x%x", link.origin.as<std::string>(),
           link.description.as<std::string>(), link.subspec);
      return false;
    };

    // clean-up before reading next TF
    auto tfID = mReader->getNextTFToRead();
    int nlinks = mReader->getNLinks();

    std::unordered_map<std::string, std::unique_ptr<FairMQParts>> messagesPerRoute;

    if (tfID > mMaxTFID) {
      if (!mReader->isEmpty() && --mLoop) {
        mLoopsDone++;
        tfID = 0;
        LOG(INFO) << "Starting new loop " << mLoopsDone << " from the beginning of data";
      } else {
        mTimer[TimerTotal].Stop();
        LOGF(INFO, "Finished: payload of %zu bytes in %zu messages sent for %d TFs", mSentSize, mSentMessages, mTFCounter);
        for (int i = 0; i < NTimers; i++) {
          LOGF(INFO, "Timing for %15s: Cpu: %.3e Real: %.3e s in %d slots", TimerName[i], mTimer[i].CpuTime(), mTimer[i].RealTime(), mTimer[i].Counter() - 1);
        }
        ctx.services().get<o2f::ControlService>().endOfStream();
        ctx.services().get<o2f::ControlService>().readyToQuit(o2f::QuitRequest::Me);
        mDone = true;
        return;
      }
    }

    if (tfID < mMinTFID) {
      tfID = mMinTFID;
    }
    mReader->setNextTFToRead(tfID);
    for (int il = 0; il < nlinks; il++) {
      mReader->getLink(il).rewindToTF(tfID);
    }
    std::vector<RawFileReader::PartStat> partsSP;
    const auto& hbfU = HBFUtils::Instance();

    // read next time frame
    size_t tfNParts = 0, tfSize = 0;
    LOG(INFO) << "Reading TF#" << mTFCounter << " (" << tfID << " at iteration " << mLoopsDone << ')';
    o2::header::Stack dummyStack{o2h::DataHeader{}, o2::framework::DataProcessingHeader{0}}; // dummy stack to just to get stack size
    auto hstackSize = dummyStack.size();

    for (int il = 0; il < nlinks; il++) {
      auto& link = mReader->getLink(il);

      if (!findOutputChannel(link, mTFCounter)) { // no output channel
        continue;
      }

      o2h::DataHeader hdrTmpl(link.description, link.origin, link.subspec); // template with 0 size
      int nParts = mPartPerSP ? link.getNextTFSuperPagesStat(partsSP) : link.getNHBFinTF();
      hdrTmpl.payloadSerializationMethod = o2h::gSerializationMethodNone;
      hdrTmpl.splitPayloadParts = nParts;

      while (hdrTmpl.splitPayloadIndex < hdrTmpl.splitPayloadParts) {

        tfSize += hdrTmpl.payloadSize = mPartPerSP ? partsSP[hdrTmpl.splitPayloadIndex].size : link.getNextHBFSize();
        auto fmqFactory = device->GetChannel(link.fairMQChannel, 0).Transport();
        auto hdMessage = fmqFactory->CreateMessage(hstackSize, fair::mq::Alignment{64});
        auto plMessage = fmqFactory->CreateMessage(hdrTmpl.payloadSize, fair::mq::Alignment{64});
        mTimer[TimerIO].Start(false);
        auto bread = mPartPerSP ? link.readNextSuperPage(reinterpret_cast<char*>(plMessage->GetData()), &partsSP[hdrTmpl.splitPayloadIndex]) : link.readNextHBF(reinterpret_cast<char*>(plMessage->GetData()));
        if (bread != hdrTmpl.payloadSize) {
          LOG(ERROR) << "Link " << il << " read " << bread << " bytes instead of " << hdrTmpl.payloadSize
                     << " expected in TF=" << mTFCounter << " part=" << hdrTmpl.splitPayloadIndex;
        }
        mTimer[TimerIO].Stop();
        // check if the RDH to send corresponds to expected orbit
        if (hdrTmpl.splitPayloadIndex == 0) {
          auto ir = o2::raw::RDHUtils::getHeartBeatIR(plMessage->GetData());
          auto tfid = hbfU.getTF(ir);
          hdrTmpl.firstTForbit = hbfU.getIRTF(tfid).orbit;                                           // will be picked for the
          hdrTmpl.tfCounter = mTFCounter;                                                            // following parts
          // reinterpret_cast<o2::header::DataHeader*>(hdMessage->GetData())->firstTForbit = hdrTmpl.firstTForbit;     // hack to fix already filled headers
          // reinterpret_cast<o2::header::DataHeader*>(hdMessage->GetData())->tfCounter = mTFCounter;   // at the moment don't use it
        }
        o2::header::Stack headerStack{hdrTmpl, o2::framework::DataProcessingHeader{mTFCounter}};
        memcpy(hdMessage->GetData(), headerStack.data(), headerStack.size());

        FairMQParts* parts = nullptr;
        parts = messagesPerRoute[link.fairMQChannel].get(); // FairMQParts*
        if (!parts) {
          messagesPerRoute[link.fairMQChannel] = std::make_unique<FairMQParts>();
          parts = messagesPerRoute[link.fairMQChannel].get();
        }
        parts->AddPart(std::move(hdMessage));
        parts->AddPart(std::move(plMessage));
        hdrTmpl.splitPayloadIndex++; // prepare for next
        tfNParts++;
      }
      LOGF(DEBUG, "Added %d parts for TF#%d(%d in iteration %d) of %s/%s/0x%u", hdrTmpl.splitPayloadParts, mTFCounter, tfID,
           mLoopsDone, link.origin.as<std::string>(), link.description.as<std::string>(), link.subspec);
    }

    if (mTFCounter) { // delay sending
      usleep(mDelayUSec);
    }
    for (auto& msgIt : messagesPerRoute) {
      LOG(INFO) << "Sending " << msgIt.second->Size() / 2 << " parts to channel " << msgIt.first;
      device->Send(*msgIt.second.get(), msgIt.first);
    }
    mTimer[TimerTotal].Stop();

    LOGF(INFO, "Sent payload of %zu bytes in %zu parts in %zu messages for TF %d | Timing (total/IO): %.3e / %.3e", tfSize, tfNParts,
         messagesPerRoute.size(), mTFCounter, mTimer[TimerTotal].CpuTime() - tTotStart, mTimer[TimerIO].CpuTime() - tIOStart);

    mSentSize += tfSize;
    mSentMessages += tfNParts;

    mReader->setNextTFToRead(++tfID);
    ++mTFCounter;
  }

  uint32_t getMinTFID() const { return mMinTFID; }
  uint32_t getMaxTFID() const { return mMaxTFID; }
  void setMinMaxTFID(uint32_t mn, uint32_t mx)
  {
    mMinTFID = mn;
    mMaxTFID = mx >= mn ? mx : mn;
  }

 private:
  int mLoop = 0;                                   // once last TF reached, loop while mLoop>=0
  uint32_t mTFCounter = 0;                         // TFId accumulator (accounts for looping)
  uint32_t mDelayUSec = 0;                         // Delay in microseconds between TFs
  uint32_t mMinTFID = 0;                           // 1st TF to extract
  uint32_t mMaxTFID = 0xffffffff;                  // last TF to extrct
  size_t mLoopsDone = 0;
  size_t mSentSize = 0;
  size_t mSentMessages = 0;
  bool mPartPerSP = true;                          // fill part per superpage
  bool mDone = false;                              // processing is over or not
  std::string mRawChannelName = "";                // name of optional non-DPL channel
  std::unique_ptr<o2::raw::RawFileReader> mReader; // matching engine

  enum TimerIDs { TimerInit,
                  TimerTotal,
                  TimerIO,
                  NTimers };
  static constexpr std::string_view TimerName[] = {"Init", "Total", "IO"};
  TStopwatch mTimer[NTimers];
};

o2f::DataProcessorSpec getReaderSpec(std::string config, int loop, uint32_t delay_us, uint32_t errmap,
                                     uint32_t minTF, uint32_t maxTF, bool partPerSP, bool cache, size_t spSize, size_t buffSize, const std::string& rawChannelConfig)
{
  // check which inputs are present in files to read
  o2f::DataProcessorSpec spec;
  spec.name = "raw-file-reader";
  std::string rawChannelName = "";
  if (rawChannelConfig.empty()) {
    if (!config.empty()) {
      auto conf = o2::raw::RawFileReader::parseInput(config);
      for (const auto& entry : conf) {
        const auto& ordescard = entry.first;
        if (!entry.second.empty()) { // origin and decription for files to process
          spec.outputs.emplace_back(o2f::OutputSpec(o2f::ConcreteDataTypeMatcher{std::get<0>(ordescard), std::get<1>(ordescard)}));
        }
      }
    }
  } else {
    auto nameStart = rawChannelConfig.find("name=");
    if (nameStart == std::string::npos) {
      throw std::runtime_error("raw channel name is not provided");
    }
    nameStart += strlen("name=");
    auto nameEnd = rawChannelConfig.find(",", nameStart + 1);
    if (nameEnd == std::string::npos) {
      nameEnd = rawChannelConfig.size();
    }
    rawChannelName = rawChannelConfig.substr(nameStart, nameEnd - nameStart);
    spec.options = {o2f::ConfigParamSpec{"channel-config", o2f::VariantType::String, rawChannelConfig, {"Out-of-band channel config"}}};
    LOG(INFO) << "Will send output to non-DPL channel " << rawChannelConfig;
  }

  spec.algorithm = o2f::adaptFromTask<RawReaderSpecs>(config, loop, delay_us, errmap, minTF, maxTF, partPerSP, cache, spSize, buffSize, rawChannelName);

  return spec;
}

o2f::WorkflowSpec o2::raw::getRawFileReaderWorkflow(std::string inifile, int loop, uint32_t delay_us, uint32_t errmap, uint32_t minTF, uint32_t maxTF,
                                                    bool partPerSP, bool cache, size_t spSize, size_t buffSize, const std::string& rawChannelConfig)
{
  o2f::WorkflowSpec specs;
  specs.emplace_back(getReaderSpec(inifile, loop, delay_us, errmap, minTF, maxTF, partPerSP, cache, spSize, buffSize, rawChannelConfig));
  return specs;
}
