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

#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ControlService.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/DomainInfoHeader.h"
#include "Framework/RateLimiter.h"

#include "DetectorsRaw/RawFileReader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "Headers/STFHeader.h"
#include "Headers/Stack.h"

#include "RawFileReaderWorkflow.h" // not installed
#include <TStopwatch.h>
#include <fairmq/Device.h>
#include <fairmq/Message.h>
#include <fairmq/Parts.h>

#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <string>
#include <climits>
#include <regex>
#include <chrono>
#include <thread>

using namespace o2::raw;
using DetID = o2::detectors::DetID;

namespace o2f = o2::framework;
namespace o2h = o2::header;

class RawReaderSpecs : public o2f::Task
{
 public:
  explicit RawReaderSpecs(const ReaderInp& rinp);
  void init(o2f::InitContext& ic) final;
  void run(o2f::ProcessingContext& ctx) final;

  uint32_t getMinTFID() const { return mMinTFID; }
  uint32_t getMaxTFID() const { return mMaxTFID; }
  void setMinMaxTFID(uint32_t mn, uint32_t mx)
  {
    mMinTFID = mn;
    mMaxTFID = mx >= mn ? mx : mn;
  }

 private:
  void processDropTF(const std::string& drops);

  int mLoop = 0;                  // once last TF reached, loop while mLoop>=0
  uint32_t mTFCounter = 0;        // TFId accumulator (accounts for looping)
  uint32_t mDelayUSec = 0;        // Delay in microseconds between TFs
  uint32_t mMinTFID = 0;          // 1st TF to extract
  uint32_t mMaxTFID = 0xffffffff; // last TF to extrct
  int mRunNumber = 0;             // run number to pass
  int mVerbosity = 0;
  int mTFRateLimit = -999;
  bool mPreferCalcTF = false;
  size_t mMinSHM = 0;
  size_t mLoopsDone = 0;
  size_t mSentSize = 0;
  size_t mSentMessages = 0;
  bool mPartPerSP = true;                                          // fill part per superpage
  bool mSup0xccdb = false;                                         // suppress explicit FLP/DISTSUBTIMEFRAME/0xccdb output
  std::string mRawChannelName = "";                                // name of optional non-DPL channel
  std::unique_ptr<o2::raw::RawFileReader> mReader;                 // matching engine
  std::unordered_map<std::string, std::pair<int, int>> mDropTFMap; // allows to drop certain fraction of TFs
  TStopwatch mTimer;
};

//___________________________________________________________
RawReaderSpecs::RawReaderSpecs(const ReaderInp& rinp)
  : mLoop(rinp.loop < 0 ? INT_MAX : (rinp.loop < 1 ? 1 : rinp.loop)), mDelayUSec(rinp.delay_us), mMinTFID(rinp.minTF), mMaxTFID(rinp.maxTF), mRunNumber(rinp.runNumber), mPartPerSP(rinp.partPerSP), mSup0xccdb(rinp.sup0xccdb), mReader(std::make_unique<o2::raw::RawFileReader>(rinp.inifile, 0, rinp.bufferSize, rinp.onlyDet)), mRawChannelName(rinp.rawChannelConfig), mPreferCalcTF(rinp.preferCalcTF), mMinSHM(rinp.minSHM)
{
  mReader->setCheckErrors(rinp.errMap);
  mReader->setMaxTFToRead(rinp.maxTF);
  mReader->setNominalSPageSize(rinp.spSize);
  mReader->setCacheData(rinp.cache);
  mReader->setTFAutodetect(rinp.autodetectTF0 ? RawFileReader::FirstTFDetection::Pending : RawFileReader::FirstTFDetection::Disabled);
  mReader->setPreferCalculatedTFStart(rinp.preferCalcTF);
  LOG(info) << "Will preprocess files with buffer size of " << rinp.bufferSize << " bytes";
  LOG(info) << "Number of loops over whole data requested: " << mLoop;
  mTimer.Stop();
  mTimer.Reset();
  processDropTF(rinp.dropTF);
}

//___________________________________________________________
void RawReaderSpecs::processDropTF(const std::string& dropTF)
{
  static const std::regex delimDet(";");
  if (dropTF.empty() || dropTF == "none") {
    return;
  }
  std::sregex_token_iterator iter(dropTF.begin(), dropTF.end(), delimDet, -1), end;
  for (; iter != end; ++iter) {
    std::string sdet = iter->str();
    if (sdet.length() < 5 || sdet[3] != ',') {
      throw std::runtime_error(fmt::format("Wrong dropTF argument {} in {}", sdet, dropTF));
    }
    std::string detName = sdet.substr(0, 3);
    o2::detectors::DetID det(detName.c_str()); // make sure this is a valid detector
    std::string sdetArg = sdet.substr(4, sdet.length());
    int modV = 0, rej = 0, posrej = sdetArg.find(',');
    if (posrej != std::string::npos) {
      modV = std::stoi(sdetArg.substr(0, posrej));
      rej = std::stoi(sdetArg.substr(++posrej, sdetArg.length()));
    } else {
      modV = std::stoi(sdetArg);
    }
    if (modV < 1 || rej < 0 || rej >= modV) {
      throw std::runtime_error(fmt::format("Wrong dropTF argument {}, 1st number must be > than 2nd", sdet));
    }
    mDropTFMap[detName] = {modV, rej};
    LOG(info) << " Will drop TF for detector " << detName << " if (TF_ID%" << modV << ")==" << rej;
  }
}

//___________________________________________________________
void RawReaderSpecs::init(o2f::InitContext& ic)
{
  assert(mReader);
  mTimer.Start();
  mTimer.Stop();
  mVerbosity = ic.options().get<int>("verbosity-level");
  mReader->setVerbosity(mVerbosity);
  mReader->init();
  if (mMaxTFID >= mReader->getNTimeFrames()) {
    mMaxTFID = mReader->getNTimeFrames() ? mReader->getNTimeFrames() - 1 : 0;
  }
  const auto& hbfU = HBFUtils::Instance();
  if (!hbfU.startTime) {
    hbfU.setValue("HBFUtils.startTime", std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
    LOG(warning) << "Run start time is not provided via HBFUtils.startTime, will use now() = " << hbfU.startTime << " ms.";
  }
  if (mRunNumber == 0 && hbfU.runNumber > 0) {
    mRunNumber = hbfU.runNumber;
  }
}

//___________________________________________________________
void RawReaderSpecs::run(o2f::ProcessingContext& ctx)
{
  assert(mReader);
  auto tTotStart = mTimer.CpuTime();
  mTimer.Start(false);
  auto device = ctx.services().get<o2f::RawDeviceService>().device();
  assert(device);
  if (mTFRateLimit == -999) {
    mTFRateLimit = std::stoi(device->fConfig->GetValue<std::string>("timeframes-rate-limit"));
  }
  auto findOutputChannel = [&ctx, this](o2h::DataHeader& h) {
    if (!this->mRawChannelName.empty()) {
      return std::string{this->mRawChannelName};
    } else {
      auto outputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs;
      for (auto& oroute : outputRoutes) {
        LOG(debug) << "comparing with matcher to route " << oroute.matcher << " TSlice:" << oroute.timeslice;
        if (o2f::DataSpecUtils::match(oroute.matcher, h.dataOrigin, h.dataDescription, h.subSpecification) && ((mTFCounter % oroute.maxTimeslices) == oroute.timeslice)) {
          LOG(debug) << "picking the route:" << o2f::DataSpecUtils::describe(oroute.matcher) << " channel " << oroute.channel;
          return std::string{oroute.channel};
        }
      }
    }
    LOGP(error, "Failed to find output channel for {}/{}/{} @ timeslice {}", h.dataOrigin, h.dataDescription, h.subSpecification, h.tfCounter);
    auto outputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs;
    for (auto& oroute : outputRoutes) {
      LOGP(info, "Available output routes: {} channel: {}", o2f::DataSpecUtils::describe(oroute.matcher), oroute.channel);
    }
    return std::string{};
  };

  size_t tfNParts = 0, tfSize = 0;
  std::unordered_map<std::string, std::unique_ptr<fair::mq::Parts>> messagesPerRoute;

  auto addPart = [&messagesPerRoute, &tfNParts, &tfSize](fair::mq::MessagePtr hd, fair::mq::MessagePtr pl, const std::string& fairMQChannel) {
    fair::mq::Parts* parts = nullptr;
    parts = messagesPerRoute[fairMQChannel].get(); // fair::mq::Parts*
    if (!parts) {
      messagesPerRoute[fairMQChannel] = std::make_unique<fair::mq::Parts>();
      parts = messagesPerRoute[fairMQChannel].get();
    }
    tfSize += pl->GetSize();
    tfNParts++;
    parts->AddPart(std::move(hd));
    parts->AddPart(std::move(pl));
  };

  // clean-up before reading next TF
  auto tfID = mReader->getNextTFToRead();
  int nlinks = mReader->getNLinks();

  if (tfID > mMaxTFID || mReader->isProcessingStopped()) {
    if (!mReader->isProcessingStopped() && !mReader->isEmpty() && --mLoop) {
      mLoopsDone++;
      tfID = 0;
      LOG(info) << "Starting new loop " << mLoopsDone << " from the beginning of data";
    } else {
      if (!mRawChannelName.empty()) { // send endOfStream message to raw channel
        o2f::SourceInfoHeader exitHdr;
        exitHdr.state = o2f::InputChannelState::Completed;
        const auto exitStack = o2::header::Stack(o2h::DataHeader(o2h::gDataDescriptionInfo, o2h::gDataOriginAny, 0, 0), o2f::DataProcessingHeader(), exitHdr);
        auto fmqFactory = device->GetChannel(mRawChannelName, 0).Transport();
        auto hdEOSMessage = fmqFactory->CreateMessage(exitStack.size(), fair::mq::Alignment{64});
        auto plEOSMessage = fmqFactory->CreateMessage(0, fair::mq::Alignment{64});
        memcpy(hdEOSMessage->GetData(), exitStack.data(), exitStack.size());
        fair::mq::Parts eosMsg;
        eosMsg.AddPart(std::move(hdEOSMessage));
        eosMsg.AddPart(std::move(plEOSMessage));
        device->Send(eosMsg, mRawChannelName);
        LOG(info) << "Sent EoS message to " << mRawChannelName;
      } else {
        ctx.services().get<o2f::ControlService>().endOfStream();
      }
      ctx.services().get<o2f::ControlService>().readyToQuit(o2f::QuitRequest::Me);
      mTimer.Stop();
      LOGP(info, "Finished: payload of {} bytes in {} messages sent for {} TFs, total timing: Real:{:3f}/CPU:{:3f}", mSentSize, mSentMessages, mTFCounter, mTimer.RealTime(), mTimer.CpuTime());
      return;
    }
  }

  if (tfID < mMinTFID) {
    tfID = mMinTFID;
  }
  mReader->setNextTFToRead(tfID);
  std::vector<RawFileReader::PartStat> partsSP;

  static o2f::RateLimiter limiter;
  limiter.check(ctx, mTFRateLimit, mMinSHM);

  // read next time frame
  LOG(info) << "Reading TF#" << mTFCounter << " (" << tfID << " at iteration " << mLoopsDone << ')';
  o2::header::Stack dummyStack{o2h::DataHeader{}, o2f::DataProcessingHeader{0}}; // dummy stack to just to get stack size
  auto hstackSize = dummyStack.size();

  uint32_t firstOrbit = 0;
  uint64_t creationTime = 0;
  const auto& hbfU = HBFUtils::Instance();

  for (int il = 0; il < nlinks; il++) {
    auto& link = mReader->getLink(il);

    if (!mDropTFMap.empty()) { // some TFs should be dropped
      auto res = mDropTFMap.find(link.origin.str);
      if (res != mDropTFMap.end() && (mTFCounter % res->second.first) == res->second.second) {
        LOG(info) << "Dropping " << mTFCounter << " for " << link.origin.str << "/" << link.description.str << "/" << link.subspec;
        continue; // drop the data
      }
    }
    if (!link.rewindToTF(tfID)) {
      continue; // this link has no data for wanted TF
    }

    o2h::DataHeader hdrTmpl(link.description, link.origin, link.subspec); // template with 0 size
    int nParts = mPartPerSP ? link.getNextTFSuperPagesStat(partsSP) : link.getNHBFinTF();
    hdrTmpl.payloadSerializationMethod = o2h::gSerializationMethodNone;
    hdrTmpl.splitPayloadParts = nParts;
    hdrTmpl.tfCounter = mTFCounter;
    hdrTmpl.runNumber = mRunNumber;
    if (mVerbosity > 1) {
      LOG(info) << link.describe() << " will read " << nParts << " HBFs starting from block " << link.nextBlock2Read;
    }
    const auto fmqChannel = findOutputChannel(hdrTmpl);
    if (fmqChannel.empty()) { // no output channel
      continue;
    }

    auto fmqFactory = device->GetChannel(fmqChannel, 0).Transport();
    while (hdrTmpl.splitPayloadIndex < hdrTmpl.splitPayloadParts) {
      hdrTmpl.payloadSize = mPartPerSP ? partsSP[hdrTmpl.splitPayloadIndex].size : link.getNextHBFSize();
      auto hdMessage = fmqFactory->CreateMessage(hstackSize, fair::mq::Alignment{64});
      auto plMessage = fmqFactory->CreateMessage(hdrTmpl.payloadSize, fair::mq::Alignment{64});
      auto bread = mPartPerSP ? link.readNextSuperPage(reinterpret_cast<char*>(plMessage->GetData()), &partsSP[hdrTmpl.splitPayloadIndex]) : link.readNextHBF(reinterpret_cast<char*>(plMessage->GetData()));
      if (bread != hdrTmpl.payloadSize) {
        LOG(error) << "Link " << il << " read " << bread << " bytes instead of " << hdrTmpl.payloadSize
                   << " expected in TF=" << mTFCounter << " part=" << hdrTmpl.splitPayloadIndex;
      }
      // check if the RDH to send corresponds to expected orbit
      if (hdrTmpl.splitPayloadIndex == 0) {
        auto ir = o2::raw::RDHUtils::getHeartBeatIR(plMessage->GetData());
        auto tfid = hbfU.getTF(ir);
        firstOrbit = hdrTmpl.firstTForbit = (mPreferCalcTF || !link.cruDetector) ? hbfU.getIRTF(tfid).orbit : ir.orbit; // will be picked for the following parts
        creationTime = hbfU.getTFTimeStamp({0, firstOrbit});
      }
      o2::header::Stack headerStack{hdrTmpl, o2f::DataProcessingHeader{mTFCounter, 1, creationTime}};
      memcpy(hdMessage->GetData(), headerStack.data(), headerStack.size());
      hdrTmpl.splitPayloadIndex++; // prepare for next

      addPart(std::move(hdMessage), std::move(plMessage), fmqChannel);
    }
    LOGF(debug, "Added %d parts for TF#%d(%d in iteration %d) of %s/%s/0x%u", hdrTmpl.splitPayloadParts, mTFCounter, tfID,
         mLoopsDone, link.origin.as<std::string>(), link.description.as<std::string>(), link.subspec);
  }

  auto& timingInfo = ctx.services().get<o2f::TimingInfo>();
  timingInfo.firstTForbit = firstOrbit;
  timingInfo.creation = creationTime;
  timingInfo.tfCounter = mTFCounter;
  timingInfo.runNumber = mRunNumber;

  // send sTF acknowledge message
  unsigned stfSS[2] = {0, 0xccdb};
  for (int iss = 0; iss < (mSup0xccdb ? 1 : 2); iss++) {
    o2::header::STFHeader stfHeader{mTFCounter, firstOrbit, 0};
    o2::header::DataHeader stfDistDataHeader(o2::header::gDataDescriptionDISTSTF, o2::header::gDataOriginFLP, stfSS[iss], sizeof(o2::header::STFHeader), 0, 1);
    stfDistDataHeader.runNumber = mRunNumber;
    stfDistDataHeader.payloadSerializationMethod = o2h::gSerializationMethodNone;
    stfDistDataHeader.firstTForbit = stfHeader.firstOrbit;
    stfDistDataHeader.tfCounter = mTFCounter;
    const auto fmqChannel = findOutputChannel(stfDistDataHeader);
    if (!fmqChannel.empty()) { // no output channel
      auto fmqFactory = device->GetChannel(fmqChannel, 0).Transport();
      o2::header::Stack headerStackSTF{stfDistDataHeader, o2f::DataProcessingHeader{mTFCounter, 1, creationTime}};
      auto hdMessageSTF = fmqFactory->CreateMessage(hstackSize, fair::mq::Alignment{64});
      auto plMessageSTF = fmqFactory->CreateMessage(stfDistDataHeader.payloadSize, fair::mq::Alignment{64});
      memcpy(hdMessageSTF->GetData(), headerStackSTF.data(), headerStackSTF.size());
      memcpy(plMessageSTF->GetData(), &stfHeader, sizeof(o2::header::STFHeader));
      addPart(std::move(hdMessageSTF), std::move(plMessageSTF), fmqChannel);
    }
  }

  if (mTFCounter) { // delay sending
    std::this_thread::sleep_for(std::chrono::microseconds((size_t)mDelayUSec));
  }
  bool sentSomething = false;
  for (auto& msgIt : messagesPerRoute) {
    LOG(info) << "Sending " << msgIt.second->Size() / 2 << " parts to channel " << msgIt.first;
    device->Send(*msgIt.second.get(), msgIt.first);
    sentSomething = msgIt.second->Size() > 0;
  }
  if (sentSomething) {
    ctx.services().get<o2f::MessageContext>().fakeDispatch();
  }

  mTimer.Stop();

  LOGP(info, "Sent payload of {} bytes in {} parts in {} messages for TF#{} firstTForbit={} timeStamp={} | Timing: {}", tfSize, tfNParts,
       messagesPerRoute.size(), mTFCounter, firstOrbit, creationTime, mTimer.CpuTime() - tTotStart);

  mSentSize += tfSize;
  mSentMessages += tfNParts;
  mReader->setNextTFToRead(++tfID);
  ++mTFCounter;
}

//_________________________________________________________
o2f::DataProcessorSpec getReaderSpec(ReaderInp rinp)
{
  // check which inputs are present in files to read
  o2f::DataProcessorSpec spec;
  spec.name = "raw-file-reader";
  std::string rawChannelName = "";
  if (rinp.rawChannelConfig.empty()) {
    if (!rinp.inifile.empty()) {
      auto conf = o2::raw::RawFileReader::parseInput(rinp.inifile, rinp.onlyDet);
      for (const auto& entry : conf) {
        const auto& ordescard = entry.first;
        if (!entry.second.empty()) { // origin and decription for files to process
          spec.outputs.emplace_back(o2f::OutputSpec(o2f::ConcreteDataTypeMatcher{std::get<0>(ordescard), std::get<1>(ordescard)}));
        }
      }
    }
    // add output for DISTSUBTIMEFRAME
    spec.outputs.emplace_back(o2f::OutputSpec{{"stfDist"}, o2::header::gDataOriginFLP, o2::header::gDataDescriptionDISTSTF, 0});
    if (!rinp.sup0xccdb) {
      spec.outputs.emplace_back(o2f::OutputSpec{{"stfDistCCDB"}, o2::header::gDataOriginFLP, o2::header::gDataDescriptionDISTSTF, 0xccdb}); // will be added automatically
    }
    if (!rinp.metricChannel.empty()) {
      spec.options.emplace_back(o2f::ConfigParamSpec{"channel-config", o2f::VariantType::String, rinp.metricChannel, {"Out-of-band channel config for TF throttling"}});
    }
  } else {
    auto nameStart = rinp.rawChannelConfig.find("name=");
    if (nameStart == std::string::npos) {
      throw std::runtime_error("raw channel name is not provided");
    }
    nameStart += strlen("name=");
    auto nameEnd = rinp.rawChannelConfig.find(",", nameStart + 1);
    if (nameEnd == std::string::npos) {
      nameEnd = rinp.rawChannelConfig.size();
    }
    spec.options.emplace_back(o2f::ConfigParamSpec{"channel-config", o2f::VariantType::String, rinp.rawChannelConfig, {"Out-of-band channel config"}});
    rinp.rawChannelConfig = rinp.rawChannelConfig.substr(nameStart, nameEnd - nameStart);
    if (!rinp.metricChannel.empty()) {
      LOGP(alarm, "Cannot apply TF rate limiting when publishing to raw channel, limiting must be applied on the level of the input raw proxy");
      LOGP(alarm, R"(To avoid reader filling shm buffer use "--shm-throw-bad-alloc 0 --shm-segment-id 2")");
    }
    LOG(info) << "Will send output to non-DPL channel " << rinp.rawChannelConfig;
  }

  spec.algorithm = o2f::adaptFromTask<RawReaderSpecs>(rinp);
  spec.options.emplace_back(o2f::ConfigParamSpec{"verbosity-level", o2f::VariantType::Int, 0, {"verbosity level"}});
  return spec;
}

o2f::WorkflowSpec o2::raw::getRawFileReaderWorkflow(ReaderInp& rinp)
{
  o2f::WorkflowSpec specs;
  specs.emplace_back(getReaderSpec(rinp));
  return specs;
}
