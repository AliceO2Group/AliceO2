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
#include "DetectorsCommonDataFormats/DetID.h"
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
#include <regex>

using namespace o2::raw;

namespace o2f = o2::framework;
namespace o2h = o2::header;

class RawReaderSpecs : public o2f::Task
{
 public:
  static constexpr o2h::DataDescription gDataDescSubTimeFrame{"DISTSUBTIMEFRAME"};
  struct STFHeader { // fake header to mimic DD SubTimeFrame::Header sent with DISTSUBTIMEFRAME message
    uint64_t mId = uint64_t(-1);
    uint32_t mFirstOrbit = uint32_t(-1);
    std::uint32_t mRunNumber = 0;
  };
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
  size_t mLoopsDone = 0;
  size_t mSentSize = 0;
  size_t mSentMessages = 0;
  bool mPartPerSP = true;                                          // fill part per superpage
  std::string mRawChannelName = "";                                // name of optional non-DPL channel
  std::unique_ptr<o2::raw::RawFileReader> mReader;                 // matching engine
  std::unordered_map<std::string, std::pair<int, int>> mDropTFMap; // allows to drop certain fraction of TFs
  enum TimerIDs { TimerInit,
                  TimerTotal,
                  TimerIO,
                  NTimers };
  static constexpr std::string_view TimerName[] = {"Init", "Total", "IO"};
  TStopwatch mTimer[NTimers];
};

//___________________________________________________________
RawReaderSpecs::RawReaderSpecs(const ReaderInp& rinp)
  : mLoop(rinp.loop < 0 ? INT_MAX : (rinp.loop < 1 ? 1 : rinp.loop)), mDelayUSec(rinp.delay_us), mMinTFID(rinp.minTF), mMaxTFID(rinp.maxTF), mPartPerSP(rinp.partPerSP), mReader(std::make_unique<o2::raw::RawFileReader>(rinp.inifile, 0, rinp.bufferSize)), mRawChannelName(rinp.rawChannelConfig)
{
  mReader->setCheckErrors(rinp.errMap);
  mReader->setMaxTFToRead(rinp.maxTF);
  mReader->setNominalSPageSize(rinp.spSize);
  mReader->setCacheData(rinp.cache);
  mReader->setTFAutodetect(rinp.autodetectTF0 ? RawFileReader::FirstTFDetection::Pending : RawFileReader::FirstTFDetection::Disabled);
  mReader->setPreferCalculatedTFStart(rinp.preferCalcTF);
  LOG(INFO) << "Will preprocess files with buffer size of " << rinp.bufferSize << " bytes";
  LOG(INFO) << "Number of loops over whole data requested: " << mLoop;
  for (int i = NTimers; i--;) {
    mTimer[i].Stop();
    mTimer[i].Reset();
  }
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
    LOG(INFO) << " Will drop TF for detector " << detName << " if (TF_ID%" << modV << ")==" << rej;
  }
}

//___________________________________________________________
void RawReaderSpecs::init(o2f::InitContext& ic)
{
  assert(mReader);
  mTimer[TimerInit].Start();
  mReader->init();
  mTimer[TimerInit].Stop();
  if (mMaxTFID >= mReader->getNTimeFrames()) {
    mMaxTFID = mReader->getNTimeFrames() ? mReader->getNTimeFrames() - 1 : 0;
  }
}

//___________________________________________________________
void RawReaderSpecs::run(o2f::ProcessingContext& ctx)
{
  assert(mReader);
  auto tTotStart = mTimer[TimerTotal].CpuTime(), tIOStart = mTimer[TimerIO].CpuTime();
  mTimer[TimerTotal].Start(false);
  auto device = ctx.services().get<o2f::RawDeviceService>().device();
  assert(device);

  auto findOutputChannel = [&ctx, this](o2h::DataHeader& h) {
    if (!this->mRawChannelName.empty()) {
      return std::string{this->mRawChannelName};
    } else {
      auto outputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs;
      for (auto& oroute : outputRoutes) {
        LOG(DEBUG) << "comparing with matcher to route " << oroute.matcher << " TSlice:" << oroute.timeslice;
        if (o2f::DataSpecUtils::match(oroute.matcher, h.dataOrigin, h.dataDescription, h.subSpecification) && ((h.tfCounter % oroute.maxTimeslices) == oroute.timeslice)) {
          LOG(DEBUG) << "picking the route:" << o2f::DataSpecUtils::describe(oroute.matcher) << " channel " << oroute.channel;
          return std::string{oroute.channel};
        }
      }
    }
    LOGP(ERROR, "Failed to find output channel for {}/{}/{} @ timeslice {}", h.dataOrigin.str, h.dataDescription.str, h.subSpecification, h.tfCounter);
    return std::string{};
  };

  size_t tfNParts = 0, tfSize = 0;
  std::unordered_map<std::string, std::unique_ptr<FairMQParts>> messagesPerRoute;

  auto addPart = [&messagesPerRoute, &tfNParts, &tfSize](FairMQMessagePtr hd, FairMQMessagePtr pl, const std::string& fairMQChannel) {
    FairMQParts* parts = nullptr;
    parts = messagesPerRoute[fairMQChannel].get(); // FairMQParts*
    if (!parts) {
      messagesPerRoute[fairMQChannel] = std::make_unique<FairMQParts>();
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
      return;
    }
  }

  if (tfID < mMinTFID) {
    tfID = mMinTFID;
  }
  mReader->setNextTFToRead(tfID);
  std::vector<RawFileReader::PartStat> partsSP;
  const auto& hbfU = HBFUtils::Instance();

  // read next time frame
  LOG(INFO) << "Reading TF#" << mTFCounter << " (" << tfID << " at iteration " << mLoopsDone << ')';
  o2::header::Stack dummyStack{o2h::DataHeader{}, o2::framework::DataProcessingHeader{0}}; // dummy stack to just to get stack size
  auto hstackSize = dummyStack.size();

  uint32_t firstOrbit = 0;
  for (int il = 0; il < nlinks; il++) {
    auto& link = mReader->getLink(il);

    if (!mDropTFMap.empty()) { // some TFs should be dropped
      auto res = mDropTFMap.find(link.origin.str);
      if (res != mDropTFMap.end() && (mTFCounter % res->second.first) == res->second.second) {
        LOG(INFO) << "Droppint " << mTFCounter << " for " << link.origin.str << "/" << link.description.str << "/" << link.subspec;
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

    const auto fmqChannel = findOutputChannel(hdrTmpl);
    if (fmqChannel.empty()) { // no output channel
      continue;
    }

    auto fmqFactory = device->GetChannel(fmqChannel, 0).Transport();
    while (hdrTmpl.splitPayloadIndex < hdrTmpl.splitPayloadParts) {
      hdrTmpl.payloadSize = mPartPerSP ? partsSP[hdrTmpl.splitPayloadIndex].size : link.getNextHBFSize();
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
        firstOrbit = hdrTmpl.firstTForbit = hbfU.getIRTF(tfid).orbit; // will be picked for the following parts
      }
      o2::header::Stack headerStack{hdrTmpl, o2::framework::DataProcessingHeader{mTFCounter}};
      memcpy(hdMessage->GetData(), headerStack.data(), headerStack.size());
      hdrTmpl.splitPayloadIndex++; // prepare for next

      addPart(std::move(hdMessage), std::move(plMessage), fmqChannel);
    }
    LOGF(DEBUG, "Added %d parts for TF#%d(%d in iteration %d) of %s/%s/0x%u", hdrTmpl.splitPayloadParts, mTFCounter, tfID,
         mLoopsDone, link.origin.as<std::string>(), link.description.as<std::string>(), link.subspec);
  }

  // send sTF acknowledge message
  {
    STFHeader stfHeader{mTFCounter, firstOrbit, 0};
    o2::header::DataHeader stfDistDataHeader(gDataDescSubTimeFrame, o2::header::gDataOriginFLP, 0, sizeof(STFHeader), 0, 1);
    stfDistDataHeader.payloadSerializationMethod = o2h::gSerializationMethodNone;
    stfDistDataHeader.firstTForbit = stfHeader.mFirstOrbit;
    stfDistDataHeader.tfCounter = mTFCounter;
    const auto fmqChannel = findOutputChannel(stfDistDataHeader);
    if (!fmqChannel.empty()) { // no output channel
      auto fmqFactory = device->GetChannel(fmqChannel, 0).Transport();
      o2::header::Stack headerStackSTF{stfDistDataHeader, o2::framework::DataProcessingHeader{mTFCounter}};
      auto hdMessageSTF = fmqFactory->CreateMessage(hstackSize, fair::mq::Alignment{64});
      auto plMessageSTF = fmqFactory->CreateMessage(stfDistDataHeader.payloadSize, fair::mq::Alignment{64});
      memcpy(hdMessageSTF->GetData(), headerStackSTF.data(), headerStackSTF.size());
      memcpy(plMessageSTF->GetData(), &stfHeader, sizeof(STFHeader));
      addPart(std::move(hdMessageSTF), std::move(plMessageSTF), fmqChannel);
    }
  }

<<<<<<< HEAD
  if (mTFCounter) { // delay sending
    usleep(mDelayUSec);
  }
  for (auto& msgIt : messagesPerRoute) {
    LOG(INFO) << "Sending " << msgIt.second->Size() / 2 << " parts to channel " << msgIt.first;
    device->Send(*msgIt.second.get(), msgIt.first);
  }
  mTimer[TimerTotal].Stop();
=======

 private:
  int mLoop = 0;                  // once last TF reached, loop while mLoop>=0
  uint32_t mTFCounter = 0;        // TFId accumulator (accounts for looping)
  uint32_t mDelayUSec = 0;        // Delay in microseconds between TFs
  uint32_t mMinTFID = 0;          // 1st TF to extract
  uint32_t mMaxTFID = 0xffffffff; // last TF to extrct
  size_t mLoopsDone = 0;
  size_t mSentSize = 0;
  size_t mSentMessages = 0;
  bool mPartPerSP = true;                          // fill part per superpage
  bool mDone = false;                              // processing is over or not
  std::string mRawChannelName = "";                // name of optional non-DPL channel
  std::unique_ptr<o2::raw::RawFileReader> mReader; // matching engine
>>>>>>> ccc611077 (Fix chipID for alignable entry)

  LOGF(INFO, "Sent payload of %zu bytes in %zu parts in %zu messages for TF %d | Timing (total/IO): %.3e / %.3e", tfSize, tfNParts,
       messagesPerRoute.size(), mTFCounter, mTimer[TimerTotal].CpuTime() - tTotStart, mTimer[TimerIO].CpuTime() - tIOStart);

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
      auto conf = o2::raw::RawFileReader::parseInput(rinp.inifile);
      for (const auto& entry : conf) {
        const auto& ordescard = entry.first;
        if (!entry.second.empty()) { // origin and decription for files to process
          spec.outputs.emplace_back(o2f::OutputSpec(o2f::ConcreteDataTypeMatcher{std::get<0>(ordescard), std::get<1>(ordescard)}));
        }
      }
    }
    // add output for DISTSUBTIMEFRAME
    spec.outputs.emplace_back(o2f::OutputSpec{{"stfDist"}, o2::header::gDataOriginFLP, RawReaderSpecs::gDataDescSubTimeFrame, 0});
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
    spec.options = {o2f::ConfigParamSpec{"channel-config", o2f::VariantType::String, rinp.rawChannelConfig, {"Out-of-band channel config"}}};
    rinp.rawChannelConfig = rinp.rawChannelConfig.substr(nameStart, nameEnd - nameStart);
    LOG(INFO) << "Will send output to non-DPL channel " << rinp.rawChannelConfig;
  }

  spec.algorithm = o2f::adaptFromTask<RawReaderSpecs>(rinp);

  return spec;
}

o2f::WorkflowSpec o2::raw::getRawFileReaderWorkflow(ReaderInp& rinp)
{
  o2f::WorkflowSpec specs;
  specs.emplace_back(getReaderSpec(rinp));
  return specs;
}
