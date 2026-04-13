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
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ControlService.h"
#include "Framework/OutputRoute.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/RateLimiter.h"
#include "Headers/DataHeaderHelpers.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <TStopwatch.h>
#include <fairmq/Device.h>
#include <fairmq/Parts.h>
#include "TFReaderSpec.h"
#include "TFReaderDD/SubTimeFrameFileReader.h"
#include "TFReaderDD/SubTimeFrameFile.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/FileFetcher.h"
#include "CommonUtils/FIFO.h"
#include "CommonUtils/IRFrameSelector.h"
#include "DataFormatsParameters/AggregatedRunInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonConstants/LHCConstants.h"
#include "Algorithm/RangeTokenizer.h"
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <string>
#include <climits>
#include <regex>
#include <deque>
#include <chrono>
#include <thread>

using namespace o2::rawdd;
using namespace std::chrono_literals;
using DetID = o2::detectors::DetID;
namespace o2f = o2::framework;
namespace o2h = o2::header;

class TFReaderSpec : public o2f::Task
{
 public:
  struct SubSpecCount {
    uint32_t defSubSpec = 0xdeadbeef;
    int count = -1;
  };

  using TFMap = std::unordered_map<std::string, std::unique_ptr<fair::mq::Parts>>; // map of channel / TFparts

  explicit TFReaderSpec(const TFReaderInp& rinp);
  void init(o2f::InitContext& ic) final;
  void run(o2f::ProcessingContext& ctx) final;
  void endOfStream(o2f::EndOfStreamContext& ec) final;

 private:
  void loadRunTimeSpans(const std::string& flname);
  void runTimeRangesToIRFrameSelector(int runNumber);
  void stopProcessing(o2f::ProcessingContext& ctx);
  void TFBuilder();

 private:
  fair::mq::Device* mDevice = nullptr;
  std::vector<o2f::OutputRoute> mOutputRoutes;
  std::unique_ptr<o2::utils::FileFetcher> mFileFetcher;
  o2::utils::FIFO<std::unique_ptr<TFMap>> mTFQueue{}; // queued TFs
  //  std::unordered_map<o2h::DataIdentifier, SubSpecCount, std::hash<o2h::DataIdentifier>> mSeenOutputMap;
  std::unordered_map<o2h::DataIdentifier, SubSpecCount> mSeenOutputMap;
  std::map<int, std::vector<std::pair<long, long>>> mRunTimeRanges;
  o2::utils::IRFrameSelector mIRFrameSelector; // optional IR frames selector
  int mConvRunTimeRangesToOrbits = -1;         // not defined yet
  int mSentTFCounter = 0;
  int mAccTFCounter = 0;
  int mTFBuilderCounter = 0;
  int mNWaits = 0;
  int mTFLength = 32;
  long mTotalWaitTime = 0;
  size_t mSelIDEntry = 0; // next TFID to select from the mInput.tfIDs (if non-empty)
  bool mRunning = false;
  bool mWaitSendingLast = false;
  TFReaderInp mInput; // command line inputs
  std::thread mTFBuilderThread{};
};

//___________________________________________________________
TFReaderSpec::TFReaderSpec(const TFReaderInp& rinp) : mInput(rinp)
{
  for (const auto& hd : rinp.hdVec) {
    mSeenOutputMap[o2h::DataIdentifier{hd.dataDescription.str, hd.dataOrigin.str}].defSubSpec = hd.subSpecification;
  }
}

//___________________________________________________________
void TFReaderSpec::init(o2f::InitContext& ic)
{
  mInput.tfIDs = o2::RangeTokenizer::tokenize<int>(ic.options().get<std::string>("select-tf-ids"));
  mInput.maxTFs = ic.options().get<int>("max-tf");
  mInput.maxTFs = mInput.maxTFs > 0 ? mInput.maxTFs : 0x7fffffff;
  mInput.maxTFsPerFile = ic.options().get<int>("max-tf-per-file");
  mInput.maxTFsPerFile = mInput.maxTFsPerFile > 0 ? mInput.maxTFsPerFile : 0x7fffffff;
  mInput.maxTFCache = std::max(1, ic.options().get<int>("max-cached-tf"));
  mInput.maxFileCache = std::max(1, ic.options().get<int>("max-cached-files"));
  if (!mInput.fileRunTimeSpans.empty()) {
    loadRunTimeSpans(mInput.fileRunTimeSpans);
  }
  mFileFetcher = std::make_unique<o2::utils::FileFetcher>(mInput.inpdata, mInput.tffileRegex, mInput.remoteRegex, mInput.copyCmd, mInput.copyDir);
  mFileFetcher->setMaxFilesInQueue(mInput.maxFileCache);
  mFileFetcher->setMaxLoops(mInput.maxLoops);
  mFileFetcher->setFailThreshold(ic.options().get<float>("fetch-failure-threshold"));
  mFileFetcher->start();
}

//___________________________________________________________
void TFReaderSpec::run(o2f::ProcessingContext& ctx)
{
  if (!mDevice) {
    mDevice = ctx.services().get<o2f::RawDeviceService>().device();
    mOutputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs; // copy!!!
    // start TFBuilder thread
    mRunning = true;
    mTFBuilderThread = std::thread(&TFReaderSpec::TFBuilder, this);
  }
  static auto tLastTF = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  auto device = ctx.services().get<o2f::RawDeviceService>().device();
  assert(device);
  if (device != mDevice) {
    throw std::runtime_error(fmt::format("FMQDevice has changed, old={} new={}", fmt::ptr(mDevice), fmt::ptr(device)));
  }
  if (mInput.tfRateLimit == -999) {
    mInput.tfRateLimit = std::stoi(device->fConfig->GetValue<std::string>("timeframes-rate-limit"));
  }
  auto acknowledgeOutput = [this](fair::mq::Parts& parts, bool verbose = false) {
    int np = parts.Size();
    size_t dsize = 0, dsizeTot = 0, nblocks = 0;
    const o2h::DataHeader* hdPrev = nullptr;
    for (int ip = 0; ip < np; ip += 2) {
      const auto& msgh = parts[ip];
      const auto* hd = o2h::get<o2h::DataHeader*>(msgh.GetData());
      const auto* dph = o2h::get<o2f::DataProcessingHeader*>(msgh.GetData());
      if (verbose && mInput.verbosity > 0) {
        LOGP(info, "Acknowledge: part {}/{} {}/{}/{:#x} size:{} split {}/{}", ip, np, hd->dataOrigin.as<std::string>(), hd->dataDescription.as<std::string>(), hd->subSpecification, msgh.GetSize() + parts[ip + 1].GetSize(), hd->splitPayloadIndex, hd->splitPayloadParts);
      }
      if (hd->splitPayloadIndex == 0) { // check the 1st one only
        auto& entry = this->mSeenOutputMap[{hd->dataDescription.str, hd->dataOrigin.str}];
        if (entry.count != this->mSentTFCounter) {
          if (verbose && hdPrev) { // report previous partition size
            LOGP(info, "Block:{} {}/{} with size {}", nblocks, hdPrev->dataOrigin.as<std::string>(), hdPrev->dataDescription.as<std::string>(), dsize);
          }
          dsizeTot += dsize;
          dsize = 0;
          entry.count = this->mSentTFCounter; // acknowledge identifier seen in the data
          LOG(debug) << "Found a part " << ip << " of " << np << " | " << hd->dataOrigin.as<std::string>() << "/" << hd->dataDescription.as<std::string>()
                     << "/" << hd->subSpecification << " part " << hd->splitPayloadIndex << " of " << hd->splitPayloadParts << " for TF " << this->mSentTFCounter;
          nblocks++;
        }
      }
      hdPrev = hd;
      dsize += msgh.GetSize() + parts[ip + 1].GetSize();
    }
    // last part
    dsizeTot += dsize;
    if (verbose && hdPrev) {
      LOGP(info, "Block:{} {}/{} with size {}", nblocks, hdPrev->dataOrigin.as<std::string>(), hdPrev->dataDescription.as<std::string>(), dsize);
    }
    return dsizeTot;
  };

  auto findOutputChannel = [&ctx, this](o2h::DataHeader& h, size_t tslice) {
    if (!this->mInput.rawChannelConfig.empty()) {
      return std::string{this->mInput.rawChannelConfig};
    } else {
      auto& outputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs;
      for (auto& oroute : outputRoutes) {
        LOG(debug) << "comparing with matcher to route " << oroute.matcher << " TSlice:" << oroute.timeslice;
        if (o2f::DataSpecUtils::match(oroute.matcher, h.dataOrigin, h.dataDescription, h.subSpecification) && ((tslice % oroute.maxTimeslices) == oroute.timeslice)) {
          LOG(debug) << "picking the route:" << o2f::DataSpecUtils::describe(oroute.matcher) << " channel " << oroute.channel;
          return std::string{oroute.channel};
        }
      }
    }
    auto& outputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs;
    LOGP(error, "Failed to find output channel for {}/{}/{} @ timeslice {}", h.dataOrigin, h.dataDescription, h.subSpecification, h.tfCounter);
    for (auto& oroute : outputRoutes) {
      LOGP(info, "Available route  route {}", o2f::DataSpecUtils::describe(oroute.matcher));
    }
    return std::string{};
  };
  auto setTimingInfo = [&ctx](TFMap& msgMap) {
    auto& timingInfo = ctx.services().get<o2::framework::TimingInfo>();
    const auto* dataptr = (*msgMap.begin()->second.get())[0].GetData();
    const auto* hd0 = o2h::get<o2h::DataHeader*>(dataptr);
    const auto* dph = o2h::get<o2f::DataProcessingHeader*>(dataptr);
    timingInfo.firstTForbit = hd0->firstTForbit;
    timingInfo.creation = dph->creation;
    timingInfo.tfCounter = hd0->tfCounter;
    timingInfo.runNumber = hd0->runNumber;
  };

  auto addMissingParts = [this, &findOutputChannel](TFMap& msgMap) {
    // at least the 1st header is guaranteed to be filled by the reader, use it for extra info
    const auto* dataptr = (*msgMap.begin()->second.get())[0].GetData();
    const auto* hd0 = o2h::get<o2h::DataHeader*>(dataptr);
    const auto* dph = o2h::get<o2f::DataProcessingHeader*>(dataptr);
    for (auto& out : this->mSeenOutputMap) {
      if (out.second.count == this->mSentTFCounter) { // was seen in the data
        continue;
      }
      LOG(debug) << "Adding dummy output for " << out.first.dataOrigin.as<std::string>() << "/" << out.first.dataDescription.as<std::string>()
                 << "/" << out.second.defSubSpec << " for TF " << this->mSentTFCounter;
      o2h::DataHeader outHeader(out.first.dataDescription, out.first.dataOrigin, out.second.defSubSpec, 0);
      outHeader.payloadSerializationMethod = o2h::gSerializationMethodNone;
      outHeader.firstTForbit = hd0->firstTForbit;
      outHeader.tfCounter = hd0->tfCounter;
      outHeader.runNumber = hd0->runNumber;
      const auto fmqChannel = findOutputChannel(outHeader, dph->startTime);
      if (fmqChannel.empty()) { // no output channel
        continue;
      }
      auto fmqFactory = this->mDevice->GetChannel(fmqChannel, 0).Transport();
      o2h::Stack headerStack{outHeader, *dph};
      auto hdMessage = fmqFactory->CreateMessage(headerStack.size(), fair::mq::Alignment{64});
      auto plMessage = fmqFactory->CreateMessage(0, fair::mq::Alignment{64});
      memcpy(hdMessage->GetData(), headerStack.data(), headerStack.size());
      fair::mq::Parts* parts = msgMap[fmqChannel].get();
      if (!parts) {
        msgMap[fmqChannel] = std::make_unique<fair::mq::Parts>();
        parts = msgMap[fmqChannel].get();
      }
      parts->AddPart(std::move(hdMessage));
      parts->AddPart(std::move(plMessage));
    }
  };

  while (1) {
    if (mTFQueue.size()) {
      static o2f::RateLimiter limiter;
      limiter.check(ctx, mInput.tfRateLimit, mInput.minSHM);

      auto tfPtr = std::move(mTFQueue.front());
      mTFQueue.pop();
      if (!tfPtr) {
        LOG(error) << "Builder provided nullptr TF pointer";
        continue;
      }
      setTimingInfo(*tfPtr.get());
      size_t nparts = 0, dataSize = 0;
      if (mInput.sendDummyForMissing) {
        for (auto& msgIt : *tfPtr.get()) { // complete with empty output for the specs which were requested but were not seen in the data
          acknowledgeOutput(*msgIt.second.get(), true);
        }
        addMissingParts(*tfPtr.get());
      }

      auto tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      auto tDiff = tNow - tLastTF;
      if (mSentTFCounter && tDiff < mInput.delay_us) {
        std::this_thread::sleep_for(std::chrono::microseconds((size_t)(mInput.delay_us - tDiff))); // respect requested delay before sending
      }
      for (auto& msgIt : *tfPtr.get()) {
        size_t szPart = acknowledgeOutput(*msgIt.second.get(), false);
        dataSize += szPart;
        const auto* hd = o2h::get<o2h::DataHeader*>((*msgIt.second.get())[0].GetData());
        nparts += msgIt.second->Size() / 2;
        device->Send(*msgIt.second.get(), msgIt.first);
      }
      // FIXME: this is to pretend we did send some messages via DPL.
      //        we should really migrate everything to use FairMQDeviceProxy,
      //        however this is a small enough hack for now.
      ctx.services().get<o2f::MessageContext>().fakeDispatch();
      tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      LOGP(info, "Sent TF {} of size {} with {} parts, {:.4f} s elapsed from previous TF., WaitSending={}", mSentTFCounter, dataSize, nparts, mSentTFCounter ? double(tNow - tLastTF) * 1e-6 : 0., mWaitSendingLast);
      tLastTF = tNow;
      ++mSentTFCounter;

      while (mTFQueue.size() == 0 && mWaitSendingLast) {
        usleep(10000);
      }
      break;
    }
    if (!mRunning) { // no more TFs will be provided
      stopProcessing(ctx);
      break;
    }
    //    usleep(5000); // wait 5ms for new TF to be built
  }
  if (mSentTFCounter >= mInput.maxTFs || (!mTFQueue.size() && !mRunning)) { // done
    stopProcessing(ctx);
  }
}

//____________________________________________________________
void TFReaderSpec::endOfStream(o2f::EndOfStreamContext& ec)
{
  if (mFileFetcher) {
    mFileFetcher->stop();
    mFileFetcher.reset();
  }
  if (mTFBuilderThread.joinable()) {
    mTFBuilderThread.join();
  }
}

//___________________________________________________________
void TFReaderSpec::stopProcessing(o2f::ProcessingContext& ctx)
{
  static bool stopDone = false;
  if (stopDone) {
    return;
  }
  stopDone = true;
  LOGP(info, "{} TFs in {}  loops were sent, spent {:.2} s in {} data waiting states", mSentTFCounter, mFileFetcher->getNLoops(), 1e-6 * mTotalWaitTime, mNWaits);
  mRunning = false;
  if (mFileFetcher) {
    mFileFetcher->stop();
    mFileFetcher.reset();
  }
  if (mTFBuilderThread.joinable()) {
    mTFBuilderThread.join();
  }
  if (!mInput.rawChannelConfig.empty()) {
    auto device = ctx.services().get<o2f::RawDeviceService>().device();
    o2f::SourceInfoHeader exitHdr;
    exitHdr.state = o2f::InputChannelState::Completed;
    o2h::DataHeader dh = o2h::DataHeader(o2h::gDataDescriptionInfo, o2h::gDataOriginAny, 0, 0);
    try {
      dh.runNumber = strtoul(device->fConfig->GetProperty<std::string>("runNumber", "").c_str(), nullptr, 10);
    } catch (...) {
    }
    const auto exitStack = o2h::Stack(dh, o2f::DataProcessingHeader(), exitHdr);
    auto fmqFactory = device->GetChannel(mInput.rawChannelConfig, 0).Transport();
    auto hdEOSMessage = fmqFactory->CreateMessage(exitStack.size(), fair::mq::Alignment{64});
    auto plEOSMessage = fmqFactory->CreateMessage(0, fair::mq::Alignment{64});
    memcpy(hdEOSMessage->GetData(), exitStack.data(), exitStack.size());
    fair::mq::Parts eosMsg;
    eosMsg.AddPart(std::move(hdEOSMessage));
    eosMsg.AddPart(std::move(plEOSMessage));
    device->Send(eosMsg, mInput.rawChannelConfig);
    LOG(info) << "Sent EoS message to " << mInput.rawChannelConfig;
  } else {
    ctx.services().get<o2f::ControlService>().endOfStream();
  }
  ctx.services().get<o2f::ControlService>().readyToQuit(o2f::QuitRequest::Me);
}

//____________________________________________________________
void TFReaderSpec::TFBuilder()
{
  // build TFs and add to the queue
  std::string tfFileName;
  auto sleepTime = std::chrono::microseconds(mInput.delay_us > 10000 ? mInput.delay_us : 10000);
  bool waitAcknowledged = false;
  long startWait = 0;
  while (mRunning && mDevice) {
    LOGP(debug, "mTFQueue.size()={} mWaitSendingLast = {}", mTFQueue.size(), mWaitSendingLast);
    if (mTFQueue.size() >= size_t(mInput.maxTFCache)) {
      mWaitSendingLast = false;
      std::this_thread::sleep_for(sleepTime);
      continue;
    }
    tfFileName = mFileFetcher ? mFileFetcher->getNextFileInQueue() : "";
    if (!mRunning ||
        (tfFileName.empty() && !mFileFetcher->isRunning()) ||
        mTFBuilderCounter >= mInput.maxTFs ||
        (!mInput.tfIDs.empty() && mSelIDEntry >= mInput.tfIDs.size())) {
      // stopped or no more files in the queue is expected or needed
      LOG(info) << "TFReader stops processing";
      if (mFileFetcher) {
        mFileFetcher->stop();
      }
      mRunning = false;
      mWaitSendingLast = false;
      break;
    }
    if (tfFileName.empty()) {
      if (!waitAcknowledged) {
        startWait = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
        waitAcknowledged = true;
      }
      std::this_thread::sleep_for(10ms); // wait for the files cache to be filled
      continue;
    }
    mWaitSendingLast = false;
    if (waitAcknowledged) {
      long waitTime = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count() - startWait;
      mTotalWaitTime += waitTime;
      if (++mNWaits > 1) {
        LOGP(warn, "Resuming reading after waiting for data {:.2} s (accumulated {:.2} s delay in {} waits)", 1e-6 * waitTime, 1e-6 * mTotalWaitTime, mNWaits);
      }
      waitAcknowledged = false;
      startWait = 0;
    }

    LOG(info) << "Processing file " << tfFileName;
    SubTimeFrameFileReader reader(tfFileName, mInput.detMask);
    size_t locID = 0;
    // try
    {
      while (mRunning && mTFBuilderCounter < mInput.maxTFs) {
        if (mTFQueue.size() >= size_t(mInput.maxTFCache)) {
          if (mTFQueue.size() > 1) {
            mWaitSendingLast = false;
          }
          std::this_thread::sleep_for(sleepTime);
          continue;
        }
        auto tf = reader.read(mDevice, mOutputRoutes, mInput.rawChannelConfig, mAccTFCounter, mInput.sup0xccdb, mInput.verbosity);
        bool acceptTF = true;
        if (tf) {
          if (mRunTimeRanges.size()) {
            const auto* dataptr = (*tf->begin()->second.get())[0].GetData();
            const auto* hd0 = o2h::get<o2h::DataHeader*>(dataptr);
            static int runNumberPrev = -1;
            if (runNumberPrev != hd0->runNumber) {
              runNumberPrev = hd0->runNumber;
              runTimeRangesToIRFrameSelector(runNumberPrev);
            }
            if (mIRFrameSelector.isSet()) {
              o2::InteractionRecord ir0(0, hd0->firstTForbit);
              o2::InteractionRecord ir1(o2::constants::lhc::LHCMaxBunches - 1, hd0->firstTForbit < 0xffffffff - (mTFLength - 1) ? hd0->firstTForbit + (mTFLength - 1) : 0xffffffff);
              auto irSpan = mIRFrameSelector.getMatchingFrames({ir0, ir1});
              acceptTF = (irSpan.size() > 0) ? !mInput.invertIRFramesSelection : mInput.invertIRFramesSelection;
              LOGP(info, "IRFrame selection contains {} frames for TF [{}] : [{}]: {}use this TF (selection inversion mode is {})",
                   irSpan.size(), ir0.asString(), ir1.asString(), acceptTF ? "" : "do not ", mInput.invertIRFramesSelection ? "ON" : "OFF");
            }
          }
          locID++;
          if (!mInput.tfIDs.empty() && acceptTF) {
            acceptTF = false;
            while ((mInput.tfIDs[mSelIDEntry] < mTFBuilderCounter) && (mSelIDEntry + 1) < mInput.tfIDs.size()) {
              mSelIDEntry++;
            }
            LOGP(info, "chec if mInput.tfIDs[{}]({}) == {}", mSelIDEntry, mInput.tfIDs[mSelIDEntry], mTFBuilderCounter);
            if (mInput.tfIDs[mSelIDEntry] == mTFBuilderCounter) {
              mWaitSendingLast = false;
              acceptTF = true;
              LOGP(info, "Retrieved TF#{} will be pushed as slice {} following user request", mTFBuilderCounter, mSelIDEntry);
            } else {
              LOGP(info, "Retrieved TF#{} will be discared following user request", mTFBuilderCounter);
            }
          }
          mTFBuilderCounter++;
        }
        if (mRunning && tf) {
          if (acceptTF) {
            mAccTFCounter++;
            mWaitSendingLast = true;
            mTFQueue.push(std::move(tf));
          }
        } else {
          break;
        }
        if (mInput.maxTFsPerFile > 0 && locID >= mInput.maxTFsPerFile) { // go to next file
          break;
        }
      }
      // remove already processed file from the queue, unless they are needed for further looping
      if (mFileFetcher) {
        mFileFetcher->popFromQueue(mFileFetcher->getNLoops() >= mInput.maxLoops);
      }
    }
  }
}

//_________________________________________________________
void TFReaderSpec::loadRunTimeSpans(const std::string& flname)
{
  std::ifstream inputFile(flname);
  if (!inputFile) {
    LOGP(fatal, "Failed to open selected run/timespans file {}", flname);
  }
  std::string line;
  size_t cntl = 0, cntr = 0;
  while (std::getline(inputFile, line)) {
    cntl++;
    for (char& ch : line) { // Replace semicolons and tabs with spaces for uniform processing
      if (ch == ';' || ch == '\t' || ch == ',') {
        ch = ' ';
      }
    }
    o2::utils::Str::trim(line);
    if (line.size() < 1 || line[0] == '#') {
      continue;
    }
    auto tokens = o2::utils::Str::tokenize(line, ' ');
    auto logError = [&cntl, &line]() { LOGP(error, "Expected format for selection is tripplet <run> <range_min> <range_max>, failed on line#{}: {}", cntl, line); };
    if (tokens.size() >= 3) {
      int run = 0;
      long rmin, rmax;
      try {
        run = std::stoi(tokens[0]);
        rmin = std::stol(tokens[1]);
        rmax = std::stol(tokens[2]);
      } catch (...) {
        logError();
        continue;
      }

      constexpr long ISTimeStamp = 1514761200000L;
      int convmn = rmin > ISTimeStamp ? 1 : 0, convmx = rmax > ISTimeStamp ? 1 : 0; // values above ISTimeStamp are timestamps (need to be converted to orbits)
      if (rmin > rmax) {
        LOGP(fatal, "Provided range limits are not in increasing order, entry is {}", line);
      }
      if (mConvRunTimeRangesToOrbits == -1) {
        if (convmn != convmx) {
          LOGP(fatal, "Provided range limits should be both consistent either with orbit number or with unix timestamp in ms, entry is {}", line);
        }
        mConvRunTimeRangesToOrbits = convmn; // need to convert to orbit if time
        LOGP(info, "Interpret selected time-spans input as {}", mConvRunTimeRangesToOrbits == 1 ? "timstamps(ms)" : "orbits");
      } else {
        if (mConvRunTimeRangesToOrbits != convmn || mConvRunTimeRangesToOrbits != convmx) {
          LOGP(fatal, "Provided range limits should are not consistent with previously determined {} input, entry is {}", mConvRunTimeRangesToOrbits == 1 ? "timestamps" : "orbits", line);
        }
      }

      mRunTimeRanges[run].emplace_back(rmin, rmax);
      cntr++;
    } else {
      logError();
    }
  }
  LOGP(info, "Read {} time-spans for {} runs from {}", cntr, mRunTimeRanges.size(), flname);
  inputFile.close();
}

//_________________________________________________________
void TFReaderSpec::runTimeRangesToIRFrameSelector(int runNumber)
{
  // convert entries in the runTimeRanges to IRFrameSelector, if needed, convert time to orbit
  mIRFrameSelector.clear();
  auto ent = mRunTimeRanges.find(runNumber);
  if (ent == mRunTimeRanges.end()) {
    LOGP(info, "RunTimeRanges selection was provided but run {} has no entries, all TFs will be processed", runNumber);
    return;
  }
  o2::parameters::AggregatedRunInfo rinfo;
  auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
  rinfo = o2::parameters::AggregatedRunInfo::buildAggregatedRunInfo(ccdb, runNumber);
  if (rinfo.runNumber != runNumber || rinfo.orbitsPerTF < 1) {
    LOGP(fatal, "failed to extract AggregatedRunInfo for run {}", runNumber);
  }
  mTFLength = rinfo.orbitsPerTF;
  std::vector<o2::dataformats::IRFrame> frames;
  for (const auto& rng : ent->second) {
    long orbMin = 0, orbMax = 0;
    if (mConvRunTimeRangesToOrbits > 0) {
      orbMin = rinfo.orbitSOR + (rng.first - rinfo.sor) / (o2::constants::lhc::LHCOrbitMUS * 0.001);
      orbMax = rinfo.orbitSOR + (rng.second - rinfo.sor) / (o2::constants::lhc::LHCOrbitMUS * 0.001);
    } else {
      orbMin = rng.first;
      orbMax = rng.second;
    }
    if (orbMin < 0) {
      orbMin = 0;
    }
    if (orbMax < 0) {
      orbMax = 0;
    }
    if (runNumber > 523897) {
      orbMin = (orbMin / rinfo.orbitsPerTF) * rinfo.orbitsPerTF;
      orbMax = (orbMax / rinfo.orbitsPerTF + 1) * rinfo.orbitsPerTF - 1;
    }
    LOGP(info, "TFs overlapping with orbits {}:{} will be {}", orbMin, orbMax, mInput.invertIRFramesSelection ? "rejected" : "selected");
    frames.emplace_back(o2::InteractionRecord{0, uint32_t(orbMin)}, o2::InteractionRecord{o2::constants::lhc::LHCMaxBunches, uint32_t(orbMax)});
  }
  mIRFrameSelector.setOwnList(frames, true);
}

//_________________________________________________________
o2f::DataProcessorSpec o2::rawdd::getTFReaderSpec(o2::rawdd::TFReaderInp& rinp)
{
  // check which inputs are present in files to read
  o2f::DataProcessorSpec spec;
  spec.name = "tf-reader";
  const DetID::mask_t DEFMask = DetID::getMask("ITS,TPC,TRD,TOF,PHS,CPV,EMC,HMP,MFT,MCH,MID,ZDC,FT0,FV0,FDD,CTP,FOC");
  rinp.detMask = DetID::getMask(rinp.detList) & DEFMask;
  rinp.detMaskRawOnly = DetID::getMask(rinp.detListRawOnly) & DEFMask;
  rinp.detMaskNonRawOnly = DetID::getMask(rinp.detListNonRawOnly) & DEFMask;
  if (rinp.rawChannelConfig.empty()) {
    // we don't know a priori what will be the content of the TF data, so we create all possible outputs
    for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
      if (rinp.detMask[id]) {
        if (!rinp.detMaskNonRawOnly[id]) {
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "RAWDATA"}});
          rinp.hdVec.emplace_back(o2h::DataHeader{"RAWDATA", DetID::getDataOrigin(id), 0xDEADBEEF, 0}); // in abcence of real data this will be sent
        }
        //
        if (rinp.detMaskRawOnly[id]) { // used asked to not open non-raw channels
          continue;
        }
        // in case detectors were processed on FLP
        if (id == DetID::CTP) {
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::OutputSpec{DetID::getDataOrigin(DetID::CTP), "LUMI", 0}});
          rinp.hdVec.emplace_back(o2h::DataHeader{"LUMI", DetID::getDataOrigin(DetID::CTP), 0, 0}); // in abcence of real data this will be sent
        }
        if (id == DetID::TOF) {
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(DetID::TOF), "CRAWDATA"}});
          rinp.hdVec.emplace_back(o2h::DataHeader{"CRAWDATA", DetID::getDataOrigin(DetID::TOF), 0xDEADBEEF, 0}); // in abcence of real data this will be sent
        } else if (id == DetID::FT0 || id == DetID::FV0 || id == DetID::FDD) {
          spec.outputs.emplace_back(o2f::OutputSpec{DetID::getDataOrigin(id), "DIGITSBC", 0});
          spec.outputs.emplace_back(o2f::OutputSpec{DetID::getDataOrigin(id), "DIGITSCH", 0});
          rinp.hdVec.emplace_back(o2h::DataHeader{"DIGITSBC", DetID::getDataOrigin(id), 0, 0}); // in abcence of real data this will be sent
          rinp.hdVec.emplace_back(o2h::DataHeader{"DIGITSCH", DetID::getDataOrigin(id), 0, 0}); // in abcence of real data this will be sent
        } else if (id == DetID::PHS) {
          spec.outputs.emplace_back(o2f::OutputSpec{DetID::getDataOrigin(id), "CELLS", 0});
          spec.outputs.emplace_back(o2f::OutputSpec{DetID::getDataOrigin(id), "CELLTRIGREC", 0});
          rinp.hdVec.emplace_back(o2h::DataHeader{"CELLS", DetID::getDataOrigin(id), 0, 0});       // in abcence of real data this will be sent
          rinp.hdVec.emplace_back(o2h::DataHeader{"CELLTRIGREC", DetID::getDataOrigin(id), 0, 0}); // in abcence of real data this will be sent
        } else if (id == DetID::CPV) {
          spec.outputs.emplace_back(DetID::getDataOrigin(id), "DIGITS", 0);
          spec.outputs.emplace_back(DetID::getDataOrigin(id), "DIGITTRIGREC", 0);
          spec.outputs.emplace_back(DetID::getDataOrigin(id), "RAWHWERRORS", 0);
          rinp.hdVec.emplace_back("DIGITS", DetID::getDataOrigin(id), 0, 0);       // in abcence of real data this will be sent
          rinp.hdVec.emplace_back("DIGITTRIGREC", DetID::getDataOrigin(id), 0, 0); // in abcence of real data this will be sent
          rinp.hdVec.emplace_back("RAWHWERRORS", DetID::getDataOrigin(id), 0, 0);  // in abcence of real data this will be sent
        } else if (id == DetID::EMC) {
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "CELLS"}});
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "CELLSTRGR"}});
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "DECODERERR"}});
          rinp.hdVec.emplace_back("CELLS", DetID::getDataOrigin(id), 0, 0);      // in abcence of real data this will be sent
          rinp.hdVec.emplace_back("CELLSTRGR", DetID::getDataOrigin(id), 0, 0);  // in abcence of real data this will be sent
          rinp.hdVec.emplace_back("DECODERERR", DetID::getDataOrigin(id), 0, 0); // in abcence of real data this will be sent
        } else if (id == DetID::FOC) {
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "PADLAYERS"}});
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "PIXELHITS"}});
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "PIXELCHIPS"}});
          spec.outputs.emplace_back(o2f::OutputSpec{o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "TRIGGERS"}});
          rinp.hdVec.emplace_back("PADLAYERS", DetID::getDataOrigin(id), 0, 0);  // in abcence of real data this will be sent
          rinp.hdVec.emplace_back("PIXELHITS", DetID::getDataOrigin(id), 0, 0);  // in abcence of real data this will be sent
          rinp.hdVec.emplace_back("PIXELCHIPS", DetID::getDataOrigin(id), 0, 0); // in abcence of real data this will be sent
          rinp.hdVec.emplace_back("TRIGGERS", DetID::getDataOrigin(id), 0, 0);   // in abcence of real data this will be sent
        }
      }
    }
    o2f::DataSpecUtils::updateOutputList(spec.outputs, o2f::OutputSpec{{"stfDist"}, o2h::gDataOriginFLP, o2h::gDataDescriptionDISTSTF, 0});
    if (!rinp.sup0xccdb) {
      o2f::DataSpecUtils::updateOutputList(spec.outputs, o2f::OutputSpec{{"stfDistCCDB"}, o2h::gDataOriginFLP, o2h::gDataDescriptionDISTSTF, 0xccdb});
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
    spec.options = {o2f::ConfigParamSpec{"channel-config", o2f::VariantType::String, rinp.rawChannelConfig, {"Out-of-band channel config"}}};
    rinp.rawChannelConfig = rinp.rawChannelConfig.substr(nameStart, nameEnd - nameStart);
    if (!rinp.metricChannel.empty()) {
      LOGP(alarm, "Cannot apply TF rate limiting when publishing to raw channel, limiting must be applied on the level of the input raw proxy");
      LOGP(alarm, R"(To avoid reader filling shm buffer use "--shm-throw-bad-alloc 0 --shm-segment-id 2")");
    }
  }
  spec.options.emplace_back(o2f::ConfigParamSpec{"select-tf-ids", o2f::VariantType::String, "", {"comma-separated list TF IDs to inject (from cumulative counter of TFs seen)"}});
  spec.options.emplace_back(o2f::ConfigParamSpec{"fetch-failure-threshold", o2f::VariantType::Float, 0.f, {"Fatil if too many failures( >0: fraction, <0: abs number, 0: no threshold)"}});
  spec.options.emplace_back(o2f::ConfigParamSpec{"max-tf", o2f::VariantType::Int, -1, {"max TF ID to process (<= 0 : infinite)"}});
  spec.options.emplace_back(o2f::ConfigParamSpec{"max-tf-per-file", o2f::VariantType::Int, -1, {"max TFs to process per raw-tf file (<= 0 : infinite)"}});
  spec.options.emplace_back(o2f::ConfigParamSpec{"max-cached-tf", o2f::VariantType::Int, 3, {"max TFs to cache in memory"}});
  spec.options.emplace_back(o2f::ConfigParamSpec{"max-cached-files", o2f::VariantType::Int, 3, {"max TF files queued (copied for remote source)"}});

  spec.algorithm = o2f::adaptFromTask<TFReaderSpec>(rinp);

  return spec;
}
