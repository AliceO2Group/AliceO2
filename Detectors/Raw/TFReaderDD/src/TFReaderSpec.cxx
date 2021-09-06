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
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DetectorsCommonDataFormats/DetID.h"
#include <TStopwatch.h>
#include <fairmq/FairMQDevice.h>
#include "TFReaderSpec.h"
#include "TFReaderDD/SubTimeFrameFileReader.h"
#include "CommonUtils/FileFetcher.h"
#include "CommonUtils/FIFO.h"
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <string>
#include <climits>
#include <regex>
#include <deque>
#include <chrono>

using namespace o2::rawdd;
using namespace std::chrono_literals;
using DetID = o2::detectors::DetID;
namespace o2f = o2::framework;

class TFReaderSpec : public o2f::Task
{
 public:
  using TFMap = std::unordered_map<std::string, std::unique_ptr<FairMQParts>>; // map of channel / TFparts

  explicit TFReaderSpec(const TFReaderInp& rinp) : mInput(rinp) {}
  void init(o2f::InitContext& ic) final;
  void run(o2f::ProcessingContext& ctx) final;
  void endOfStream(o2f::EndOfStreamContext& ec) final;

 private:
  void stopProcessing(o2f::ProcessingContext& ctx);
  void TFBuilder();

 private:
  FairMQDevice* mDevice = nullptr;
  std::vector<o2f::OutputRoute> mOutputRoutes;
  std::unique_ptr<o2::utils::FileFetcher> mFileFetcher;
  o2::utils::FIFO<std::unique_ptr<TFMap>> mTFQueue{}; // queued TFs
  int mTFCounter = 0;
  int mTFBuilderCounter = 0;
  bool mRunning = false;
  TFReaderInp mInput; // command line inputs
  std::thread mTFBuilderThread{};
};

//___________________________________________________________
void TFReaderSpec::init(o2f::InitContext& ic)
{
  mFileFetcher = std::make_unique<o2::utils::FileFetcher>(mInput.inpdata, mInput.tffileRegex, mInput.remoteRegex, mInput.copyCmd);
  mFileFetcher->setMaxFilesInQueue(mInput.maxFileCache);
  mFileFetcher->setMaxLoops(mInput.maxLoops);
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
  static long deltaSending = 0; // time correction for sending
  auto device = ctx.services().get<o2f::RawDeviceService>().device();
  assert(device);
  if (device != mDevice) {
    throw std::runtime_error(fmt::format("FMQDevice has changed, old={} new={}", fmt::ptr(mDevice), fmt::ptr(device)));
  }

  while (1) {
    if (mTFCounter >= mInput.maxTFs) { // done
      stopProcessing(ctx);
      break;
    }
    if (mTFQueue.size()) {
      auto tfPtr = std::move(mTFQueue.front());
      mTFQueue.pop();
      if (!tfPtr) {
        LOG(ERROR) << "Builder provided nullptr TF pointer";
        continue;
      }
      auto tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      auto tDiff = tNow - tLastTF + 2 * deltaSending;
      if (mTFCounter && tDiff < mInput.delay_us) {
        usleep(mInput.delay_us - tDiff); // respect requested delay before sending
      }
      size_t nparts = 0;
      auto tSend = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      for (auto& msgIt : *tfPtr.get()) {
        nparts += msgIt.second->Size() / 2;
        device->Send(*msgIt.second.get(), msgIt.first);
      }
      tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      deltaSending = mTFCounter ? tNow - tLastTF : 0;
      LOGP(INFO, "Sent TF {} of {} parts, {:.4f} s elapsed from previous TF.", mTFCounter, nparts, double(deltaSending) * 1e-6);
      deltaSending -= mInput.delay_us;
      if (!mTFCounter || deltaSending < 0) {
        deltaSending = 0; // correction for next delay
      }
      tLastTF = tNow;
      ++mTFCounter;
      break;
    }
    if (!mRunning) { // no more TFs will be provided
      stopProcessing(ctx);
      break;
    }
    usleep(10); // wait for new TF to be built
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
  LOG(INFO) << mTFCounter << " TFs in " << mFileFetcher->getNLoops() << " loops were sent";
  mRunning = false;
  mFileFetcher->stop();
  mFileFetcher.reset();
  if (mTFBuilderThread.joinable()) {
    mTFBuilderThread.join();
  }
  ctx.services().get<o2f::ControlService>().endOfStream();
  ctx.services().get<o2f::ControlService>().readyToQuit(o2f::QuitRequest::Me);
}

//____________________________________________________________
void TFReaderSpec::TFBuilder()
{
  // build TFs and add to the queue
  std::string tfFileName;
  auto sleepTime = std::chrono::microseconds(mInput.delay_us > 10000 ? mInput.delay_us : 10000);
  while (mRunning && mDevice) {
    if (mTFQueue.size() >= size_t(mInput.maxTFCache)) {
      std::this_thread::sleep_for(sleepTime);
      continue;
    }
    tfFileName = mFileFetcher ? mFileFetcher->getNextFileInQueue() : "";
    if (!mRunning || (tfFileName.empty() && !mFileFetcher->isRunning()) || mTFBuilderCounter >= mInput.maxTFs) {
      // stopped or no more files in the queue is expected or needed
      LOG(INFO) << "TFBuilder stops processing";
      if (mFileFetcher) {
        mFileFetcher->stop();
      }
      mRunning = false;
      break;
    }
    if (tfFileName.empty()) {
      std::this_thread::sleep_for(10ms); // fait for the files cache to be filled
      continue;
    }
    LOG(INFO) << "Processing file " << tfFileName;
    SubTimeFrameFileReader reader(tfFileName);
    size_t locID = 0;
    //try
    {
      while (mRunning && mTFBuilderCounter < mInput.maxTFs) {
        if (mTFQueue.size() >= size_t(mInput.maxTFCache)) {
          std::this_thread::sleep_for(sleepTime);
          continue;
        }
        auto tf = reader.read(mDevice, mOutputRoutes, mInput.rawChannelConfig, mInput.verbosity);
        if (tf) {
          mTFBuilderCounter++;
        }
        if (mRunning && tf) {
          mTFQueue.push(std::move(tf));
          locID++;
        } else {
          break;
        }
      }
      // remove already processed file from the queue, unless they are needed for further looping
      if (mFileFetcher) {
        mFileFetcher->popFromQueue(mFileFetcher->getNLoops() >= mInput.maxLoops);
      }
    } /*catch (...) {
      LOGP(ERROR, "Error when building {}-th TF from file {}", locID, tfFileName);
      mFileFetcher->popFromQueue(mFileFetcher->getNLoops() >= mInput.maxLoops); // remove faile TF file
    } */
  }
}

//_________________________________________________________
o2f::DataProcessorSpec o2::rawdd::getTFReaderSpec(o2::rawdd::TFReaderInp& rinp)
{
  // check which inputs are present in files to read
  o2f::DataProcessorSpec spec;
  spec.name = "tf-reader";
  const DetID::mask_t DEFMask = DetID::getMask("ITS,TPC,TRD,TOF,PHS,CPV,EMC,HMP,MFT,MCH,MID,ZDC,FT0,FV0,FDD,CTP");
  DetID::mask_t detMask = DEFMask; //DetID::getMask(rinp.detList) & DEFMask; // RS TODO

  if (rinp.rawChannelConfig.empty()) {
    // we don't know a priori what will be the content of the TF data, so we create all possible outputs
    for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
      if (detMask[id]) {
        spec.outputs.emplace_back(o2f::OutputSpec(o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(id), "RAWDATA"}));
      }
    }
    // in case compessed TOF is present
    if (detMask[DetID::TOF]) {
      spec.outputs.emplace_back(o2f::OutputSpec(o2f::ConcreteDataTypeMatcher{DetID::getDataOrigin(DetID::TOF), "CRAWDATA"}));
    }
    spec.outputs.emplace_back(o2f::OutputSpec{{"stfDist"}, o2::header::gDataOriginFLP, "DISTSUBTIMEFRAME", 0});
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
    rinp.rawChannelConfig = rinp.rawChannelConfig.substr(nameStart, nameEnd - nameStart);
  }

  spec.algorithm = o2f::adaptFromTask<TFReaderSpec>(rinp);

  return spec;
}
