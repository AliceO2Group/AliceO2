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

#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include "RawFileReaderWorkflow.h" // not installed

#include <fairmq/FairMQDevice.h>

#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <string>

using namespace o2::raw;

namespace o2f = o2::framework;
namespace o2h = o2::header;

class rawReaderSpecs : public o2f::Task
{
 public:
  explicit rawReaderSpecs(const std::string& config, bool tfAsMessage = false, bool outPerRoute = true, int loop = 1, uint32_t delay_us = 0)
    : mLoop(loop), mHBFPerMessage(!tfAsMessage), mOutPerRoute(outPerRoute), mDelayUSec(delay_us), mReader(std::make_unique<o2::raw::RawFileReader>(config))
  {
    LOG(INFO) << "Number of loops over whole data requested: " << mLoop;
    if (mHBFPerMessage) {
      LOG(INFO) << "Every link TF will be sent as multipart of HBF messages";
    } else {
      LOG(INFO) << "HBF of single TF of each link will be sent as a single message";
    }
    LOG(INFO) << "A message per " << (mOutPerRoute ? "route" : "link") << " will be sent";
    LOG(INFO) << "Delay of " << mDelayUSec << " microseconds will be added between TFs";
  }

  void init(o2f::InitContext& ic) final
  {
    assert(mReader);
    mReader->init();
  }

  void run(o2f::ProcessingContext& ctx) final
  {
    assert(mReader);
    static size_t loopsDone = 0, sentSize = 0, sentMessages = 0;

    if (mDone) {
      return;
    }

    auto device = ctx.services().get<o2f::RawDeviceService>().device();
    assert(device);

    auto findOutputChannel = [&ctx](RawFileReader::LinkData& link) {
      auto outputRoutes = ctx.services().get<o2f::RawDeviceService>().spec().outputs;
      for (auto& oroute : outputRoutes) {
        LOG(INFO) << "comparing with matcher to route " << oroute.matcher << " TSlice:" << oroute.timeslice;
        if (o2f::DataSpecUtils::match(oroute.matcher, link.origin, link.description, link.subspec)) {
          link.fairMQChannel = oroute.channel;
          LOG(INFO) << "picking the route:" << o2f::DataSpecUtils::describe(oroute.matcher) << " channel " << oroute.channel;
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
    std::vector<std::unique_ptr<FairMQParts>> messagesPerLink;
    if (!mOutPerRoute) {
      messagesPerLink.resize(nlinks);
    }

    if (tfID >= mReader->getNTimeFrames()) {
      if (mReader->getNTimeFrames() && mLoop--) {
        tfID = 0;
        mReader->setNextTFToRead(tfID);
        loopsDone++;
        for (int il = 0; il < nlinks; il++) {
          mReader->getLink(il).nextBlock2Read = 0; // think about more elaborate looping scheme, e.g. incrementing the orbits in RDHs
        }
        LOG(INFO) << "Starting new loop " << loopsDone << " from the beginning of data";
      } else {
        LOGF(INFO, "Finished: payload of %zu bytes in %zu messages sent for %d TFs", sentSize, sentMessages, mTFIDaccum);
        ctx.services().get<o2f::ControlService>().endOfStream();
        ctx.services().get<o2f::ControlService>().readyToQuit(o2f::QuitRequest::Me);
        mDone = true;
        return;
      }
    }

    // read next time frame
    size_t tfNParts = 0, tfSize = 0;
    LOG(INFO) << "Reading TF#" << mTFIDaccum << " (" << tfID << " at iteration " << loopsDone << ')';

    for (int il = 0; il < nlinks; il++) {
      auto& link = mReader->getLink(il);

      if (link.fairMQChannel.empty() && !findOutputChannel(link)) { // no output channel
        continue;
      }

      o2h::DataHeader hdrTmpl(link.description, link.origin, link.subspec); // template with 0 size
      hdrTmpl.payloadSerializationMethod = o2h::gSerializationMethodNone;
      hdrTmpl.splitPayloadParts = mHBFPerMessage ? link.getNHBFinTF() : 1;

      while (hdrTmpl.splitPayloadIndex < hdrTmpl.splitPayloadParts) {

        tfSize += hdrTmpl.payloadSize = mHBFPerMessage ? link.getNextHBFSize() : link.getNextTFSize();
        o2::header::Stack headerStack{hdrTmpl, o2::framework::DataProcessingHeader{mTFIDaccum}};

        auto hdMessage = device->NewMessage(headerStack.size());
        memcpy(hdMessage->GetData(), headerStack.data(), headerStack.size());

        auto plMessage = device->NewMessage(hdrTmpl.payloadSize);
        auto bread = mHBFPerMessage ? link.readNextHBF(reinterpret_cast<char*>(plMessage->GetData())) : link.readNextTF(reinterpret_cast<char*>(plMessage->GetData()));
        if (bread != hdrTmpl.payloadSize) {
          LOG(ERROR) << "Link " << il << " read " << bread << " bytes instead of " << hdrTmpl.payloadSize
                     << " expected in TF=" << mTFIDaccum << " part=" << hdrTmpl.splitPayloadIndex;
        }
        // check if the RDH to send corresponds to expected orbit
        if (hdrTmpl.splitPayloadIndex == 0) {
          uint32_t hbOrbExpected = mReader->getOrbitMin() + tfID * mReader->getNominalHBFperTF();
          uint32_t hbOrbRead = o2::raw::RDHUtils::getHeartBeatOrbit(plMessage->GetData());
          if (hbOrbExpected != hbOrbRead) {
            LOGF(ERROR, "Expected orbit=%u but got %u for %d-th HBF in TF#%d of %s/%s/0x%u",
                 hbOrbExpected, hbOrbRead, hdrTmpl.splitPayloadIndex, tfID,
                 link.origin.as<std::string>(), link.description.as<std::string>(), link.subspec);
          }
        }
        FairMQParts* parts = nullptr;
        if (mOutPerRoute) {
          parts = messagesPerRoute[link.fairMQChannel].get(); // FairMQParts*
          if (!parts) {
            messagesPerRoute[link.fairMQChannel] = std::make_unique<FairMQParts>();
            parts = messagesPerRoute[link.fairMQChannel].get();
          }
        } else {                             // message per link
          parts = messagesPerLink[il].get(); // FairMQParts*
          if (!parts) {
            messagesPerLink[il] = std::make_unique<FairMQParts>();
            parts = messagesPerLink[il].get();
          }
        }
        parts->AddPart(std::move(hdMessage));
        parts->AddPart(std::move(plMessage));
        hdrTmpl.splitPayloadIndex++; // prepare for next
        tfNParts++;
      }
      LOGF(DEBUG, "Added %d parts for TF#%d(%d in iteration %d) of %s/%s/0x%u", hdrTmpl.splitPayloadParts, mTFIDaccum, tfID,
           loopsDone, link.origin.as<std::string>(), link.description.as<std::string>(), link.subspec);
    }

    if (mTFIDaccum) { // delay sending
      usleep(mDelayUSec);
    }

    if (mOutPerRoute) {
      for (auto& msgIt : messagesPerRoute) {
        LOG(INFO) << "Sending " << msgIt.second->Size() / 2 << " parts to channel " << msgIt.first;
        device->Send(*msgIt.second.get(), msgIt.first);
      }
    } else {
      for (int il = 0; il < nlinks; il++) {
        auto& link = mReader->getLink(il);
        auto* parts = messagesPerLink[il].get(); // FairMQParts*
        if (parts) {
          LOG(INFO) << "Sending " << parts->Size() / 2 << " parts to channel " << link.fairMQChannel << " for " << link.describe();
          device->Send(*parts, link.fairMQChannel);
        }
      }
    }

    LOGF(INFO, "Sent payload of %zu bytes in %zu parts in %zu messages for TF %d", tfSize, tfNParts,
         (mOutPerRoute ? messagesPerRoute.size() : messagesPerLink.size()), mTFIDaccum);
    sentSize += tfSize;
    sentMessages += tfNParts;

    mReader->setNextTFToRead(++tfID);
    ++mTFIDaccum;
  }

 private:
  int mLoop = 0;                                   // once last TF reached, loop while mLoop>=0
  size_t mTFIDaccum = 0;                           // TFId accumulator (accounts for looping)
  uint32_t mDelayUSec = 0;                         // Delay in microseconds between TFs
  bool mHBFPerMessage = true;                      // true: send TF as multipart of HBFs, false: single message per TF
  bool mOutPerRoute = true;                        // true: send 1 large output route, otherwise 1 outpur per link
  bool mDone = false;                              // processing is over or not
  std::unique_ptr<o2::raw::RawFileReader> mReader; // matching engine
};

o2f::DataProcessorSpec getReaderSpec(std::string config, bool tfAsMessage, bool outPerRoute, int loop, uint32_t delay_us)
{
  // check which inputs are present in files to read
  o2f::Outputs outputs;
  if (!config.empty()) {
    auto conf = o2::raw::RawFileReader::parseInput(config);
    for (const auto& entry : conf) {
      const auto& ordesc = entry.first;
      if (!entry.second.empty()) { // origin and decription for files to process
        outputs.emplace_back(o2f::OutputSpec(o2f::ConcreteDataTypeMatcher{ordesc.first, ordesc.second}));
      }
    }
  }
  return o2f::DataProcessorSpec{
    "raw-file-reader",
    o2f::Inputs{},
    outputs,
    o2f::AlgorithmSpec{o2f::adaptFromTask<rawReaderSpecs>(config, tfAsMessage, outPerRoute, loop, delay_us)},
    o2f::Options{}};
}

o2f::WorkflowSpec o2::raw::getRawFileReaderWorkflow(std::string inifile, bool tfAsMessage, bool outPerRoute, int loop, uint32_t delay_us)
{
  o2f::WorkflowSpec specs;
  specs.emplace_back(getReaderSpec(inifile, tfAsMessage, outPerRoute, loop, delay_us));
  return specs;
}
