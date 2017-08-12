// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ChannelMatching.h"
#include "Framework/DataProcessingDevice.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/MetricsService.h"
#include <fairmq/FairMQParts.h>

using namespace o2::framework;

using DataHeader = o2::Header::DataHeader;

namespace o2 {
namespace framework {

DataProcessingDevice::DataProcessingDevice(const DeviceSpec &spec,
                                           ServiceRegistry &registry)
: mInit{spec.init},
  mProcess{spec.process},
  mError{spec.onError},
  mChannels{spec.channels},
  mAllocator{this, &mContext, spec.outputs},
  mRelayer{spec.inputs, spec.forwards, registry.get<MetricsService>()},
  mInputs{spec.inputs},
  mForwards{spec.forwards},
  mServiceRegistry{registry},
  mErrorCount{0},
  mProcessingCount{0}
{
}

// We need to register all the input channels..
void DataProcessingDevice::InitTask() {
  LOG(DEBUG) << "DataProcessingDevice::InitTask::START";
  size_t inChannels = 0;
  for (auto &channel : mChannels) {
    if (channel.method == Bind) {
      continue;
    }
    inChannels++;
    OnData(channel.name.c_str(), &DataProcessingDevice::HandleData);
  }
  if (!inChannels) {
    LOG(ERROR) << "DataProcessingDevice should have at least one input channel";
  }
  if (mInit) {
    mInit(mConfigRegistry, mServiceRegistry);
  }
  LOG(DEBUG) << "DataProcessingDevice::InitTask::END";
}

// This is the inner loop of our framework
// This should:
// - Check what message we got and which argument it is.
// - FIXME: make up some timeframe id for now.
// - Insert the header and the payload in the multimap.
// - Check if any of the timeframes has all the required messages
// - Invoke the process callback, if this is the case
// - Forward the parts to the next stage
bool
DataProcessingDevice::HandleData(FairMQParts &parts, int /*index*/) {
  auto &metricsService = mServiceRegistry.get<MetricsService>();
  // FIXME: I need to construct the DataRefs
  metricsService.post("inputs/parts/total", (int)parts.Size());

  for (size_t i = 0; i < parts.Size() ; ++i) {
    LOG(DEBUG) << " part " << i << " is " << parts.At(i)->GetSize() << "bytes";
  }

  if (parts.Size() % 2) {
    error("Parts should come in couples. Dropping it.");
    return true;
  }

  // We relay execution to make sure we have a complete set of parts 
  // available.
  for (size_t pi = 0; pi < (parts.Size()/2); ++pi) {
    auto headerIndex = 2*pi;
    auto payloadIndex = 2*pi+1;
    assert(payloadIndex < parts.Size());
    auto relayed = mRelayer.relay(std::move(parts.At(headerIndex)),
                                  std::move(parts.At(payloadIndex)));
    if (relayed == DataRelayer::WillNotRelay) {
      error("Unable to relay part.");
      return true;
    }
    LOG(DEBUG) << "Relaying part idx: " << headerIndex;
    assert(mRelayer.getCacheSize() > 0);
  }

  // Notice that compleated represent ONE set of complete
  // inputs because an incoming message can complete only 
  // one pending set of inputs.
  // FIXME: is the above true?!?!?!
  LOG(DEBUG) << "Getting parts to process";
  auto completed = mRelayer.getReadyToProcess();

  metricsService.post("inputs/relayed/pending", (int)mRelayer.getCacheSize());
  if (completed.readyInputs.empty()) {
    metricsService.post("inputs/relayed/incomplete", 1);
    return true;
  }

  assert(!mInputs.empty());
  if (completed.readyInputs.size() != mInputs.size()) {
    std::ostringstream err;
    err << "Number of parts (" << completed.readyInputs.size()
        << ") should match the declared inputs ("
        << mInputs.size() << "). Dropping.";
    error(err.str().c_str());
    return true;
  }

  std::vector<DataRef> inputs;

  for (auto &readyParts : completed.readyInputs) {
    assert(readyParts.header->GetData());
    assert(readyParts.header->GetSize() == 80);
    assert(readyParts.payload->GetData());
    inputs.push_back(std::move(DataRef{nullptr,
                               reinterpret_cast<char *>(readyParts.header->GetData()),
                               reinterpret_cast<char *>(readyParts.payload->GetData())}));
  }

  // The above check should enforce this, which
  // should never happen.
  assert(inputs.size() == mInputs.size());
  mContext.clear();

  // If we are here, we have a complete set of inputs,
  // therefore we dispatch the calculation, if available.
  // After the computation is done, we get the output message
  // context and we send the messages we find in it.
  try {
    if (mProcess) {
      LOG(DEBUG) << "PROCESSING:START";
      metricsService.post("dataprocessing/process", mProcessingCount++);
      mProcess(inputs, mServiceRegistry, mAllocator);
      LOG(DEBUG) << "PROCESSING:END";
      for (auto &message : mContext) {
        metricsService.post("output/parts", message.parts.Size());
        assert(message.parts.Size() == 2);
        FairMQParts outParts = std::move(message.parts);
        assert(message.parts.Size() == 0);
        assert(outParts.Size() == 2);
        assert(outParts.At(0)->GetSize() == 80);
        this->Send(outParts, message.channel, message.index);
        assert(outParts.Size() == 2);
      }
    }
  } catch(std::exception &e) {
    LOG(DEBUG) << "Exception caught" << e.what() << std::endl;
    if (mError) {
      metricsService.post("error", 1);
      mError(inputs, mServiceRegistry, e);
    }
  }

  // Do the forwarding. We check if any of the inputs
  // should be forwarded elsewhere.
  // FIXME: do it in a smarter way than O(N^2)
  LOG(DEBUG) << "FORWARDING:START";
  for (auto &input : completed.readyInputs) {
    assert(input.header);
    assert(input.header->GetSize() == 80);
    //auto h = o2::Header::get<DataHeader>(input.header->GetData());
    auto h = reinterpret_cast<DataHeader*>(input.header->GetData());
    if (!h) {
      error("Header is not a DataHeader?");
      continue;
    }
    for (auto forward : mForwards) {
      // This should check if any of the parts can be forwarded and
      // where. 
      if (strncmp(h->magicString, "O2O2", 4)) {
        error("Could not find magic string");
      }
      LOG(DEBUG) << "Input part content";
      LOG(DEBUG) << h->dataOrigin.str;
      LOG(DEBUG) << h->dataDescription.str;
      LOG(DEBUG) << h->subSpecification;
      if (DataSpecUtils::match(forward.second, h->dataOrigin,
                               h->dataDescription,
                               h->subSpecification)) {
        LOG(DEBUG) << "Forwarding data to " << forward.first;
        FairMQParts forwardedParts;
        forwardedParts.AddPart(std::move(input.header));
        forwardedParts.AddPart(std::move(input.payload));
        // FIXME: this should use a correct subchannel
        this->Send(forwardedParts, forward.first, 0);
      }
    }
  }
  LOG(DEBUG) << "FORWARDING:END";

  // Once we are done, we can start processing a new set of inputs.
  inputs.clear();
  return true;
}

void
DataProcessingDevice::error(const char *msg) {
  LOG(ERROR) << msg;
  mErrorCount++;
  mServiceRegistry.get<MetricsService>().post("dataprocessing/errors", mErrorCount);
}

} // namespace framework
} // namespace o2
