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

/// @file   GPUWorkflowPipeline.cxx
/// @author David Rohr

#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2Interface.h"
#include "GPUDataTypes.h"
#include "GPUSettings.h"
#include "GPUWorkflowInternal.h"

#include "Framework/WorkflowSpec.h" // o2::framework::mergeInputs
#include "Framework/DataRefUtils.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/SerializationMethods.h"
#include "Framework/Logger.h"
#include "Framework/CallbackService.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/RawDeviceService.h"

#include <fairmq/Device.h>
#include <fairmq/Channel.h>
#include <fairmq/States.h>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;
using namespace o2::dataformats;
using namespace o2::gpu::gpurecoworkflow_internals;

namespace o2::gpu
{

static const std::string GPURecoWorkflowSpec_FMQCallbackKey = "GPURecoWorkflowSpec_FMQCallbackKey";

struct pipelinePrepareMessage {
  static constexpr size_t MAGIC_WORD = 0X8473957353424134;
  size_t magicWord = MAGIC_WORD;
  DataProcessingHeader::StartTime timeSliceId;
  GPUSettingsTF tfSettings;
  size_t pointerCounts[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  size_t pointersTotal;
  bool flagEndOfStream;
};

void GPURecoWorkflowSpec::initPipeline(o2::framework::InitContext& ic)
{
  if (mSpecConfig.enableDoublePipeline == 1) {
    mPipeline->fmqDevice = ic.services().get<RawDeviceService>().device();
    mPipeline->fmqDevice->SubscribeToStateChange(GPURecoWorkflowSpec_FMQCallbackKey, [this](fair::mq::State s) { receiveFMQStateCallback(s); });
    mPolicyOrder = [this](o2::framework::DataProcessingHeader::StartTime timeslice) {
      std::unique_lock lk(mPipeline->completionPolicyMutex);
      mPipeline->completionPolicyNotify.wait(lk, [pipeline = mPipeline.get()] { return pipeline->pipelineSenderTerminating || !pipeline->completionPolicyQueue.empty(); });
      if (mPipeline->completionPolicyQueue.front() == timeslice) {
        mPipeline->completionPolicyQueue.pop();
        return true;
      }
      return false;
    };
    mPipeline->receiveThread = std::thread([this]() { RunReceiveThread(); });
    for (unsigned int i = 0; i < mPipeline->workers.size(); i++) {
      mPipeline->workers[i].thread = std::thread([this, i]() { RunWorkerThread(i); });
    }
  }
}

void GPURecoWorkflowSpec::RunWorkerThread(int id)
{
  LOG(debug) << "Running pipeline worker " << id;
  auto& workerContext = mPipeline->workers[id];
  while (!mPipeline->shouldTerminate) {
    GPURecoWorkflow_QueueObject* context;
    {
      std::unique_lock lk(workerContext.inputQueueMutex);
      workerContext.inputQueueNotify.wait(lk, [this, &workerContext]() { return mPipeline->shouldTerminate || !workerContext.inputQueue.empty(); });
      if (workerContext.inputQueue.empty()) {
        break;
      }
      context = workerContext.inputQueue.front();
      workerContext.inputQueue.pop();
    }
    context->jobReturnValue = runMain(nullptr, context->jobPtrs, context->jobOutputRegions, id, context->jobInputUpdateCallback.get());
    {
      std::lock_guard lk(context->jobFinishedMutex);
      context->jobFinished = true;
    }
    context->jobFinishedNotify.notify_one();
  }
}

void GPURecoWorkflowSpec::enqueuePipelinedJob(GPUTrackingInOutPointers* ptrs, GPUInterfaceOutputs* outputRegions, GPURecoWorkflow_QueueObject* context, bool inputFinal)
{
  {
    std::unique_lock lk(mPipeline->mayInjectMutex);
    mPipeline->mayInjectCondition.wait(lk, [this, context]() { return mPipeline->mayInject && mPipeline->mayInjectTFId == context->mTFId; });
    mPipeline->mayInjectTFId = mPipeline->mayInjectTFId + 1;
    mPipeline->mayInject = false;
  }
  context->jobSubmitted = true;
  context->jobInputFinal = inputFinal;
  context->jobPtrs = ptrs;
  context->jobOutputRegions = outputRegions;

  context->jobInputUpdateCallback = std::make_unique<GPUInterfaceInputUpdate>();

  if (!inputFinal) {
    context->jobInputUpdateCallback->callback = [context](GPUTrackingInOutPointers*& data, GPUInterfaceOutputs*& outputs) {
      std::unique_lock lk(context->jobInputFinalMutex);
      context->jobInputFinalNotify.wait(lk, [context]() { return context->jobInputFinal; });
      data = context->jobPtrs;
      outputs = context->jobOutputRegions;
    };
  }
  context->jobInputUpdateCallback->notifyCallback = [this]() {
    {
      std::lock_guard lk(mPipeline->mayInjectMutex);
      mPipeline->mayInject = true;
    }
    mPipeline->mayInjectCondition.notify_one();
  };

  mNextThreadIndex = (mNextThreadIndex + 1) % 2;

  {
    std::lock_guard lk(mPipeline->workers[mNextThreadIndex].inputQueueMutex);
    mPipeline->workers[mNextThreadIndex].inputQueue.emplace(context);
  }
  mPipeline->workers[mNextThreadIndex].inputQueueNotify.notify_one();
}

void GPURecoWorkflowSpec::finalizeInputPipelinedJob(GPUTrackingInOutPointers* ptrs, GPUInterfaceOutputs* outputRegions, GPURecoWorkflow_QueueObject* context)
{
  {
    std::lock_guard lk(context->jobInputFinalMutex);
    context->jobPtrs = ptrs;
    context->jobOutputRegions = outputRegions;
    context->jobInputFinal = true;
  }
  context->jobInputFinalNotify.notify_one();
}

int GPURecoWorkflowSpec::handlePipeline(ProcessingContext& pc, GPUTrackingInOutPointers& ptrs, GPURecoWorkflowSpec_TPCZSBuffers& tpcZSmeta, o2::gpu::GPUTrackingInOutZS& tpcZS, std::unique_ptr<GPURecoWorkflow_QueueObject>& context)
{
  mPipeline->runStarted = true;
  mPipeline->stateNotify.notify_all();

  auto* device = pc.services().get<RawDeviceService>().device();
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  if (mSpecConfig.enableDoublePipeline == 1) {
    std::unique_lock lk(mPipeline->queueMutex);
    mPipeline->queueNotify.wait(lk, [this] { return !mPipeline->pipelineQueue.empty(); });
    context = std::move(mPipeline->pipelineQueue.front());
    mPipeline->pipelineQueue.pop();
    lk.unlock();

    if (context->timeSliceId != tinfo.timeslice) {
      LOG(fatal) << "Prepare message for incorrect time frame received, time frames seem out of sync";
    }

    tpcZSmeta = std::move(context->tpcZSmeta);
    tpcZS = context->tpcZS;
    ptrs.tpcZS = &tpcZS;
  }
  if (mSpecConfig.enableDoublePipeline == 2) {
    auto prepareBuffer = pc.outputs().make<DataAllocator::UninitializedVector<char>>(Output{gDataOriginGPU, "PIPELINEPREPARE", 0}, 0u);

    size_t ptrsTotal = 0;
    const void* firstPtr = nullptr;
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        if (firstPtr == nullptr && ptrs.tpcZS->slice[i].count[j]) {
          firstPtr = ptrs.tpcZS->slice[i].zsPtr[j][0];
        }
        ptrsTotal += ptrs.tpcZS->slice[i].count[j];
      }
    }

    size_t prepareBufferSize = sizeof(pipelinePrepareMessage) + ptrsTotal * sizeof(size_t) * 4;
    std::vector<size_t> messageBuffer(prepareBufferSize / sizeof(size_t));
    pipelinePrepareMessage& preMessage = *(pipelinePrepareMessage*)messageBuffer.data();
    preMessage.magicWord = preMessage.MAGIC_WORD;
    preMessage.timeSliceId = tinfo.timeslice;
    preMessage.pointersTotal = ptrsTotal;
    preMessage.flagEndOfStream = false;
    memcpy((void*)&preMessage.tfSettings, (const void*)ptrs.settingsTF, sizeof(preMessage.tfSettings));

    size_t* ptrBuffer = messageBuffer.data() + sizeof(preMessage) / sizeof(size_t);
    size_t ptrsCopied = 0;
    int lastRegion = -1;
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        preMessage.pointerCounts[i][j] = ptrs.tpcZS->slice[i].count[j];
        for (unsigned int k = 0; k < ptrs.tpcZS->slice[i].count[j]; k++) {
          const void* curPtr = ptrs.tpcZS->slice[i].zsPtr[j][k];
          bool regionFound = lastRegion != -1 && (size_t)curPtr >= (size_t)mRegionInfos[lastRegion].ptr && (size_t)curPtr < (size_t)mRegionInfos[lastRegion].ptr + mRegionInfos[lastRegion].size;
          if (!regionFound) {
            for (unsigned int l = 0; l < mRegionInfos.size(); l++) {
              if ((size_t)curPtr >= (size_t)mRegionInfos[l].ptr && (size_t)curPtr < (size_t)mRegionInfos[l].ptr + mRegionInfos[l].size) {
                lastRegion = l;
                regionFound = true;
                break;
              }
            }
          }
          if (!regionFound) {
            LOG(fatal) << "Found a TPC ZS pointer outside of shared memory";
          }
          ptrBuffer[ptrsCopied + k] = (size_t)curPtr - (size_t)mRegionInfos[lastRegion].ptr;
          ptrBuffer[ptrsTotal + ptrsCopied + k] = ptrs.tpcZS->slice[i].nZSPtr[j][k];
          ptrBuffer[2 * ptrsTotal + ptrsCopied + k] = mRegionInfos[lastRegion].managed;
          ptrBuffer[3 * ptrsTotal + ptrsCopied + k] = mRegionInfos[lastRegion].id;
        }
        ptrsCopied += ptrs.tpcZS->slice[i].count[j];
      }
    }

    auto channel = device->GetChannels().find("gpu-prepare-channel");
    fair::mq::MessagePtr payload(device->NewMessage());
    LOG(info) << "Sending gpu-reco-workflow prepare message of size " << prepareBufferSize;
    payload->Rebuild(messageBuffer.data(), prepareBufferSize, nullptr, nullptr);
    channel->second[0].Send(payload);
    return 2;
  }
  return 0;
}

void GPURecoWorkflowSpec::handlePipelineEndOfStream(EndOfStreamContext& ec)
{
  if (mSpecConfig.enableDoublePipeline == 1) {
    mPipeline->endOfStreamDplReceived = true;
    mPipeline->stateNotify.notify_all();
  }
  if (mSpecConfig.enableDoublePipeline == 2) {
    auto* device = ec.services().get<RawDeviceService>().device();
    pipelinePrepareMessage preMessage;
    preMessage.flagEndOfStream = true;
    auto channel = device->GetChannels().find("gpu-prepare-channel");
    fair::mq::MessagePtr payload(device->NewMessage());
    LOG(info) << "Sending end-of-stream message over out-of-bands channel";
    payload->Rebuild(&preMessage, sizeof(preMessage), nullptr, nullptr);
    channel->second[0].Send(payload);
  }
}

void GPURecoWorkflowSpec::receiveFMQStateCallback(fair::mq::State newState)
{
  {
    std::lock_guard lk(mPipeline->stateMutex);
    if (mPipeline->fmqState != fair::mq::State::Running && newState == fair::mq::State::Running) {
      mPipeline->endOfStreamAsyncReceived = false;
      mPipeline->endOfStreamDplReceived = false;
    }
    mPipeline->fmqPreviousState = mPipeline->fmqState;
    mPipeline->fmqState = newState;
    if (newState == fair::mq::State::Exiting) {
      mPipeline->fmqDevice->UnsubscribeFromStateChange(GPURecoWorkflowSpec_FMQCallbackKey);
    }
  }
  mPipeline->stateNotify.notify_all();
}

void GPURecoWorkflowSpec::RunReceiveThread()
{
  auto* device = mPipeline->fmqDevice;
  while (!mPipeline->shouldTerminate) {
    bool received = false;
    int recvTimeot = 1000;
    fair::mq::MessagePtr msg;
    LOG(debug) << "Waiting for out of band message";
    auto shouldReceive = [this]() { return ((mPipeline->fmqState == fair::mq::State::Running || (mPipeline->fmqState == fair::mq::State::Ready && mPipeline->fmqPreviousState == fair::mq::State::Running)) && !mPipeline->endOfStreamAsyncReceived); };
    do {
      {
        std::unique_lock lk(mPipeline->stateMutex);
        mPipeline->stateNotify.wait(lk, [this, shouldReceive]() { return shouldReceive() || mPipeline->shouldTerminate; }); // Do not check mPipeline->fmqDevice->NewStatePending() since we wait for EndOfStream!
      }
      if (mPipeline->shouldTerminate) {
        break;
      }
      try {
        do {
          std::unique_lock lk(mPipeline->stateMutex);
          if (!shouldReceive()) {
            break;
          }
          msg = device->NewMessageFor("gpu-prepare-channel", 0, 0);
          received = device->Receive(msg, "gpu-prepare-channel", 0, recvTimeot) > 0;
        } while (!received && !mPipeline->shouldTerminate);
      } catch (...) {
        usleep(1000000);
      }
    } while (!received && !mPipeline->shouldTerminate);
    if (mPipeline->shouldTerminate) {
      break;
    }
    if (msg->GetSize() < sizeof(pipelinePrepareMessage)) {
      LOG(fatal) << "Received prepare message of invalid size " << msg->GetSize() << " < " << sizeof(pipelinePrepareMessage);
    }
    const pipelinePrepareMessage* m = (const pipelinePrepareMessage*)msg->GetData();
    if (m->magicWord != m->MAGIC_WORD) {
      LOG(fatal) << "Prepare message corrupted, invalid magic word";
    }
    if (m->flagEndOfStream) {
      LOG(info) << "Received end-of-stream from out-of-band channel";
      std::lock_guard lk(mPipeline->stateMutex);
      mPipeline->endOfStreamAsyncReceived = true;
      mPipeline->mNTFReceived = 0;
      mPipeline->runStarted = false;
      continue;
    }

    {
      std::lock_guard lk(mPipeline->completionPolicyMutex);
      mPipeline->completionPolicyQueue.emplace(m->timeSliceId);
    }
    mPipeline->completionPolicyNotify.notify_one();

    {
      std::unique_lock lk(mPipeline->stateMutex);
      mPipeline->stateNotify.wait(lk, [this]() { return (mPipeline->runStarted && !mPipeline->endOfStreamAsyncReceived) || mPipeline->shouldTerminate; });
      if (!mPipeline->runStarted) {
        continue;
      }
    }

    auto context = std::make_unique<GPURecoWorkflow_QueueObject>();
    context->timeSliceId = m->timeSliceId;
    context->tfSettings = m->tfSettings;

    size_t ptrsCopied = 0;
    size_t* ptrBuffer = (size_t*)msg->GetData() + sizeof(pipelinePrepareMessage) / sizeof(size_t);
    context->tpcZSmeta.Pointers[0][0].resize(m->pointersTotal);
    context->tpcZSmeta.Sizes[0][0].resize(m->pointersTotal);
    int lastRegion = -1;
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        context->tpcZS.slice[i].count[j] = m->pointerCounts[i][j];
        for (unsigned int k = 0; k < context->tpcZS.slice[i].count[j]; k++) {
          bool regionManaged = ptrBuffer[2 * m->pointersTotal + ptrsCopied + k];
          size_t regionId = ptrBuffer[3 * m->pointersTotal + ptrsCopied + k];
          bool regionFound = lastRegion != -1 && mRegionInfos[lastRegion].managed == regionManaged && mRegionInfos[lastRegion].id == regionId;
          if (!regionFound) {
            for (unsigned int l = 0; l < mRegionInfos.size(); l++) {
              if (mRegionInfos[l].managed == regionManaged && mRegionInfos[l].id == regionId) {
                lastRegion = l;
                regionFound = true;
                break;
              }
            }
          }
          if (!regionFound) {
            LOG(fatal) << "Received ZS Ptr for SHM region (managed " << (int)regionManaged << ", id " << regionId << "), which was not registered for us";
          }
          context->tpcZSmeta.Pointers[0][0][ptrsCopied + k] = (void*)(ptrBuffer[ptrsCopied + k] + (size_t)mRegionInfos[lastRegion].ptr);
          context->tpcZSmeta.Sizes[0][0][ptrsCopied + k] = ptrBuffer[m->pointersTotal + ptrsCopied + k];
        }
        context->tpcZS.slice[i].zsPtr[j] = context->tpcZSmeta.Pointers[0][0].data() + ptrsCopied;
        context->tpcZS.slice[i].nZSPtr[j] = context->tpcZSmeta.Sizes[0][0].data() + ptrsCopied;
        ptrsCopied += context->tpcZS.slice[i].count[j];
      }
    }
    context->ptrs.tpcZS = &context->tpcZS;
    context->ptrs.settingsTF = &context->tfSettings;
    context->mTFId = mPipeline->mNTFReceived;
    if (mPipeline->mNTFReceived++ >= mPipeline->workers.size()) { // Do not inject the first workers.size() TFs, since we need a first round of calib updates from DPL before starting
      enqueuePipelinedJob(&context->ptrs, nullptr, context.get(), false);
    }
    {
      std::lock_guard lk(mPipeline->queueMutex);
      mPipeline->pipelineQueue.emplace(std::move(context));
    }
    mPipeline->queueNotify.notify_one();
  }
  mPipeline->pipelineSenderTerminating = true;
  mPipeline->completionPolicyNotify.notify_one();
}

void GPURecoWorkflowSpec::ExitPipeline()
{
  if (mSpecConfig.enableDoublePipeline == 1 && mPipeline->fmqDevice) {
    mPipeline->fmqDevice = nullptr;
    mPipeline->shouldTerminate = true;
    mPipeline->stateNotify.notify_all();
    for (unsigned int i = 0; i < mPipeline->workers.size(); i++) {
      mPipeline->workers[i].inputQueueNotify.notify_one();
    }
    if (mPipeline->receiveThread.joinable()) {
      mPipeline->receiveThread.join();
    }
    for (unsigned int i = 0; i < mPipeline->workers.size(); i++) {
      if (mPipeline->workers[i].thread.joinable()) {
        mPipeline->workers[i].thread.join();
      }
    }
  }
}

} // namespace o2::gpu
