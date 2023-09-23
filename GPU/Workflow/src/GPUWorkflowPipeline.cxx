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

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;
using namespace o2::dataformats;
using namespace o2::gpu::gpurecoworkflow_internals;

namespace o2::gpu
{

struct pipelinePrepareMessage {
  static constexpr size_t MAGIC_WORD = 0X8473957353424134;
  size_t magicWord = MAGIC_WORD;
  DataProcessingHeader::StartTime timeSliceId;
  GPUSettingsTF tfSettings;
  fair::mq::RegionInfo regionInfo;
  size_t pointerCounts[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  size_t pointersTotal;
};

int GPURecoWorkflowSpec::handlePipeline(ProcessingContext& pc, GPUTrackingInOutPointers& ptrs, GPURecoWorkflowSpec_TPCZSBuffers& tpcZSmeta, o2::gpu::GPUTrackingInOutZS& tpcZS)
{
  auto* device = pc.services().get<RawDeviceService>().device();
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  if (mSpecConfig.enableDoublePipeline == 1) {
    std::unique_lock lk(mPipeline->queueMutex);
    mPipeline->queueNotify.wait(lk, [this] { return !mPipeline->pipelineQueue.empty(); });
    auto o = std::move(mPipeline->pipelineQueue.front());
    mPipeline->pipelineQueue.pop();
    lk.unlock();

    if (o->timeSliceId != tinfo.timeslice) {
      LOG(fatal) << "Prepare message for incorrect time frame received, time frames seem out of sync";
    }

    tpcZSmeta = std::move(o->tpcZSmeta);
    tpcZS = o->tpcZS;
    ptrs.tpcZS = &tpcZS;
  }
  if (mSpecConfig.enableDoublePipeline == 2) {
    auto prepareBuffer = pc.outputs().make<DataAllocator::UninitializedVector<char>>(Output{gDataOriginGPU, "PIPELINEPREPARE", 0, Lifetime::Timeframe}, 0u);

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

    size_t prepareBufferSize = sizeof(pipelinePrepareMessage) + ptrsTotal * sizeof(size_t) * 2;
    std::vector<size_t> messageBuffer(prepareBufferSize / sizeof(size_t));
    pipelinePrepareMessage& preMessage = *(pipelinePrepareMessage*)messageBuffer.data();
    preMessage.magicWord = preMessage.MAGIC_WORD;
    preMessage.timeSliceId = tinfo.timeslice;
    preMessage.pointersTotal = ptrsTotal;
    memcpy((void*)&preMessage.tfSettings, (const void*)ptrs.settingsTF, sizeof(preMessage.tfSettings));

    if (ptrsTotal) {
      bool regionFound = false;
      for (unsigned int i = 0; i < mRegionInfos.size(); i++) {
        if ((size_t)firstPtr >= (size_t)mRegionInfos[i].ptr && (size_t)firstPtr < (size_t)mRegionInfos[i].ptr + mRegionInfos[i].size) {
          preMessage.regionInfo = mRegionInfos[i];
          regionFound = true;
          break;
        }
      }
      if (!regionFound) {
        LOG(fatal) << "Found a TPC ZS pointer outside of shared memory";
      }
    }

    size_t* ptrBuffer = messageBuffer.data() + sizeof(preMessage) / sizeof(size_t);
    size_t ptrsCopied = 0;
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        preMessage.pointerCounts[i][j] = ptrs.tpcZS->slice[i].count[j];
        for (unsigned int k = 0; k < ptrs.tpcZS->slice[i].count[j]; k++) {
          ptrBuffer[ptrsCopied + k] = (size_t)ptrs.tpcZS->slice[i].zsPtr[j][k] - (size_t)preMessage.regionInfo.ptr;
          ptrBuffer[ptrsTotal + ptrsCopied + k] = ptrs.tpcZS->slice[i].nZSPtr[j][k];
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

void GPURecoWorkflowSpec::RunReceiveThread()
{
  std::unique_lock lk(mPipeline->threadMutex);
  mPipeline->notifyThread.wait(lk, [this]() { return mPipeline->shouldTerminate || mPipeline->mayReceive; });
  lk.unlock();
  while (!mPipeline->shouldTerminate) {
    auto* device = mPipeline->fmqDevice;
    bool received = false;
    int recvTimeot = 1000;
    fair::mq::MessagePtr msg;
    LOG(info) << "Waiting for out of band message";
    do {
      try {
        msg = device->NewMessageFor("gpu-prepare-channel", 0, 0);
        do {
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

    auto o = std::make_unique<GPURecoWorkflow_QueueObject>();
    o->timeSliceId = m->timeSliceId;
    o->tfSettings = m->tfSettings;

    size_t regionOffset = 0;
    if (m->pointersTotal) {
      bool regionFound = false;
      for (unsigned int i = 0; i < mRegionInfos.size(); i++) {
        if (mRegionInfos[i].managed == m->regionInfo.managed && mRegionInfos[i].id == m->regionInfo.id) {
          regionFound = true;
          regionOffset = (size_t)mRegionInfos[i].ptr;
          break;
        }
      }
    }
    size_t ptrsCopied = 0;
    size_t* ptrBuffer = (size_t*)msg->GetData() + sizeof(pipelinePrepareMessage) / sizeof(size_t);
    o->tpcZSmeta.Pointers[0][0].resize(m->pointersTotal);
    o->tpcZSmeta.Sizes[0][0].resize(m->pointersTotal);
    for (unsigned int i = 0; i < GPUTrackingInOutZS::NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        o->tpcZS.slice[i].count[j] = m->pointerCounts[i][j];
        for (unsigned int k = 0; k < o->tpcZS.slice[i].count[j]; k++) {
          o->tpcZSmeta.Pointers[0][0][ptrsCopied + k] = (void*)(ptrBuffer[ptrsCopied + k] + regionOffset);
          o->tpcZSmeta.Sizes[0][0][ptrsCopied + k] = ptrBuffer[m->pointersTotal + ptrsCopied + k];
        }
        o->tpcZS.slice[i].zsPtr[j] = o->tpcZSmeta.Pointers[0][0].data() + ptrsCopied;
        o->tpcZS.slice[i].nZSPtr[j] = o->tpcZSmeta.Sizes[0][0].data() + ptrsCopied;
        ptrsCopied += o->tpcZS.slice[i].count[j];
      }
    }
    o->ptrs.tpcZS = &o->tpcZS;
    {
      std::lock_guard lk(mPipeline->queueMutex);
      mPipeline->pipelineQueue.emplace(std::move(o));
    }
    mPipeline->queueNotify.notify_one();
  }
}

void GPURecoWorkflowSpec::TerminateReceiveThread()
{
  if (mPipeline->receiveThread.joinable()) {
    mPipeline->shouldTerminate = true;
    mPipeline->receiveThread.join();
  }
}

} // namespace o2::gpu
