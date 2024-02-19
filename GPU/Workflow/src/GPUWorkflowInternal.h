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

/// @file   GPUWorkflowInternal.h
/// @author David Rohr

#ifndef O2_GPU_GPUWORKFLOWINTERNAL_H
#define O2_GPU_GPUWORKFLOWINTERNAL_H

#include "GPUDataTypes.h"
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <array>
#include <fairmq/States.h>

namespace o2::gpu
{
namespace gpurecoworkflow_internals
{

struct GPURecoWorkflowSpec_TPCZSBuffers {
  std::vector<const void*> Pointers[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  std::vector<unsigned int> Sizes[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  const void** Pointers2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
  const unsigned int* Sizes2[GPUTrackingInOutZS::NSLICES][GPUTrackingInOutZS::NENDPOINTS];
};

struct GPURecoWorkflow_QueueObject {
  GPURecoWorkflowSpec_TPCZSBuffers tpcZSmeta;
  GPUTrackingInOutZS tpcZS;
  GPUSettingsTF tfSettings;
  GPUTrackingInOutPointers ptrs;
  o2::framework::DataProcessingHeader::StartTime timeSliceId;

  unsigned long mTFId;

  bool jobSubmitted = false;
  bool jobFinished = false;
  int jobReturnValue = 0;
  std::mutex jobFinishedMutex;
  std::condition_variable jobFinishedNotify;
  bool jobInputFinal = false;
  std::mutex jobInputFinalMutex;
  std::condition_variable jobInputFinalNotify;
  GPUTrackingInOutPointers* jobPtrs = nullptr;
  GPUInterfaceOutputs* jobOutputRegions = nullptr;
  std::unique_ptr<GPUInterfaceInputUpdate> jobInputUpdateCallback = nullptr;
};

struct GPURecoWorkflowSpec_PipelineInternals {
  std::mutex mutexDecodeInput;

  fair::mq::Device* fmqDevice = nullptr;

  volatile fair::mq::State fmqState = fair::mq::State::Undefined, fmqPreviousState = fair::mq::State::Undefined;
  volatile bool endOfStreamAsyncReceived = false;
  volatile bool endOfStreamDplReceived = false;
  volatile bool runStarted = false;
  volatile bool shouldTerminate = false;
  std::mutex stateMutex;
  std::condition_variable stateNotify;

  std::thread receiveThread;

  struct pipelineWorkerStruct {
    std::thread thread;
    std::queue<GPURecoWorkflow_QueueObject*> inputQueue;
    std::mutex inputQueueMutex;
    std::condition_variable inputQueueNotify;
  };
  std::array<pipelineWorkerStruct, 2> workers;

  std::queue<std::unique_ptr<GPURecoWorkflow_QueueObject>> pipelineQueue;
  std::mutex queueMutex;
  std::condition_variable queueNotify;

  std::queue<o2::framework::DataProcessingHeader::StartTime> completionPolicyQueue;
  volatile bool pipelineSenderTerminating = false;
  std::mutex completionPolicyMutex;
  std::condition_variable completionPolicyNotify;

  unsigned long mNTFReceived = 0;

  volatile bool mayInject = true;
  volatile unsigned long mayInjectTFId = 0;
  std::mutex mayInjectMutex;
  std::condition_variable mayInjectCondition;
};

} // namespace gpurecoworkflow_internals
} // namespace o2::gpu

#endif
