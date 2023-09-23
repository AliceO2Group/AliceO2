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
};

struct GPURecoWorkflowSpec_PipelineInternals {
  std::mutex mutexDecodeInput;

  fair::mq::Device* fmqDevice;

  std::thread receiveThread;
  std::condition_variable notifyThread;
  std::mutex threadMutex;
  volatile bool shouldTerminate = false;

  std::queue<std::unique_ptr<GPURecoWorkflow_QueueObject>> pipelineQueue;
  std::mutex queueMutex;
  std::condition_variable queueNotify;
};

} // namespace gpurecoworkflow_internals
} // namespace o2::gpu

#endif
