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

#include "Framework/RateLimiter.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RunningWorkflowInfo.h"
#include <fairmq/Device.h>
#include <fairmq/shmem/Monitor.h>
#include <fairmq/shmem/Common.h>

using namespace o2::framework;

void RateLimiter::check(ProcessingContext& ctx, int maxInFlight, size_t minSHM)
{
  if (!maxInFlight && !minSHM) {
    return;
  }
  auto device = ctx.services().get<RawDeviceService>().device();
  if (maxInFlight && device->fChannels.count("metric-feedback")) {
    int waitMessage = 0;
    int recvTimeot = 0;
    while ((mSentTimeframes - mConsumedTimeframes) >= maxInFlight) {
      if (recvTimeot == -1 && waitMessage == 0) {
        LOG(alarm) << "Maximum number of TF in flight reached (" << maxInFlight << ": published " << mSentTimeframes << " - finished " << mConsumedTimeframes << "), waiting";
        waitMessage = 1;
      }
      auto msg = device->NewMessageFor("metric-feedback", 0, 0);

      auto count = device->Receive(msg, "metric-feedback", 0, recvTimeot);
      if (count <= 0) {
        recvTimeot = -1;
        continue;
      }
      assert(msg->GetSize() == 8);
      mConsumedTimeframes = *(int64_t*)msg->GetData();
    }
    if (waitMessage) {
      LOG(important) << (mSentTimeframes - mConsumedTimeframes) << " / " << maxInFlight << " TF in flight, continuing to publish";
    }
  }
  if (minSHM) {
    int waitMessage = 0;
    auto& runningWorkflow = ctx.services().get<RunningWorkflowInfo const>();
    while (true) {
      long freeMemory = -1;
      try {
        freeMemory = fair::mq::shmem::Monitor::GetFreeMemory(fair::mq::shmem::ShmId{fair::mq::shmem::makeShmIdStr(device->fConfig->GetProperty<uint64_t>("shmid"))}, runningWorkflow.shmSegmentId);
      } catch (...) {
      }
      if (freeMemory == -1) {
        try {
          freeMemory = fair::mq::shmem::Monitor::GetFreeMemory(fair::mq::shmem::SessionId{device->fConfig->GetProperty<std::string>("session")}, runningWorkflow.shmSegmentId);
        } catch (...) {
        }
      }
      if (freeMemory == -1) {
        throw std::runtime_error("Could not obtain free SHM memory");
      }
      uint64_t freeSHM = freeMemory;
      if (freeSHM > minSHM) {
        if (waitMessage) {
          LOG(important) << "Sufficient SHM memory free (" << freeSHM << " >= " << minSHM << "), continuing to publish";
        }
        break;
      }
      if (waitMessage == 0) {
        LOG(alarm) << "Free SHM memory too low: " << freeSHM << " < " << minSHM << ", waiting";
        waitMessage = 1;
      }
    }
  }
  mSentTimeframes++;
}
