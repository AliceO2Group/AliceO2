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
#include "Framework/DataTakingContext.h"
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
    auto& dtc = ctx.services().get<DataTakingContext>();
    while ((mSentTimeframes - mConsumedTimeframes) >= maxInFlight) {
      if (recvTimeot == -1 && waitMessage == 0) {
        if (dtc.deploymentMode == DeploymentMode::OnlineDDS || dtc.deploymentMode == DeploymentMode::OnlineECS || dtc.deploymentMode == DeploymentMode::FST) {
          LOG(alarm) << "Maximum number of TF in flight reached (" << maxInFlight << ": published " << mSentTimeframes << " - finished " << mConsumedTimeframes << "), waiting";
        } else {
          LOG(info) << "Maximum number of TF in flight reached (" << maxInFlight << ": published " << mSentTimeframes << " - finished " << mConsumedTimeframes << "), waiting";
        }
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

    bool doSmothThrottling = getenv("DPL_SMOOTH_RATE_LIMITING") && atoi(getenv("DPL_SMOOTH_RATE_LIMITING"));
    if (doSmothThrottling) {
      constexpr float factorStart = 0.5f;
      constexpr float factorFinal = 0.9f;
      constexpr float factorOfAverage = 0.7f;
      constexpr int64_t iterationsFinal = 2;
      auto curTime = std::chrono::system_clock::now();
      if (mTfTimes.size() != maxInFlight) {
        mTfTimes.resize(maxInFlight);
        mTimeCountingSince = mSentTimeframes;
        mFirstTime = curTime;
      }
      if (mSentTimeframes >= mTimeCountingSince + maxInFlight) {
        float iterationDuration = std::chrono::duration_cast<std::chrono::duration<float>>(curTime - mTfTimes[mSentTimeframes % maxInFlight]).count();
        float totalAverage = std::chrono::duration_cast<std::chrono::duration<float>>(curTime - mFirstTime).count() / (mSentTimeframes - mTimeCountingSince);
        if (mSmothDelay == 0.f) {
          mSmothDelay = iterationDuration / maxInFlight * factorStart;
          LOG(debug) << "TF Throttling delay initialized to " << mSmothDelay;
        } else {
          float factor;
          if (mSentTimeframes < maxInFlight) {
            factor = factorStart;
          } else if (mSentTimeframes >= (iterationsFinal + 1) * maxInFlight) {
            factor = factorFinal;
          } else {
            factor = factorStart + (factorFinal - factorStart) * (float)(mSentTimeframes - maxInFlight) / (float)(iterationsFinal * maxInFlight);
          }
          float newDelay = iterationDuration / maxInFlight * factor;
          if (newDelay > totalAverage) {
            LOG(debug) << "TF Throttling: Correcting delay down to average " << newDelay << " --> " << totalAverage;
            newDelay = totalAverage;
          } else if (newDelay < factorOfAverage * totalAverage) {
            LOG(debug) << "TF Throttling: Correcting delay up to " << factorOfAverage << " * average " << newDelay << " --> " << factorOfAverage * totalAverage;
            newDelay = factorOfAverage * totalAverage;
          }
          mSmothDelay = (float)(maxInFlight - 1) / (float)maxInFlight * mSmothDelay + newDelay / (float)maxInFlight;
          LOG(debug) << "TF Throttling delay updated to " << mSmothDelay << " (factor " << factor << " Duration " << iterationDuration / maxInFlight << " = " << iterationDuration << " / " << maxInFlight << " --> " << newDelay << ")";
        }
        float elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(curTime - mLastTime).count();
        if (elapsed < mSmothDelay) {
          LOG(debug) << "TF Throttling: Elapsed " << elapsed << " --> Waiting for " << mSmothDelay - elapsed;
          usleep((mSmothDelay - elapsed) * 1e6);
        }
      }
      mLastTime = std::chrono::system_clock::now();
      mTfTimes[mSentTimeframes % maxInFlight] = curTime;
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
