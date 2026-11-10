// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef GPURECONSTRUCTIONOMETALINCLUDESHOST_H
#define GPURECONSTRUCTIONOMETALINCLUDESHOST_H

#ifdef GPUCOMMONDEF_H
#error "this header establishes GPUCA_GPUTYPE_METAL and __METAL_HOST__, so it must be included before any other GPU header"
#endif

#define GPUCA_GPUTYPE_METAL

#import <Metal/Metal.h>
#ifndef __METAL_VERSION__
// __METAL_VERSION__ is only defined in device code.
#define __METAL_HOST__
// The parameter table picks MSL's `half` for the device layouts this translation
// unit has to agree with; _Float16 is the same IEEE binary16 type.
using half = _Float16;
#endif

#import <Foundation/Foundation.h>

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <cstdio>
#include <mach/mach_time.h>
#include "GPULogging.h"

#include "GPUReconstructionMetal.h"
#include "GPUReconstructionIncludes.h"
#include "GPUCommonHelpers.h"

using namespace o2::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

namespace o2::gpu
{

struct GPUReconstructionMetalInternals {
  id<MTLDevice> device;

  std::array<id<MTLCommandQueue>, constants::GPU_MAX_STREAMS> commandQueues;
  // The buffer currently open for encoding, plus the ones already committed and
  // not yet waited for. A Metal command buffer takes one commit and cannot be
  // encoded into afterwards, so the two cannot be the same object.
  std::array<id<MTLCommandBuffer>, constants::GPU_MAX_STREAMS> commandBuffers;
  std::array<std::vector<id<MTLCommandBuffer>>, constants::GPU_MAX_STREAMS> submittedBuffers;
  // Signalled by each committed buffer and waited for by the next one, see CommitStream
  std::array<id<MTLSharedEvent>, constants::GPU_MAX_STREAMS> streamOrder;
  std::array<uint64_t, constants::GPU_MAX_STREAMS> streamOrderValue;
  // Orders the encoders within the open command buffer of a stream, which
  // collects the stream's work until something has to wait for it.
  std::array<id<MTLFence>, constants::GPU_MAX_STREAMS> streamFence;
  std::array<bool, constants::GPU_MAX_STREAMS> streamHasWork;
  // Set while the device is torn down, when a failed command buffer is no longer fatal
  bool exiting = false;
  // Opt-in command-buffer timeline, GPUCA_METAL_TIMELINE=<file>: what each open
  // buffer contains (kernels, blits, waits, signals), written out once it completes
  FILE* timeline = nullptr;
  std::array<std::string, constants::GPU_MAX_STREAMS> timelineOps;
  std::array<uint32_t, constants::GPU_MAX_STREAMS> timelineKernels;
  uint64_t timelineBuffers = 0;
  // Opt-in limits on the GPU work in flight, 0 = none, so that no stretch of GPU
  // work runs long enough for the macOS GPU watchdog: GPUCA_METAL_MAX_KERNELS_PER_BUFFER
  // commits a stream's buffer after that many kernels, GPUCA_METAL_MAX_STREAMS_IN_FLIGHT
  // is how many streams may have committed buffers that are not complete yet.
  uint32_t maxKernelsPerBuffer = 0;
  uint32_t maxStreamsInFlight = 0;
  std::array<uint32_t, constants::GPU_MAX_STREAMS> streamKernels{};
  std::array<uint32_t, constants::GPU_MAX_STREAMS> streamBuffersInFlight{};
  uint32_t streamsInFlight = 0;
  std::mutex inFlightMutex;
  std::condition_variable inFlightDone;

  std::vector<id<MTLFunction>> functions;
  std::vector<id<MTLComputePipelineState>> pipelines;
  std::vector<uint32_t> maxThreadsPerGroup;

  id<MTLBuffer> mem_gpu;
  id<MTLBuffer> mem_constant;
  id<MTLBuffer> mem_host;

  id<MTLLibrary> library;

  // The value each event was last recorded with. An MTLSharedEvent only counts
  // up, so recording signals the next value and waiting asks for the last one,
  // which makes waiting for a never recorded event a no-op, as in CUDA.
  std::unordered_map<void*, uint64_t> eventValues;
  std::mutex eventMutex;
};
// An encoder of a stream starts after the previous one of the same stream and
// leaves its work in the open command buffer.
template <class E>
inline void metalBeginEncoder(GPUReconstructionMetalInternals* internals, E encoder, int32_t stream)
{
  [encoder waitForFence:internals->streamFence[stream]];
}
// A failed command buffer does not signal its events, and the work waiting for
// them would wait forever: signal them from the host, so that everything
// completes and the failure is reported when the host synchronises.
inline void metalSignalOnError(id<MTLCommandBuffer> cb, id<MTLSharedEvent> event, uint64_t value)
{
  [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
    if (c.status == MTLCommandBufferStatusError && event.signaledValue < value) {
      event.signaledValue = value;
    }
  }];
}
// Before a buffer of an idle stream is committed, wait until fewer than the
// limit of streams have work in flight. This cannot deadlock: a buffer only waits
// for events that buffers committed before it signal (RecordMarker commits the
// signal at once), so the work in flight completes without the host's help.
inline void metalLimitStreamsInFlight(GPUReconstructionMetalInternals* internals, id<MTLCommandBuffer> cb, int32_t stream)
{
  std::unique_lock<std::mutex> lock(internals->inFlightMutex);
  if (internals->streamBuffersInFlight[stream]++ == 0) {
    internals->inFlightDone.wait(lock, [internals] { return internals->streamsInFlight < internals->maxStreamsInFlight; });
    internals->streamsInFlight++;
  }
  [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
    std::lock_guard<std::mutex> l(internals->inFlightMutex);
    if (--internals->streamBuffersInFlight[stream] == 0) {
      internals->streamsInFlight--;
      internals->inFlightDone.notify_all();
    }
  }];
}
// In the time base of the command buffers' GPUStartTime and GPUEndTime
inline double metalHostTime()
{
  static const double scale = [] { mach_timebase_info_data_t tb; mach_timebase_info(&tb); return 1e-9 * tb.numer / tb.denom; }();
  return mach_absolute_time() * scale;
}
inline void metalTimelineAdd(GPUReconstructionMetalInternals* internals, int32_t stream, const char* op, const void* id, uint64_t value)
{
  if (internals->timeline) {
    char buf[96];
    snprintf(buf, sizeof(buf), " %s:%p=%llu", op, id, (unsigned long long)value);
    internals->timelineOps[stream] += buf;
  }
}
template <class E>
inline void metalEndEncoder(GPUReconstructionMetalInternals* internals, E encoder, int32_t stream)
{
  [encoder updateFence:internals->streamFence[stream]];
  [encoder endEncoding];
  internals->streamHasWork[stream] = true;
}
} // namespace o2::gpu

#endif // GPURECONSTRUCTIONOMETALINCLUDESHOST_H
