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

/// \file GPUReconstructionKernels.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONKERNELS_H
#define GPURECONSTRUCTIONKERNELS_H

#include "GPUReconstruction.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

namespace gpu_reconstruction_kernels
{
struct deviceEvent {
  constexpr deviceEvent() = default;
  constexpr deviceEvent(std::nullptr_t p) : v(nullptr){};
  template <class T>
  void set(T val) { v = reinterpret_cast<void*&>(val); }
  template <class T>
  T& get() { return reinterpret_cast<T&>(v); }
  template <class T>
  T* getEventList() { return reinterpret_cast<T*>(this); }
  bool isSet() const { return v; }

 private:
  void* v = nullptr; // We use only pointers anyway, and since cl_event and cudaEvent_t and hipEvent_t are actually pointers, we can cast them to deviceEvent (void*) this way.
};

template <class T, int32_t I = 0>
struct classArgument {
  using t = T;
  static constexpr int32_t i = I;
};

struct krnlExec {
  constexpr krnlExec(uint32_t b, uint32_t t, int32_t s, GPUReconstruction::krnlDeviceType d = GPUReconstruction::krnlDeviceType::Auto) : nBlocks(b), nThreads(t), stream(s), device(d), step(GPUCA_RECO_STEP::NoRecoStep) {}
  constexpr krnlExec(uint32_t b, uint32_t t, int32_t s, GPUCA_RECO_STEP st) : nBlocks(b), nThreads(t), stream(s), device(GPUReconstruction::krnlDeviceType::Auto), step(st) {}
  constexpr krnlExec(uint32_t b, uint32_t t, int32_t s, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st) : nBlocks(b), nThreads(t), stream(s), device(d), step(st) {}
  uint32_t nBlocks;
  uint32_t nThreads;
  int32_t stream;
  GPUReconstruction::krnlDeviceType device;
  GPUCA_RECO_STEP step;
};
struct krnlRunRange {
  constexpr krnlRunRange() = default;
  constexpr krnlRunRange(uint32_t a) : start(a), num(0) {}
  constexpr krnlRunRange(uint32_t s, int32_t n) : start(s), num(n) {}

  uint32_t start = 0;
  int32_t num = 0;
};
struct krnlEvent {
  constexpr krnlEvent(deviceEvent* e = nullptr, deviceEvent* el = nullptr, int32_t n = 1) : ev(e), evList(el), nEvents(n) {}
  deviceEvent* ev;
  deviceEvent* evList;
  int32_t nEvents;
};

struct krnlProperties {
  krnlProperties(int32_t t = 0, int32_t b = 1, int32_t b2 = 0) : nThreads(t), minBlocks(b), forceBlocks(b2) {}
  uint32_t nThreads;
  uint32_t minBlocks;
  uint32_t forceBlocks;
  uint32_t total() { return forceBlocks ? forceBlocks : (nThreads * minBlocks); }
};

struct krnlSetup {
  krnlSetup(const krnlExec& xx, const krnlRunRange& yy = {0, -1}, const krnlEvent& zz = {nullptr, nullptr, 0}) : x(xx), y(yy), z(zz) {}
  krnlExec x;
  krnlRunRange y;
  krnlEvent z;
};

struct krnlSetupTime : public krnlSetup {
  double& t;
};

template <class T, int32_t I = 0, typename... Args>
struct krnlSetupArgs : public gpu_reconstruction_kernels::classArgument<T, I> {
  krnlSetupArgs(const krnlExec& xx, const krnlRunRange& yy, const krnlEvent& zz, double& tt, const Args&... args) : s{{xx, yy, zz}, tt}, v(args...) {}
  const krnlSetupTime s;
  std::tuple<typename std::conditional<(sizeof(Args) > sizeof(void*)), const Args&, const Args>::type...> v;
};
} // namespace gpu_reconstruction_kernels

template <class T>
class GPUReconstructionKernels : public T
{
 public:
  GPUReconstructionKernels(const GPUSettingsDeviceBackend& cfg) : T(cfg) {}

 protected:
  using deviceEvent = gpu_reconstruction_kernels::deviceEvent;
  using krnlExec = gpu_reconstruction_kernels::krnlExec;
  using krnlRunRange = gpu_reconstruction_kernels::krnlRunRange;
  using krnlEvent = gpu_reconstruction_kernels::krnlEvent;
  using krnlSetup = gpu_reconstruction_kernels::krnlSetup;
  using krnlSetupTime = gpu_reconstruction_kernels::krnlSetupTime;
  template <class S, int32_t I = 0, typename... Args>
  using krnlSetupArgs = gpu_reconstruction_kernels::krnlSetupArgs<S, I, Args...>;

#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward, x_types)                                                                                \
  virtual int32_t runKernelImpl(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>& args)                                       \
  {                                                                                                                                                     \
    return T::template runKernelBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>(args);                                                                          \
  }                                                                                                                                                     \
  virtual gpu_reconstruction_kernels::krnlProperties getKernelPropertiesImpl(gpu_reconstruction_kernels::classArgument<GPUCA_M_KRNL_TEMPLATE(x_class)>) \
  {                                                                                                                                                     \
    return T::template getKernelPropertiesBackend<GPUCA_M_KRNL_TEMPLATE(x_class)>();                                                                    \
  }
#include "GPUReconstructionKernelList.h"
#undef GPUCA_KRNL
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
