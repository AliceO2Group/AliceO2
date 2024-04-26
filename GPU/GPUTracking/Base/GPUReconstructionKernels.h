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

template <class T, int I = 0>
struct classArgument {
  using t = T;
  static constexpr int i = I;
};

struct krnlExec {
  constexpr krnlExec(unsigned int b, unsigned int t, int s, GPUReconstruction::krnlDeviceType d = GPUReconstruction::krnlDeviceType::Auto) : nBlocks(b), nThreads(t), stream(s), device(d), step(GPUCA_RECO_STEP::NoRecoStep) {}
  constexpr krnlExec(unsigned int b, unsigned int t, int s, GPUCA_RECO_STEP st) : nBlocks(b), nThreads(t), stream(s), device(GPUReconstruction::krnlDeviceType::Auto), step(st) {}
  constexpr krnlExec(unsigned int b, unsigned int t, int s, GPUReconstruction::krnlDeviceType d, GPUCA_RECO_STEP st) : nBlocks(b), nThreads(t), stream(s), device(d), step(st) {}
  unsigned int nBlocks;
  unsigned int nThreads;
  int stream;
  GPUReconstruction::krnlDeviceType device;
  GPUCA_RECO_STEP step;
};
struct krnlRunRange {
  constexpr krnlRunRange() = default;
  constexpr krnlRunRange(unsigned int a) : start(a), num(0) {}
  constexpr krnlRunRange(unsigned int s, int n) : start(s), num(n) {}

  unsigned int start = 0;
  int num = 0;
};
struct krnlEvent {
  constexpr krnlEvent(deviceEvent* e = nullptr, deviceEvent* el = nullptr, int n = 1) : ev(e), evList(el), nEvents(n) {}
  deviceEvent* ev;
  deviceEvent* evList;
  int nEvents;
};

struct krnlProperties {
  krnlProperties(int t = 0, int b = 1, int b2 = 0) : nThreads(t), minBlocks(b), forceBlocks(b2) {}
  unsigned int nThreads;
  unsigned int minBlocks;
  unsigned int forceBlocks;
  unsigned int total() { return forceBlocks ? forceBlocks : (nThreads * minBlocks); }
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

template <class T, int I = 0, typename... Args>
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
  template <class S, int I = 0, typename... Args>
  using krnlSetupArgs = gpu_reconstruction_kernels::krnlSetupArgs<S, I, Args...>;

#define GPUCA_KRNL(x_class, attributes, x_arguments, x_forward, x_types)                                                                                \
  virtual int runKernelImpl(const krnlSetupArgs<GPUCA_M_KRNL_TEMPLATE(x_class) GPUCA_M_STRIP(x_types)>& args)                                           \
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
