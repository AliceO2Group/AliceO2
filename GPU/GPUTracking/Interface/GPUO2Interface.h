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

/// \file GPUO2Interface.h
/// \author David Rohr

#ifndef GPUO2INTERFACE_H
#define GPUO2INTERFACE_H

#include "GPUO2ExternalUser.h"
#include "GPUCommonDef.h"
#include "GPUDataTypes.h"

#include <memory>
#include <array>
#include <vector>

namespace o2::base
{
template <typename value_T>
class PropagatorImpl;
using Propagator = PropagatorImpl<float>;
} // namespace o2::base
namespace o2::tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
} // namespace o2::tpc

namespace o2::its
{
template <int>
class TrackerTraits;
template <int>
class VertexerTraits;
template <int>
class TimeFrame;
} // namespace o2::its

namespace o2::gpu
{
class GPUReconstruction;
class GPUChainTracking;
class GPUChainITS;
struct GPUO2InterfaceConfiguration;
struct GPUInterfaceOutputs;
struct GPUInterfaceInputUpdate;
struct GPUTrackingOutputs;
struct GPUConstantMem;
struct GPUNewCalibValues;

struct GPUO2Interface_processingContext;
struct GPUO2Interface_Internals;

class GPUO2Interface
{
 public:
  GPUO2Interface();
  ~GPUO2Interface();

  int32_t Initialize(const GPUO2InterfaceConfiguration& config);
  void Deinitialize();

  int32_t RunTracking(GPUTrackingInOutPointers* data, GPUInterfaceOutputs* outputs = nullptr, uint32_t iThread = 0, GPUInterfaceInputUpdate* inputUpdateCallback = nullptr);
  void Clear(bool clearOutputs, uint32_t iThread = 0);
  void DumpEvent(int32_t nEvent, GPUTrackingInOutPointers* data, uint32_t iThread, const char* dir = "");
  void DumpSettings(uint32_t iThread, const char* dir = "");

  void GetITSTraits(o2::its::TrackerTraits<7>*& trackerTraits, o2::its::VertexerTraits<7>*& vertexerTraits, o2::its::TimeFrame<7>*& timeFrame);
  const o2::base::Propagator* GetDeviceO2Propagator(int32_t iThread = 0) const;
  void UseGPUPolynomialFieldInPropagator(o2::base::Propagator* prop) const;

  // Updates all calibration objects that are != nullptr in newCalib
  int32_t UpdateCalibration(const GPUCalibObjectsConst& newCalib, const GPUNewCalibValues& newVals, uint32_t iThread = 0);

  int32_t registerMemoryForGPU(const void* ptr, size_t size);
  int32_t unregisterMemoryForGPU(const void* ptr);
  void setErrorCodeOutput(std::vector<std::array<uint32_t, 4>>* v);

  const GPUO2InterfaceConfiguration& getConfig() const { return *mConfig; }

 private:
  GPUO2Interface(const GPUO2Interface&);
  GPUO2Interface& operator=(const GPUO2Interface&);

  bool mContinuous = false;

  uint32_t mNContexts = 0;
  std::unique_ptr<GPUO2Interface_processingContext[]> mCtx;

  std::unique_ptr<GPUO2InterfaceConfiguration> mConfig;
  GPUChainITS* mChainITS = nullptr;
  std::unique_ptr<GPUO2Interface_Internals> mInternals;
};
} // namespace o2::gpu

#endif
