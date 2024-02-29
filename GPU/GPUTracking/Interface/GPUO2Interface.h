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

// Some defines denoting that we are compiling for O2
#ifndef GPUCA_HAVE_O2HEADERS
#define GPUCA_HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include <memory>
#include <array>
#include <vector>
#include "GPUCommonDef.h"
#include "GPUDataTypes.h"

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
template <class T>
class CalDet;
} // namespace o2::tpc

namespace o2::its
{
class TrackerTraits;
class VertexerTraits;
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

  int Initialize(const GPUO2InterfaceConfiguration& config);
  void Deinitialize();

  int RunTracking(GPUTrackingInOutPointers* data, GPUInterfaceOutputs* outputs = nullptr, unsigned int iThread = 0, GPUInterfaceInputUpdate* inputUpdateCallback = nullptr);
  void Clear(bool clearOutputs, unsigned int iThread = 0);
  void DumpEvent(int nEvent, GPUTrackingInOutPointers* data);
  void DumpSettings();

  void GetITSTraits(o2::its::TrackerTraits*& trackerTraits, o2::its::VertexerTraits*& vertexerTraits, o2::its::TimeFrame*& timeFrame);
  const o2::base::Propagator* GetDeviceO2Propagator(int iThread = 0) const;

  // Updates all calibration objects that are != nullptr in newCalib
  int UpdateCalibration(const GPUCalibObjectsConst& newCalib, const GPUNewCalibValues& newVals, unsigned int iThread = 0);

  static std::unique_ptr<TPCPadGainCalib> getPadGainCalibDefault();
  static std::unique_ptr<TPCPadGainCalib> getPadGainCalib(const o2::tpc::CalDet<float>& in);

  static std::unique_ptr<o2::tpc::CalibdEdxContainer> getCalibdEdxContainerDefault();

  int registerMemoryForGPU(const void* ptr, size_t size);
  int unregisterMemoryForGPU(const void* ptr);
  void setErrorCodeOutput(std::vector<std::array<unsigned int, 4>>* v);

  const GPUO2InterfaceConfiguration& getConfig() const { return *mConfig; }

 private:
  GPUO2Interface(const GPUO2Interface&);
  GPUO2Interface& operator=(const GPUO2Interface&);

  bool mContinuous = false;

  unsigned int mNContexts = 0;
  std::unique_ptr<GPUO2Interface_processingContext[]> mCtx;

  std::unique_ptr<GPUO2InterfaceConfiguration> mConfig;
  GPUChainITS* mChainITS = nullptr;
  std::unique_ptr<GPUO2Interface_Internals> mInternals;
};
} // namespace o2::gpu

#endif
