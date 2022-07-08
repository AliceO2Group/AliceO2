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
#include <vector>
#include "GPUCommonDef.h"
#include "GPUDataTypes.h"
namespace o2::tpc
{
struct ClusterNativeAccess;
struct ClusterNative;
template <class T>
class CalDet;
} // namespace o2::tpc

namespace o2::gpu
{
class GPUReconstruction;
class GPUChainTracking;
struct GPUO2InterfaceConfiguration;
struct GPUInterfaceOutputs;
struct GPUTrackingOutputs;
struct GPUConstantMem;
struct GPUNewCalibValues;

class GPUO2Interface
{
 public:
  GPUO2Interface();
  ~GPUO2Interface();

  int Initialize(const GPUO2InterfaceConfiguration& config);
  void Deinitialize();

  int RunTracking(GPUTrackingInOutPointers* data, GPUInterfaceOutputs* outputs = nullptr);
  void Clear(bool clearOutputs);

  // Updates all calibration objects that are != nullptr in newCalib
  int UpdateCalibration(const GPUCalibObjectsConst& newCalib, const GPUNewCalibValues& newVals);

  bool GetParamContinuous() { return (mContinuous); }
  void GetClusterErrors2(int row, float z, float sinPhi, float DzDs, short clusterState, float& ErrY2, float& ErrZ2) const;

  static std::unique_ptr<TPCPadGainCalib> getPadGainCalibDefault();
  static std::unique_ptr<TPCPadGainCalib> getPadGainCalib(const o2::tpc::CalDet<float>& in);

  static std::unique_ptr<o2::tpc::CalibdEdxContainer> getCalibdEdxContainerDefault();

  int registerMemoryForGPU(const void* ptr, size_t size);
  int unregisterMemoryForGPU(const void* ptr);

  const GPUO2InterfaceConfiguration& getConfig() const { return *mConfig; }

 private:
  GPUO2Interface(const GPUO2Interface&);
  GPUO2Interface& operator=(const GPUO2Interface&);

  bool mInitialized = false;
  bool mContinuous = false;

  std::unique_ptr<GPUReconstruction> mRec;              //!
  GPUChainTracking* mChain = nullptr;                   //!
  std::unique_ptr<GPUO2InterfaceConfiguration> mConfig; //!
  std::unique_ptr<GPUTrackingOutputs> mOutputRegions;   //!
};
} // namespace o2::gpu

#endif
