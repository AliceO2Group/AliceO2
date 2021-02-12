// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2FakeClasses.h
/// \author David Rohr

#ifndef O2_GPU_GPUO2FAKECLASSES_H
#define O2_GPU_GPUO2FAKECLASSES_H

#include "GPUCommonDef.h"
#include "GPUDataTypes.h"

// These are some dummies of O2 classes needed by AliGPU, to be used when O2 header unavailable

namespace o2
{
class MCCompLabel
{
};
namespace gpu
{
} // namespace gpu
namespace tpc
{
class Digit
{
};
class TrackTPC
{
};
struct ClusterNative {
  GPUd() static float getTime() { return 0.f; }
  GPUd() static float getPad() { return 0.f; }
  GPUd() static int getFlags() { return 0; }
  GPUd() static void setTimeFlags(float t, int f) {}
  GPUd() static void setPad(float p) {}
  GPUd() static void setSigmaTime(float s) {}
  GPUd() static void setSigmaPad(float s) {}

  unsigned char qTot, qMax;
};
struct ClusterNativeAccess {
  const ClusterNative* clustersLinear;
  const ClusterNative* clusters[GPUCA_NSLICES][GPUCA_ROW_COUNT];
  unsigned int nClusters[GPUCA_NSLICES][GPUCA_ROW_COUNT];
  unsigned int nClustersSector[GPUCA_NSLICES];
  unsigned int clusterOffset[GPUCA_NSLICES][GPUCA_ROW_COUNT];
  unsigned int nClustersTotal;
  void setOffsetPtrs() {}
};
#ifndef __OPENCL__
struct TPCZSHDR {
  static const unsigned int TPC_ZS_PAGE_SIZE = 8192;
};
#endif
} // namespace tpc
namespace base
{
struct MatBudget {
};
class MatLayerCylSet
{
};
} // namespace base
namespace trd
{
class GeometryFlat
{
};
} // namespace trd
namespace dataformats
{
template <class T>
class ConstMCTruthContainerView
{
};
} // namespace dataformats
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class TPCCFCalibration
{
};
class TPCdEdxCalibrationSplines
{
};
class GPUFakeEmpty
{
};
class GPUITSFitter
{
};
class GPUTPCConvert
{
};
class GPUTPCCompression
{
 public:
  GPUFakeEmpty* mOutput;
};
class GPUTPCClusterFinder
{
};
class GPUTrackingRefitProcessor
{
};
struct GPUTPCCFChainContext {
};
struct GPUTPCDigitsMCInput {
};
struct TPCPadGainCalib {
};
#ifndef __OPENCL__
struct GPUParam;
class GPUTPCClusterStatistics
{
 public:
  void Finish() {}
  void RunStatistics(const o2::tpc::ClusterNativeAccess* clustersNative, const GPUFakeEmpty* clustersCompressed, const GPUParam& param) {}
};
#endif
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
