// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2Interface.h
/// \author David Rohr

#ifndef GPUO2INTERFACE_H
#define GPUO2INTERFACE_H

// Some defines denoting that we are compiling for O2
#ifndef GPUCA_O2_LIB
#define GPUCA_O2_LIB
#endif
#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif

#include <memory>
#include "GPUCommonDef.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
namespace o2
{
namespace tpc
{
struct ClusterNativeAccessFullTPC;
struct ClusterNative;
} // namespace tpc
} // namespace o2

namespace o2
{
namespace gpu
{
class GPUReconstruction;
class GPUChainTracking;
struct GPUO2InterfaceConfiguration;
class TPCFastTransform;

class GPUTPCO2Interface
{
 public:
  GPUTPCO2Interface();
  ~GPUTPCO2Interface();

  int Initialize(const GPUO2InterfaceConfiguration& config);
  void Deinitialize();

  int RunTracking(const o2::tpc::ClusterNativeAccessFullTPC* inputClusters, const GPUTPCGMMergedTrack*& outputTracks, int& nOutputTracks, const GPUTPCGMMergedTrackHit*& outputTrackClusters);
  void Cleanup();

  bool GetParamContinuous() { return (mContinuous); }
  void GetClusterErrors2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const;

 private:
  GPUTPCO2Interface(const GPUTPCO2Interface&);
  GPUTPCO2Interface& operator=(const GPUTPCO2Interface&);

  bool mInitialized = false;
  bool mDumpEvents = false;
  bool mContinuous = false;

  std::unique_ptr<GPUReconstruction> mRec;
  GPUChainTracking* mChain = nullptr;
  std::unique_ptr<GPUO2InterfaceConfiguration> mConfig;
};
} // namespace gpu
} // namespace o2

#endif
