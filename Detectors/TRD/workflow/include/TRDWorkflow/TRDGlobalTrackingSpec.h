// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TRDGlobalTrackingSpec.h

#ifndef O2_TRD_GLOBALTRACKING
#define O2_TRD_GLOBALTRACKING

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TStopwatch.h"
#include "TRDBase/GeometryFlat.h"
#include "GPUO2Interface.h"
#include "GPUTRDTracker.h"

namespace o2
{
namespace trd
{

class TRDGlobalTracking : public o2::framework::Task
{
 public:
  TRDGlobalTracking(bool useMC, bool useTrkltTransf) : mUseMC(useMC), mUseTrackletTransform(useTrkltTransf) {}
  ~TRDGlobalTracking() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  o2::gpu::GPUTRDTracker* mTracker{nullptr};          ///< TRD tracking engine
  o2::gpu::GPUReconstruction* mRec{nullptr};          ///< GPU reconstruction pointer, handles memory for the tracker
  o2::gpu::GPUChainTracking* mChainTracking{nullptr}; ///< TRD tracker is run in the tracking chain
  std::unique_ptr<GeometryFlat> mFlatGeo{nullptr};    ///< flat TRD geometry
  bool mUseMC{false};                                 ///< MC flag
  bool mUseTrackletTransform{false};                  ///< if true, output from TrackletTransformer is used instead of uncalibrated Tracklet64 directly
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getTRDGlobalTrackingSpec(bool useMC, bool useTrkltTransf);

} // namespace trd
} // namespace o2

#endif /* O2_TRD_TRACKLETREADER */
