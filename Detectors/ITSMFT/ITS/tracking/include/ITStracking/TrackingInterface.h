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

#ifndef O2_ITS_TRACKINGINTERFACE
#define O2_ITS_TRACKINGINTERFACE

#include "Framework/DataProcessorSpec.h"

#include "ITStracking/TimeFrame.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/BoundedAllocator.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsCalibration/MeanVertexObject.h"

#include "GPUDataTypesIO.h"
#include "GPUO2ExternalUser.h"
#include "GPUChainITS.h"

#include <oneapi/tbb/task_arena.h>

namespace o2::its
{
class ITSTrackingInterface
{
  static constexpr int NLayers{7};
  using VertexerN = Vertexer<NLayers>;
  using VertexerTraitsN = VertexerTraits<NLayers>;
  using TrackerN = Tracker<NLayers>;
  using TrackerTraitsN = TrackerTraits<NLayers>;
  using TimeFrameN = TimeFrame<NLayers>;

 public:
  ITSTrackingInterface(bool isMC,
                       bool doStag,
                       int trgType,
                       const bool overrBeamEst)
    : mIsMC{isMC},
      mDoStaggering(doStag),
      mUseTriggers{trgType},
      mOverrideBeamEstimation{overrBeamEst} {}

  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }
  void setMeanVertex(const o2::dataformats::MeanVertexObject* v)
  {
    if (v == nullptr) {
      LOGP(error, "Mean Vertex Object is nullptr");
      return;
    } else {
      LOGP(info, "Mean Vertex set with x: {} y: {}", v->getX(), v->getY());
    }
    mMeanVertex = v;
  }
  // Task callbacks
  void initialise();
  void run(framework::ProcessingContext& pc);
  void printSummary() const;

  virtual void updateTimeDependentParams(framework::ProcessingContext& pc);
  virtual void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj);

  // Custom
  void setTraitsFromProvider(VertexerTraitsN*, TrackerTraitsN*, TimeFrameN*);
  void setTrackingMode(TrackingMode::Type mode = TrackingMode::Unset) { mMode = mode; }

  auto getTracker() const { return mTracker.get(); }
  auto getVertexer() const { return mVertexer.get(); }

  TimeFrameN* mTimeFrame = nullptr;

 protected:
  virtual void loadROF(gsl::span<const itsmft::ROFRecord>& trackROFspan,
                       gsl::span<const itsmft::CompClusterExt> clusters,
                       gsl::span<const unsigned char>::iterator& pattIt,
                       int layer,
                       const dataformats::MCTruthContainer<MCCompLabel>* mcLabels);

 private:
  bool mIsMC = false;
  bool mDoStaggering = false;
  bool mRunVertexer = true;
  bool mCosmicsProcessing = false;
  int mUseTriggers = 0;
  std::vector<o2::framework::InputSpec> mFilter;
  TrackingMode::Type mMode = TrackingMode::Unset;
  bool mOverrideBeamEstimation = false;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  std::unique_ptr<TrackerN> mTracker = nullptr;
  std::unique_ptr<VertexerN> mVertexer = nullptr;
  const o2::dataformats::MeanVertexObject* mMeanVertex;
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;
  std::shared_ptr<tbb::task_arena> mTaskArena;
};

} // namespace o2::its
#endif // O2_ITS_TRACKINGINTERFACE
