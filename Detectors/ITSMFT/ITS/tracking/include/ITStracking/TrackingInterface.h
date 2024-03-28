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
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsCalibration/MeanVertexObject.h"

#include "DetectorsBase/GRPGeomHelper.h"

#include "GPUDataTypes.h"
#include "GPUO2Interface.h"
#include "GPUChainITS.h"

namespace o2::its
{
class ITSTrackingInterface
{
 public:
  ITSTrackingInterface(std::shared_ptr<o2::base::GRPGeomRequest> gr,
                       bool isMC,
                       int trgType,
                       const TrackingMode trMode,
                       const bool overrBeamEst)
    : mGGCCDBRequest(gr),
      mIsMC{isMC},
      mUseTriggers{trgType},
      mMode{trMode},
      mOverrideBeamEstimation{overrBeamEst}
  {
  }

  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }
  void setMeanVertex(const o2::dataformats::MeanVertexObject* v)
  {
    if (!v) {
      return;
    }
    mMeanVertex = v;
  }
  // Task handles
  void initialise();
  void run(framework::ProcessingContext& pc);
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj);
  // Custom
  void setTraitsFromProvider(VertexerTraits*, TrackerTraits*, TimeFrame*);

 private:
  bool mIsMC = false;
  bool mRunVertexer = true;
  bool mCosmicsProcessing = false;
  int mUseTriggers = 0;
  TrackingMode mMode = TrackingMode::Sync;
  bool mOverrideBeamEstimation = false;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  std::unique_ptr<Tracker> mTracker = nullptr;
  std::unique_ptr<Vertexer> mVertexer = nullptr;
  TimeFrame* mTimeFrame = nullptr;
  const o2::dataformats::MeanVertexObject* mMeanVertex;
};

} // namespace o2::its
#endif // O2_ITS_TRACKINGINTERFACE