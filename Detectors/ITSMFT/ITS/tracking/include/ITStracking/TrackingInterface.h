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

#include "GPUDataTypes.h"
#include "GPUO2Interface.h"
#include "GPUChainITS.h"

namespace o2::its
{
class ITSTrackingInterface
{
 public:
  ITSTrackingInterface(bool isMC,
                       int trgType,
                       const bool overrBeamEst)
    : mIsMC{isMC},
      mUseTriggers{trgType},
      mOverrideBeamEstimation{overrBeamEst}
  {
  }

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
  template <bool isGPU = false>
  void run(framework::ProcessingContext& pc);

  virtual void updateTimeDependentParams(framework::ProcessingContext& pc);
  virtual void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj);

  // Custom
  void setTraitsFromProvider(VertexerTraits*, TrackerTraits*, TimeFrame*);
  void setTrackingMode(TrackingMode mode = TrackingMode::Unset)
  {
    if (mode == TrackingMode::Unset) {
      LOGP(fatal, "ITS Tracking mode Unset is meant to be a default. Specify the mode");
    }
    mMode = mode;
  }

  TimeFrame* mTimeFrame = nullptr;

 protected:
  virtual void loadROF(gsl::span<itsmft::ROFRecord>& trackROFspan,
                       gsl::span<const itsmft::CompClusterExt> clusters,
                       gsl::span<const unsigned char>::iterator& pattIt,
                       const dataformats::MCTruthContainer<MCCompLabel>* mcLabels);
  void getConfiguration(framework::ProcessingContext& pc);

 private:
  bool mIsMC = false;
  bool mRunVertexer = true;
  bool mCosmicsProcessing = false;
  int mUseTriggers = 0;
  TrackingMode mMode = TrackingMode::Unset;
  bool mOverrideBeamEstimation = false;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  std::unique_ptr<Tracker> mTracker = nullptr;
  std::unique_ptr<Vertexer> mVertexer = nullptr;
  const o2::dataformats::MeanVertexObject* mMeanVertex;
};

} // namespace o2::its
#endif // O2_ITS_TRACKINGINTERFACE
