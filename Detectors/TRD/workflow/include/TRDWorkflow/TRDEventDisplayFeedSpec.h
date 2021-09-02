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

#include <gsl/span>

#include "TRDBase/Geometry.h"
#include "TRDBase/TrackletTransformer.h"

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"

#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Digit.h"

#include "ITStracking/json.h"

using json = nlohmann::json;

namespace o2
{
namespace trd
{

class TRDEventDisplayFeedSpec : public o2::framework::Task
{
 public:
  TRDEventDisplayFeedSpec(int nEventsMax) : mNeventsMax(nEventsMax){};
  ~TRDEventDisplayFeedSpec() override = default;
  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;
  json getTracksJson(gsl::span<const TrackTRD> tracks, gsl::span<const Tracklet64> tracklets, gsl::span<const TrackTriggerRecord> trackTrigRecs, int iEvent);
  json getTrackletsJson(gsl::span<const Tracklet64> tracklets, int iEvent);
  void writeDigits(gsl::span<const Digit> digits, int iEvent);

 private:
  TrackletTransformer mTransformer;
  o2::trd::Geometry* mGeo;

  std::map<std::string, std::string> mTrackletMap;
  std::bitset<constants::MAXCHAMBER> mUsedDetectors;
  gsl::span<const TriggerRecord> mTrigRecs;

  float mBz;
  int mNeventsMax;
};

o2::framework::DataProcessorSpec getTRDEventDisplayFeedSpec(int nEventsMax);

} // end namespace trd
} // end namespace o2
