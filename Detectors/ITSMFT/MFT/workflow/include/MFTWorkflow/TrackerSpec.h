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

/// @file   TrackerSpec.h

#ifndef O2_MFT_TRACKERDPL_H_
#define O2_MFT_TRACKERDPL_H_

#include "MFTTracking/Tracker.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include "Framework/DataProcessorSpec.h"
#include "MFTTracking/TrackCA.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "TStopwatch.h"

namespace o2
{
namespace mft
{
using o2::mft::TrackLTF;

class TrackerDPL : public o2::framework::Task
{

 public:
  TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC) : mGGCCDBRequest(gr), mUseMC(useMC) {}
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  bool mUseMC = false;
  bool mFieldOn = true;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  std::unique_ptr<o2::parameters::GRPObject> mGRP = nullptr;
  std::unique_ptr<o2::mft::Tracker<TrackLTF>> mTracker = nullptr;
  std::unique_ptr<o2::mft::Tracker<TrackLTFL>> mTrackerL = nullptr;
  enum TimerIDs { SWTot,
                  SWLoadData,
                  SWFindLTFTracks,
                  SWFindCATracks,
                  SWFitTracks,
                  SWComputeLabels,
                  NStopWatches };
  static constexpr std::string_view TimerName[] = {"Total",
                                                   "LoadData",
                                                   "FindLTFTracks",
                                                   "FindCATracks",
                                                   "FitTracks",
                                                   "ComputeLabels"};
  TStopwatch mTimer[NStopWatches];
};

/// create a processor spec
/// run MFT CA tracker
o2::framework::DataProcessorSpec getTrackerSpec(bool useMC);

} // namespace mft
} // namespace o2

#endif /* O2_MFT_TRACKERDPL */
