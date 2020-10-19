// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <gsl/span>

#include "TRDWorkflow/TRDTrackletTransformerSpec.h"

#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace trd
{

void TRDTrackletTransformerSpec::init(o2::framework::InitContext& ic)
{
  LOG(info) << "initializing tracklet transformer";
}

void TRDTrackletTransformerSpec::run(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "running tracklet transformer";

  // below line relies on changes made to Tracklet64.h in PR 4766
  auto tracklets = pc.inputs().get<gsl::span<Tracklet64>>("inTracklets");
  // std::vector<Tracklet64> tracklets = pc.inputs().get<std::vector<Tracklet64>>("inTracklets");
  // std::vector<TriggerRecord> triggerRec = pc.inputs().get<std::vector<TriggerRecord>>("triggerRecord");

  std::vector<CalibratedTracklet> calibratedTracklets;
  calibratedTracklets.reserve(tracklets.size());

  // for (int reci=0; reci < triggerRec.size(); reci++)
  // {
  //   LOG(info) << triggerRec[reci].getFirstEntry() << " | " << triggerRec[reci].getNumberOfObjects();
  // }

  LOG(info) << tracklets.size() << " tracklets found!";

  for (const auto& tracklet : tracklets) {

    uint64_t hcid = tracklet.getHCID();
    uint64_t padrow = tracklet.getPadRow();
    uint64_t column = tracklet.getColumn();
    // 0-2048 | units:pad-widths | granularity=1/75 (measured from center pad 10) 1024 is 0/center of pad 10
    uint64_t position = tracklet.getPosition();
    // 0-127 | units:pads/timebin | granularity=1/1000
    uint64_t slope = tracklet.getSlope();

    // calculate raw local chamber space point
    mTransformer.loadPadPlane(hcid);
    float x = mTransformer.calculateX();
    float y = mTransformer.calculateY(hcid, column, position);
    float z = mTransformer.calculateZ(padrow);
    float dy = mTransformer.calculateDy(slope);

    // temporary placeholder for dummy values
    float t0Correction = -0.1;
    float oldLorentzAngle = 0.16;
    float lorentzAngle = -0.14;
    float driftVRatio = 1.1;

    // calibrate x and dy
    float calibratedX = mTransformer.calibrateX(x, t0Correction);
    float calibratedDy = mTransformer.calibrateDy(dy, oldLorentzAngle, lorentzAngle, driftVRatio);

    std::array<float, 3> sectorSpacePoint = mTransformer.transformL2T(hcid, std::array<double, 3>{x, y, z});

    LOG(debug) << "x: " << sectorSpacePoint[0] << " | "
               << "y: " << sectorSpacePoint[1] << " | "
               << "z: " << sectorSpacePoint[2];

    calibratedTracklets.emplace_back(hcid, column, sectorSpacePoint[0], sectorSpacePoint[1], sectorSpacePoint[2], calibratedDy);
  }
  pc.outputs().snapshot(Output{"TRD", "CTRACKLETS", 0, Lifetime::Timeframe}, calibratedTracklets);
}

o2::framework::DataProcessorSpec getTRDTrackletTransformerSpec()
{
  LOG(info) << "getting TRDTrackletTransformerSpec";
  return DataProcessorSpec{
    "TRDTRACKLETTRANSFORMER",
    Inputs{
      InputSpec{"inTracklets", "TRD", "TRACKLETS", 0},
      //  InputSpec{"triggerRecord", "TRD", "TRKTRGRD", 0}
      //  InputSpec{"triggerRecord", "TRD", "TRIGGERREC", 0}
    },
    Outputs{OutputSpec{"TRD", "CTRACKLETS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TRDTrackletTransformerSpec>()},
    Options{}};
}

} //end namespace trd
} //end namespace o2
