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

  auto tracklets = pc.inputs().get<gsl::span<Tracklet64>>("inTracklets");
  // std::vector<TriggerRecord> triggerRec = pc.inputs().get<std::vector<TriggerRecord>>("triggerRecord");

  std::vector<CalibratedTracklet> calibratedTracklets;
  calibratedTracklets.reserve(tracklets.size());

  // temporary. For testing
  // for (int reci=0; reci < triggerRec.size(); reci++)
  // {
  //   LOG(info) << triggerRec[reci].getFirstEntry() << " | " << triggerRec[reci].getNumberOfObjects();
  // }

  LOG(info) << tracklets.size() << " tracklets found!";

  for (const auto& tracklet : tracklets) {
    calibratedTracklets.push_back(mTransformer.transformTracklet(tracklet));
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
    },
    Outputs{OutputSpec{"TRD", "CTRACKLETS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TRDTrackletTransformerSpec>()},
    Options{}};
}

} //end namespace trd
} //end namespace o2
