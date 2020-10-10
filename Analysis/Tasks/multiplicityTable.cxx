// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Multiplicity.h"
#include "iostream"
using namespace o2;
using namespace o2::framework;

struct MultiplicityTableTaskIndexed {
  Produces<aod::Mults> mult;
  Partition<aod::Tracks> tracklets = (aod::track::trackType == static_cast<uint8_t>(o2::aod::track::TrackTypeEnum::Run2Tracklet));

  void process(aod::Run2MatchedSparse::iterator const& collision, aod::Tracks const& tracks, aod::BCs const&, aod::Zdcs const&, aod::Run2V0s const&)
  {
    float multV0A = -1.f;
    float multV0C = -1.f;
    float multZNA = -1.f;
    float multZNC = -1.f;
    int multTracklets = tracklets.size();

    if (collision.has_run2v0()) {
      auto v0 = collision.run2v0();
      multV0A = v0.multA();
      multV0C = v0.multC();
    }
    if (collision.has_zdc()) {
      auto zdc = collision.zdc();
      multZNA = zdc.energyCommonZNA();
      multZNC = zdc.energyCommonZNC();
    }
    LOGF(debug, "multV0A=%5.0f multV0C=%5.0f multZNA=%6.0f multZNC=%6.0f multTracklets=%i", multV0A, multV0C, multZNA, multZNC, multTracklets);
    mult(multV0A, multV0C, multZNA, multZNC, multTracklets);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<MultiplicityTableTaskIndexed>("multiplicity-table")};
}
