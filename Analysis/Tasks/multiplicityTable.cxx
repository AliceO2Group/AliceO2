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

struct MultiplicityTableTask {
  Produces<aod::Mults> mult;

  aod::Run2V0 getVZero(aod::BC const& bc, aod::Run2V0s const& vzeros)
  {
    for (auto& vzero : vzeros)
      if (vzero.bc() == bc)
        return vzero;
    aod::Run2V0 dummy;
    return dummy;
  }

  aod::Zdc getZdc(aod::BC const& bc, aod::Zdcs const& zdcs)
  {
    for (auto& zdc : zdcs)
      if (zdc.bc() == bc)
        return zdc;
    aod::Zdc dummy;
    return dummy;
  }

  void process(aod::Collision const& collision, aod::BCs const& bcs, aod::Zdcs const& zdcs, aod::Run2V0s const& vzeros, aod::Tracks const& tracks)
  {
    auto zdc = getZdc(collision.bc(), zdcs);
    auto vzero = getVZero(collision.bc(), vzeros);
    float multV0A = vzero.multA();
    float multV0C = vzero.multC();
    float multZNA = zdc.energyCommonZNA();
    float multZNC = zdc.energyCommonZNC();

    int multTracklets = 0;
    for (auto& tr : tracks)
      if (tr.trackType() == o2::aod::track::TrackTypeEnum::Run2Tracklet)
        multTracklets++;

    LOGF(debug, "multV0A=%5.0f multV0C=%5.0f multZNA=%6.0f multZNC=%6.0f multTracklets=%i", multV0A, multV0C, multZNA, multZNC, multTracklets);
    // fill multiplicity columns
    mult(multV0A, multV0C, multZNA, multZNC, multTracklets);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<MultiplicityTableTask>("multiplicity-table")};
}
