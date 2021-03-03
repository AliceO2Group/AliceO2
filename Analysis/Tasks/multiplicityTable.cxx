// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ConfigParamSpec.h"

using namespace o2;
using namespace o2::framework;

// custom configurable for switching between run2 and run3 selection types
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"selection-run", VariantType::Int, 2, {"selection type: 2 - run 2, 3 - run 3"}});
}

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "iostream"

struct MultiplicityTableTaskIndexed {
  Produces<aod::Mults> mult;
  Partition<aod::Tracks> tracklets = (aod::track::trackType == static_cast<uint8_t>(o2::aod::track::TrackTypeEnum::Run2Tracklet));

  void process(aod::Run2MatchedSparse::iterator const& collision, aod::Tracks const& tracks, aod::BCs const&, aod::Zdcs const&, aod::FV0As const& fv0as, aod::FV0Cs const& fv0cs, aod::FT0s const& ft0s)
  {
    float multV0A = -1.f;
    float multV0C = -1.f;
    float multT0A = -1.f;
    float multT0C = -1.f;
    float multZNA = -1.f;
    float multZNC = -1.f;
    int multTracklets = tracklets.size();

    if (collision.has_fv0a()) {
      auto v0a = collision.fv0a();
      for (int i = 0; i < 48; i++) {
        multV0A += v0a.amplitude()[i];
      }
    }
    if (collision.has_fv0c()) {
      auto v0c = collision.fv0c();
      for (int i = 0; i < 32; i++) {
        multV0C += v0c.amplitude()[i];
      }
    }
    if (collision.has_ft0()) {
      auto ft0 = collision.ft0();
      for (int i = 0; i < 96; i++) {
        multT0A += ft0.amplitudeA()[i];
      }
      for (int i = 0; i < 112; i++) {
        multT0C += ft0.amplitudeC()[i];
      }
    }
    if (collision.has_zdc()) {
      auto zdc = collision.zdc();
      multZNA = zdc.energyCommonZNA();
      multZNC = zdc.energyCommonZNC();
    }
    LOGF(debug, "multV0A=%5.0f multV0C=%5.0f multT0A=%5.0f multT0C=%5.0f multZNA=%6.0f multZNC=%6.0f multTracklets=%i", multV0A, multV0C, multT0A, multT0C, multZNA, multZNC, multTracklets);
    mult(multV0A, multV0C, multT0A, multT0C, multZNA, multZNC, multTracklets);
  }
};

struct MultiplicityTableTaskRun3 {
  Produces<aod::Mults> mult;
  Partition<aod::Tracks> tracklets = (aod::track::trackType == static_cast<uint8_t>(o2::aod::track::TrackTypeEnum::Run2Tracklet));

  void process(soa::Join<aod::Collisions, aod::EvSels> const& collisions, aod::Tracks const& tracks, aod::BCs const& bcs, aod::Zdcs const& zdcs, aod::FV0As const& fv0as, aod::FT0s const& ft0s)
  {
    for (auto& collision : collisions) {
      float multV0A = -1.f;
      float multV0C = -1.f;
      float multT0A = -1.f;
      float multT0C = -1.f;
      float multZNA = -1.f;
      float multZNC = -1.f;
      int multTracklets = tracklets.size();

      const float* aAmplitudesA;
      const float* aAmplitudesC;

      // using FT0 row index from event selection task
      int64_t foundFT0 = collision.foundFT0();

      if (foundFT0 != -1) {
        auto ft0 = ft0s.iteratorAt(foundFT0);
        aAmplitudesA = ft0.amplitudeA();
        aAmplitudesC = ft0.amplitudeC();
        for (int i = 0; i < 96; i++) {
          multT0A += aAmplitudesA[i];
        }
        for (int i = 0; i < 112; i++) {
          multT0C += aAmplitudesC[i];
        }
      }

      LOGF(debug, "multV0A=%5.0f multV0C=%5.0f multT0A=%5.0f multT0C=%5.0f multZNA=%6.0f multZNC=%6.0f multTracklets=%i", multV0A, multV0C, multT0A, multT0C, multZNA, multZNC, multTracklets);
      mult(multV0A, multV0C, multT0A, multT0C, multZNA, multZNC, multTracklets);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  if (cfgc.options().get<int>("selection-run") == 2) {
    return WorkflowSpec{adaptAnalysisTask<MultiplicityTableTaskIndexed>(cfgc, "multiplicity-table")};
  } else {
    return WorkflowSpec{adaptAnalysisTask<MultiplicityTableTaskRun3>(cfgc, "multiplicity-table")};
  }
}
