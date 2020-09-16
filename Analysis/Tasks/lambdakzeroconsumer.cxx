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
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Analysis/RecoDecay.h"
#include "Analysis/trackUtilities.h"
#include "Analysis/StrangenessTables.h"
#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include "PID/PIDResponse.h"
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

struct lambdakzeroQA {
  //Basic checks
  OutputObj<TH1F> hMassK0Short{TH1F("hMassK0Short", "", 3000, 0.0, 3.0)};
  OutputObj<TH1F> hMassLambda{TH1F("hMassLambda", "", 3000, 0.0, 3.0)};
  OutputObj<TH1F> hMassAntiLambda{TH1F("hMassAntiLambda", "", 3000, 0.0, 3.0)};

  OutputObj<TH1F> hV0Radius{TH1F("hV0Radius", "", 1000, 0.0, 100)};
  OutputObj<TH1F> hV0CosPA{TH1F("hV0CosPA", "", 1000, 0.95, 1.0)};
  OutputObj<TH1F> hDCAPosToPV{TH1F("hDCAPosToPV", "", 1000, 0.0, 10.0)};
  OutputObj<TH1F> hDCANegToPV{TH1F("hDCANegToPV", "", 1000, 0.0, 10.0)};
  OutputObj<TH1F> hDCAV0Dau{TH1F("hDCAV0Dau", "", 1000, 0.0, 10.0)};

  void process(aod::Collision const& collision, soa::Join<aod::V0s, aod::V0DataExt> const& fullV0s)
  {
    for (auto& v0 : fullV0s) {
      hMassLambda->Fill(v0.mLambda());
      hMassAntiLambda->Fill(v0.mAntiLambda());
      hMassK0Short->Fill(v0.mK0Short());

      hV0Radius->Fill(v0.v0radius());
      hV0CosPA->Fill(v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()));

      hDCAPosToPV->Fill(v0.dcapostopv());
      hDCANegToPV->Fill(v0.dcanegtopv());
      hDCAV0Dau->Fill(v0.dcaV0daughters());
    }
  }
};

struct lambdakzeroconsumer {
  OutputObj<TH2F> h2dMassK0Short{TH2F("h2dMassK0Short", "", 200, 0, 10, 200, 0.450, 0.550)};
  OutputObj<TH2F> h2dMassLambda{TH2F("h2dMassLambda", "", 200, 0, 10, 200, 1.115 - 0.100, 1.115 + 0.100)};
  OutputObj<TH2F> h2dMassAntiLambda{TH2F("h2dMassAntiLambda", "", 200, 0, 10, 200, 1.115 - 0.100, 1.115 + 0.100)};

  //Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.995, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};

  Filter preFilterV0 = aod::v0data::dcapostopv > dcapostopv&&
                                                   aod::v0data::dcanegtopv > dcanegtopv&& aod::v0data::dcaV0daughters < dcav0dau;

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::V0s, aod::V0DataExt>> const& fullV0s)
  {
    for (auto& v0 : fullV0s) {
      //FIXME: could not find out how to filter cosPA and radius variables (dynamic columns)
      if (v0.v0radius() > v0radius && v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) > v0cospa) {
        h2dMassLambda->Fill(v0.pt(), v0.mLambda());
        h2dMassAntiLambda->Fill(v0.pt(), v0.mAntiLambda());
        h2dMassK0Short->Fill(v0.pt(), v0.mK0Short());
      }
    }
  }
};

/// Extends the v0data table with expression columns
struct lambdakzeroinitializer {
  Spawns<aod::V0DataExt> v0dataext;
  void init(InitContext const&) {}
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<lambdakzeroconsumer>("lf-lambdakzeroconsumer"),
    adaptAnalysisTask<lambdakzeroQA>("lf-lambdakzeroQA"),
    adaptAnalysisTask<lambdakzeroinitializer>("lf-lambdakzeroinitializer")};
}
