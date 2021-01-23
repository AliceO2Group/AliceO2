// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Example V0 analysis task
// ========================
//
// This code loops over a V0Data table and produces some
// standard analysis output. It requires either
// the lambdakzerofinder or the lambdakzeroproducer tasks
// to have been executed in the workflow (before).
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "AnalysisCore/RecoDecay.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisDataModel/StrangenessTables.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"

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

  void process(aod::Collision const& collision, aod::V0DataExt const& fullV0s)
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

struct lambdakzeroanalysis {
  OutputObj<TH3F> h3dMassK0Short{TH3F("h3dMassK0Short", "", 20, 0, 100, 200, 0, 10, 200, 0.450, 0.550)};
  OutputObj<TH3F> h3dMassLambda{TH3F("h3dMassLambda", "", 20, 0, 100, 200, 0, 10, 200, 1.115 - 0.100, 1.115 + 0.100)};
  OutputObj<TH3F> h3dMassAntiLambda{TH3F("h3dMassAntiLambda", "", 20, 0, 100, 200, 0, 10, 200, 1.115 - 0.100, 1.115 + 0.100)};

  //Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.995, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};
  Configurable<float> rapidity{"rapidity", 0.5, "rapidity"};

  // Filter preFilterV0 = aod::v0data::dcapostopv > dcapostopv&&
  //                                                  aod::v0data::dcanegtopv > dcanegtopv&& aod::v0data::dcaV0daughters < dcav0dau; we can use this again once (and if) math expressions can be used there

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator const& collision, aod::V0DataExt const& fullV0s)
  {
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    for (auto& v0 : fullV0s) {
      //FIXME: could not find out how to filter cosPA and radius variables (dynamic columns)
      if (v0.v0radius() > v0radius && v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) > v0cospa) {
        if (fabs(v0.dcapostopv()) < dcapostopv) {
          continue;
        }
        if (fabs(v0.dcanegtopv()) < dcanegtopv) {
          continue;
        }
        if (fabs(v0.dcaV0daughters()) > dcav0dau) {
          continue;
        }
        if (TMath::Abs(v0.yLambda()) < rapidity) {
          h3dMassLambda->Fill(collision.centV0M(), v0.pt(), v0.mLambda());
          h3dMassAntiLambda->Fill(collision.centV0M(), v0.pt(), v0.mAntiLambda());
        }
        if (TMath::Abs(v0.yK0Short()) < rapidity) {
          h3dMassK0Short->Fill(collision.centV0M(), v0.pt(), v0.mK0Short());
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<lambdakzeroanalysis>("lf-lambdakzeroanalysis"),
    adaptAnalysisTask<lambdakzeroQA>("lf-lambdakzeroQA")};
}
