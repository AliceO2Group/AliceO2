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
#include "Analysis/SecondaryVertexHF.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Analysis/RecoDecay.h"
#include "Analysis/trackUtilities.h"
#include "Analysis/StrangenessTables.h"

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
    OutputObj<TH1F> hMassK0Short   {TH1F("hMassK0Short"   , "", 3000, 0.0, 3.0)};
    OutputObj<TH1F> hMassLambda    {TH1F("hMassLambda"    , "", 3000, 0.0, 3.0)};
    OutputObj<TH1F> hMassAntiLambda{TH1F("hMassAntiLambda", "", 3000, 0.0, 3.0)};
    
    OutputObj<TH1F> hV0Radius     {TH1F("hV0Radius",   "", 1000, 0.0, 100)};
    OutputObj<TH1F> hV0CosPA      {TH1F("hV0CosPA",    "", 1000, 0.95, 1.0)};
    OutputObj<TH1F> hDCAPosToPV   {TH1F("hDCAPosToPV", "", 1000, 0.0, 10.0)};
    OutputObj<TH1F> hDCANegToPV   {TH1F("hDCANegToPV", "", 1000, 0.0, 10.0)};
    OutputObj<TH1F> hDCAV0Dau     {TH1F("hDCAV0Dau", "", 1000, 0.0, 10.0)};

    void process(aod::Collision const& collision, soa::Join<aod::V0s, aod::V0Data> const& fullV0s)
    {
        for (auto& v0 : fullV0s) {
            hMassLambda->Fill(v0.MassAsLambdas());
            hMassAntiLambda->Fill(v0.MassAsAntiLambdas());
            hMassK0Short->Fill(v0.MassAsK0Shorts());
            
            hV0Radius->Fill(v0.V0Radii());
            hV0CosPA->Fill(v0.V0CosPAs());
            hDCAPosToPV->Fill(v0.DCAPosToPVs());
            hDCANegToPV->Fill(v0.DCANegToPVs());
            hDCAV0Dau->Fill(v0.DCAV0Daughters());
        }
    }
};

struct lambdakzeroconsumer {
    OutputObj<TH2F> h2dMassK0Short   {TH2F("h2dMassK0Short"   , "", 200,0,10, 200, 0.450, 0.550)};
    OutputObj<TH2F> h2dMassLambda    {TH2F("h2dMassLambda"    , "", 200,0,10, 200, 1.115-0.100,1.115+0.100)};
    OutputObj<TH2F> h2dMassAntiLambda{TH2F("h2dMassAntiLambda", "", 200,0,10, 200, 1.115-0.100,1.115+0.100)};
    
    void process(aod::Collision const& collision, soa::Join<aod::V0s, aod::V0Data> const& fullV0s)
    {
        for (auto& v0 : fullV0s) {
            h2dMassLambda->Fill(v0.Pts(), v0.MassAsLambdas());
            h2dMassAntiLambda->Fill(v0.Pts(), v0.MassAsAntiLambdas());
            h2dMassK0Short->Fill(v0.Pts(), v0.MassAsK0Shorts());
            
            hMassLambda->Fill(v0.MassAsLambdas());
            hMassAntiLambda->Fill(v0.MassAsAntiLambdas());
            hMassK0Short->Fill(v0.MassAsK0Shorts());
        }
    }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
    return WorkflowSpec{
        adaptAnalysisTask<lambdakzeroQA>("lf-lambdakzeroQA")
        adaptAnalysisTask<lambdakzeroconsumer>("lf-lambdakzeroconsumer")
    };
}
