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

struct lambdakzeroconsumer {
    OutputObj<TH1F> hMassK0Short   {TH1F("hMassK0Short"   , "", 3000, 0.0, 3.0)};
    OutputObj<TH1F> hMassLambda    {TH1F("hMassLambda"    , "", 3000, 0.0, 3.0)};
    OutputObj<TH1F> hMassAntiLambda{TH1F("hMassAntiLambda", "", 3000, 0.0, 3.0)};
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
        adaptAnalysisTask<lambdakzeroconsumer>("lf-lambdakzeroconsumer")
    };
}
