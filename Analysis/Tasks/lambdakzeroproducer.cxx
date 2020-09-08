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

/// Cascade builder task: rebuilds cascades
struct lambdakzeroproducer {
    Produces<aod::V0Data> v0data;
    
    OutputObj<TH1F> hEventCounter  {TH1F("hEventCounter"   , "",1, 0, 1)};
    OutputObj<TH1F> hCascCandidate {TH1F("hCascCandidate"  , "",10, 0, 10)};
    
    OutputObj<TH1F> hMassK0Short   {TH1F("hMassK0Short"   , "", 3000, 0.0, 3.0)};
    OutputObj<TH1F> hMassLambda    {TH1F("hMassLambda"    , "", 3000, 0.0, 3.0)};
    OutputObj<TH1F> hMassAntiLambda{TH1F("hMassAntiLambda", "", 3000, 0.0, 3.0)};
    
    //Configurables
    Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
    Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};

    double massPi = TDatabasePDG::Instance()->GetParticle(kPiPlus)->Mass();
    double massKa = TDatabasePDG::Instance()->GetParticle(kKPlus)->Mass();
    double massPr = TDatabasePDG::Instance()->GetParticle(kProton)->Mass();
    
    void process(aod::Collision const& collision, aod::V0s const& V0s, aod::FullTracks const& tracks)
    {
        //Define o2 fitter, 2-prong
        o2::vertexing::DCAFitterN<2> fitter;
        fitter.setBz(d_bz);
        fitter.setPropagateToPCA(true);
        fitter.setMaxR(200.);
        fitter.setMinParamChange(1e-3);
        fitter.setMinRelChi2Change(0.9);
        fitter.setMaxDZIni(1e9);
        fitter.setMaxChi2(1e9);
        fitter.setUseAbsDCA(d_UseAbsDCA);
        
        hEventCounter->Fill(0.5);
        
        TLorentzVector p1, p2, p;
        
        for (auto& V0 : V0s) {
            
            hCascCandidate->Fill(0.5);
            
            auto pTrack = getTrackParCov(V0.posTrack());
            auto nTrack = getTrackParCov(V0.negTrack());
            
            //Calculate DCAs
            double sinAlpha = sin(V0.posTrack().alpha());
            double cosAlpha = cos(V0.posTrack().alpha());
            double globalX = V0.posTrack().x() * cosAlpha - V0.posTrack().y() * sinAlpha;
            double globalY = V0.posTrack().x() * sinAlpha + V0.posTrack().y() * cosAlpha;
            double dcaXYpos = sqrt(pow((globalX - collision.posX()), 2) +
                                   pow((globalY - collision.posY()), 2));
            
            sinAlpha = sin(V0.negTrack().alpha());
            cosAlpha = cos(V0.negTrack().alpha());
            globalX = V0.negTrack().x() * cosAlpha - V0.negTrack().y() * sinAlpha;
            globalY = V0.negTrack().x() * sinAlpha + V0.negTrack().y() * cosAlpha;
            double dcaXYneg = sqrt(pow((globalX - collision.posX()), 2) +
                                   pow((globalY - collision.posY()), 2));
            
            int nCand = fitter.process(pTrack, nTrack);
            
            double v0cosPA=-1, lMassK0Short=-1, lMassLambda=-1, lMassAntiLambda=-1, lPt=-1, lV0Radius = -1;
            
            if (nCand != 0){
                
                fitter.propagateTracksToVertex();
                
                hCascCandidate->Fill(2.5);
                
                const auto& vtx = fitter.getPCACandidate();
                
                std::array<float, 3> pvec0;
                std::array<float, 3> pvec1;
                fitter.getTrack(0).getPxPyPzGlo(pvec0);
                fitter.getTrack(1).getPxPyPzGlo(pvec1);
                
                //CosPA: manual calculation (transform into helper afterwards)
                v0cosPA = (vtx[0]-collision.posX())*(pvec0[0]+pvec1[0]) +
                (vtx[1]-collision.posY())*(pvec0[1]+pvec1[1]) +
                (vtx[2]-collision.posZ())*(pvec0[2]+pvec1[2]);
                
                double lNormR = TMath::Sqrt( TMath::Power( (vtx[0]-collision.posX()) , 2) +
                                            TMath::Power( (vtx[1]-collision.posY()) , 2) +
                                            TMath::Power( (vtx[2]-collision.posZ()) , 2) );
                double lNormP = TMath::Sqrt( TMath::Power( (pvec0[0]+pvec1[0]) , 2) +
                                            TMath::Power( (pvec0[1]+pvec1[1]) , 2) +
                                            TMath::Power( (pvec0[2]+pvec1[2]) , 2) );
                v0cosPA /= (lNormR*lNormP+1e-9);
                
                // calculate invariant masses
                auto track0p2 = pvec0[0]*pvec0[0]+pvec0[1]*pvec0[1]+pvec0[2]*pvec0[2];
                auto track1p2 = pvec1[0]*pvec1[0]+pvec1[1]*pvec1[1]+pvec1[2]*pvec1[2];
                
                auto e0asPr  = TMath::Sqrt(massPr*massPr + track0p2);
                auto e0asPi  = TMath::Sqrt(massPi*massPi + track0p2);
                auto e1asPr  = TMath::Sqrt(massPr*massPr + track1p2);
                auto e1asPi  = TMath::Sqrt(massPi*massPi + track1p2);
                
                lMassK0Short = TMath::Sqrt((e0asPi+e1asPi)*(e0asPi+e1asPi)-
                                                (pvec1[0]+pvec0[0])*(pvec1[0]+pvec0[0])-
                                                (pvec1[1]+pvec0[1])*(pvec1[1]+pvec0[1])-
                                                (pvec1[2]+pvec0[2])*(pvec1[2]+pvec0[2]));
                hMassK0Short->Fill(lMassK0Short);
                lMassLambda = TMath::Sqrt((e0asPr+e1asPi)*(e0asPr+e1asPi)-
                                               (pvec1[0]+pvec0[0])*(pvec1[0]+pvec0[0])-
                                               (pvec1[1]+pvec0[1])*(pvec1[1]+pvec0[1])-
                                               (pvec1[2]+pvec0[2])*(pvec1[2]+pvec0[2]));
                hMassLambda->Fill(lMassLambda);
                lMassAntiLambda = TMath::Sqrt((e0asPi+e1asPr)*(e0asPi+e1asPr)-
                                                   (pvec1[0]+pvec0[0])*(pvec1[0]+pvec0[0])-
                                                   (pvec1[1]+pvec0[1])*(pvec1[1]+pvec0[1])-
                                                   (pvec1[2]+pvec0[2])*(pvec1[2]+pvec0[2]));
                hMassAntiLambda->Fill(lMassAntiLambda);
                
                lPt = TMath::Sqrt( (pvec1[0]+pvec0[0])*(pvec1[0]+pvec0[0])+
                                       (pvec1[1]+pvec0[1])*(pvec1[1]+pvec0[1]) );
                
                lV0Radius = TMath::Sqrt(vtx[0]*vtx[0] + vtx[1]*vtx[1] );
            }
            
            v0data(dcaXYneg, dcaXYpos, lV0Radius,
                   fitter.getChi2AtPCACandidate(), v0cosPA,
                   lMassLambda, lMassAntiLambda, lMassK0Short, lPt);
        }
    }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
    return WorkflowSpec{
        adaptAnalysisTask<lambdakzeroproducer>("lf-lambdakzeroproducer")
    };
}
