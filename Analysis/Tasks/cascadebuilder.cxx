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

struct TaskIndexer {
    Produces<aod::TransientV0s> transientV0s;
    Produces<aod::TransientCascades> transientCascades;
    
    void process(aod::StoredV0s const& v0s, aod::StoredCascades const& cascades, aod::FullTracks const& tracks)
    {
        for (auto& v0 : v0s) {
            transientV0s(v0.posTrack().collisionId());
        }
        for (auto& cascade : cascades) {
            transientCascades(cascade.bachelor().collisionId());
        }
    }
};

namespace o2::aod
{

} // namespace o2::aod

/// Cascade builder task: rebuilds cascades
struct cascadebuilder {
    Produces<aod::CascData> cascdata;
    
    OutputObj<TH1F> hEventCounter  {TH1F("hEventCounter"   , "",1, 0, 1)};
    OutputObj<TH1F> hCascCandidate {TH1F("hCascCandidate"  , "",10, 0, 10)};
    
    OutputObj<TH1F> hMassXiMinus   {TH1F("hMassXiMinus"   , "", 2000, 0,2)};
    OutputObj<TH1F> hMassXiPlus    {TH1F("hMassXiPlus"    , "", 2000, 0,2)};
    OutputObj<TH1F> hMassOmegaMinus{TH1F("hMassOmegaMinus", "", 2000, 0,2)};
    OutputObj<TH1F> hMassOmegaPlus {TH1F("hMassOmegaPlus" , "", 2000, 0,2)};
    
    //Configurables
    Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
    Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};
    /*
     Configurable<double> DCANegToPV{"DCANegToPV", 0.1, "Min Neg Track DCA to PV"};
     Configurable<double> DCAPosToPV{"DCAPosToPV", 0.1, "Min Pos Track DCA to PV"};
     Configurable<double> DCABachToPV{"DCABachToPV", 0.1, "Min Bach Track DCA to PV"};
     Configurable<double> V0MinRadius{"V0MinRadius", 0.9, "Minimum V0 Radius"};
     Configurable<double> CascMinRadius{"CascMinRadius", 1.2, "Minimum Cascade Radius"};
     Configurable<double> DCAV0Daughters{"DCAV0Daughters", 1.5, "Max V0 Daughter DCA"};
     Configurable<double> DCAV0Daughters{"DCACascDaughters", 1.5, "Max Casc Daughter DCA"};
     Configurable<double> DCAV0ToPV{"DCAV0ToPV", 0.05, "Min V0 DCA to PV"};
     Configurable<double> V0CosPA{"V0CosPA",   0.99, "V0 CosPA"};
     Configurable<double> CascCosPA{"CascCosPA", 0.99, "Casc CosPA"};
     Configurable<double> V0InvMassWindow{"V0InvMassWindow", 0.008, "V0InvMassWindow"};
     
     Configurable<int> MinNbrCrossedRows{"MinNbrCrossedRows", 70, "MinNbrCrossedRows"};
     */
    double massPi = TDatabasePDG::Instance()->GetParticle(kPiPlus)->Mass();
    double massKa = TDatabasePDG::Instance()->GetParticle(kKPlus)->Mass();
    double massPr = TDatabasePDG::Instance()->GetParticle(kProton)->Mass();
    
    void process(aod::Collision const& collision, aod::V0s const& V0s, aod::Cascades const& Cascades, aod::FullTracks const& tracks)
    {
        //Define o2 fitter, 2-prong
        o2::vertexing::DCAFitterN<2> fitterV0, fitterCasc;
        fitterV0.setBz(d_bz);
        fitterV0.setPropagateToPCA(true);
        fitterV0.setMaxR(200.);
        fitterV0.setMinParamChange(1e-3);
        fitterV0.setMinRelChi2Change(0.9);
        fitterV0.setMaxDZIni(1e9);
        fitterV0.setMaxChi2(1e9);
        fitterV0.setUseAbsDCA(d_UseAbsDCA);
        
        fitterCasc.setBz(d_bz);
        fitterCasc.setPropagateToPCA(true);
        fitterCasc.setMaxR(200.);
        fitterCasc.setMinParamChange(1e-3);
        fitterCasc.setMinRelChi2Change(0.9);
        fitterCasc.setMaxDZIni(1e9);
        fitterCasc.setMaxChi2(1e9);
        fitterCasc.setUseAbsDCA(d_UseAbsDCA);
        
        hEventCounter->Fill(0.5);
        
        TLorentzVector p1, p2, p;
        
        for (auto& casc : Cascades) {
            
            hCascCandidate->Fill(0.5);
            
            auto pTrack = getTrackParCov(casc.v0().posTrack());
            auto nTrack = getTrackParCov(casc.v0().negTrack());
            auto bTrack = getTrackParCov(casc.bachelor());
            
            //Calculate DCAs
            double sinAlpha = sin(casc.v0().posTrack().alpha());
            double cosAlpha = cos(casc.v0().posTrack().alpha());
            double globalX = casc.v0().posTrack().x() * cosAlpha - casc.v0().posTrack().y() * sinAlpha;
            double globalY = casc.v0().posTrack().x() * sinAlpha + casc.v0().posTrack().y() * cosAlpha;
            double dcaXYpos = sqrt(pow((globalX - collision.posX()), 2) +
                                   pow((globalY - collision.posY()), 2));
            
            sinAlpha = sin(casc.v0().negTrack().alpha());
            cosAlpha = cos(casc.v0().negTrack().alpha());
            globalX = casc.v0().negTrack().x() * cosAlpha - casc.v0().negTrack().y() * sinAlpha;
            globalY = casc.v0().negTrack().x() * sinAlpha + casc.v0().negTrack().y() * cosAlpha;
            double dcaXYneg = sqrt(pow((globalX - collision.posX()), 2) +
                                   pow((globalY - collision.posY()), 2));
            
            sinAlpha = sin(casc.bachelor().alpha());
            cosAlpha = cos(casc.bachelor().alpha());
            globalX = casc.bachelor().x() * cosAlpha - casc.bachelor().y() * sinAlpha;
            globalY = casc.bachelor().x() * sinAlpha + casc.bachelor().y() * cosAlpha;
            double dcaXYbach = sqrt(pow((globalX - collision.posX()), 2) +
                                    pow((globalY - collision.posY()), 2));
            
            hCascCandidate->Fill(1.5);
            
            double v0cosPA=-1, XicosPA=-1, lMassXi=-1, lMassOm=-1, lPt=-1, lV0Radius=-1, lXiRadius=-1;
            
            int nCand = fitterV0.process(pTrack, nTrack);
            if(nCand!=0){
                
                hCascCandidate->Fill(2.5);
                
                const auto& v0vtx = fitterV0.getPCACandidate();
                
                std::array<float, 3> pvec0;
                std::array<float, 3> pvec1;
                std::array<float, 21> cov0 = {0};
                std::array<float, 21> cov1 = {0};
                std::array<float, 21> covV0 = {0};
                
                //CosPA: manual calculation (transform into helper afterwards)
                v0cosPA = (v0vtx[0]-collision.posX())*(pvec0[0]+pvec1[0]) +
                (v0vtx[1]-collision.posY())*(pvec0[1]+pvec1[1]) +
                (v0vtx[2]-collision.posZ())*(pvec0[2]+pvec1[2]);
                
                double lNormR = TMath::Sqrt( TMath::Power( (v0vtx[0]-collision.posX()) , 2) +
                                            TMath::Power( (v0vtx[1]-collision.posY()) , 2) +
                                            TMath::Power( (v0vtx[2]-collision.posZ()) , 2) );
                double lNormP = TMath::Sqrt( TMath::Power( (pvec0[0]+pvec1[0]) , 2) +
                                            TMath::Power( (pvec0[1]+pvec1[1]) , 2) +
                                            TMath::Power( (pvec0[2]+pvec1[2]) , 2) );
                v0cosPA /= (lNormR*lNormP+1e-9);
                
                const int momInd[6] = {9,13,14,18,19,20}; // cov matrix elements for momentum component
                fitterV0.getTrack(0).getPxPyPzGlo(pvec0);
                fitterV0.getTrack(1).getPxPyPzGlo(pvec1);
                fitterV0.getTrack(0).getCovXYZPxPyPzGlo(cov0);
                fitterV0.getTrack(1).getCovXYZPxPyPzGlo(cov1);
                for (int i=0;i<6;i++) {
                    int j = momInd[i];
                    covV0[j] = cov0[j] + cov1[j];
                }
                auto covVtxV0 = fitterV0.calcPCACovMatrix();
                covV0[0] = covVtxV0(0,0);
                covV0[1] = covVtxV0(1,0);
                covV0[2] = covVtxV0(1,1);
                covV0[3] = covVtxV0(2,0);
                covV0[4] = covVtxV0(2,1);
                covV0[5] = covVtxV0(2,2);
                
                const std::array<float, 3> vertex = {(float)v0vtx[0], (float)v0vtx[1], (float)v0vtx[2]};
                const std::array<float, 3> momentum = {pvec0[0]+pvec1[0], pvec0[1]+pvec1[1],pvec0[2]+pvec1[2]};
                
                auto tV0 = o2::track::TrackParCov(vertex, momentum, covV0, 0);
                tV0.setQ2Pt(0); //No bending, please
                
                int nCand2 = fitterCasc.process(tV0, bTrack);
                if(nCand2!=0){
                    
                    hCascCandidate->Fill(3.5);
                    
                    const auto& vtxcasc = fitterCasc.getPCACandidate();
                    std::array<float, 3> pveccasc0;
                    std::array<float, 3> pveccasc1;
                    fitterCasc.getTrack(0).getPxPyPzGlo(pveccasc0);
                    fitterCasc.getTrack(1).getPxPyPzGlo(pveccasc1);
                    
                    //CosPA: manual calculation (transform into helper afterwards)
                    XicosPA = (vtxcasc[0]-collision.posX())*(pveccasc0[0]+pveccasc1[0]) +
                    (vtxcasc[1]-collision.posY())*(pveccasc0[1]+pveccasc1[1]) +
                    (vtxcasc[2]-collision.posZ())*(pveccasc0[2]+pveccasc1[2]);
                    double lNormRxi = TMath::Sqrt( TMath::Power( (vtxcasc[0]-collision.posX()) , 2) +
                                                  TMath::Power( (vtxcasc[1]-collision.posY()) , 2) +
                                                  TMath::Power( (vtxcasc[2]-collision.posZ()) , 2) );
                    double lNormPxi = TMath::Sqrt( TMath::Power( (momentum[0]+pveccasc1[0]) , 2) +
                                                  TMath::Power( (momentum[1]+pveccasc1[1]) , 2) +
                                                  TMath::Power( (momentum[2]+pveccasc1[2]) , 2) );
                    XicosPA /= (lNormRxi*lNormPxi+1e-9);
                    
                    // calculate invariant masses
                    auto track0p2 = momentum[0]*momentum[0]+momentum[1]*momentum[1]+momentum[2]*momentum[2];
                    auto track1p2 = pveccasc1[0]*pveccasc1[0]+pveccasc1[1]*pveccasc1[1]+pveccasc1[2]*pveccasc1[2];
                    
                    auto e0  = TMath::Sqrt(massPr*massPr + track0p2);
                    auto e1  = TMath::Sqrt(massPi*massPi + track1p2);
                    auto e2  = TMath::Sqrt(massKa*massKa + track1p2);
                    
                    lMassXi = TMath::Sqrt((e0+e1)*(e0+e1)-
                                               (pveccasc1[0]+momentum[0])*(pveccasc1[0]+momentum[0])-
                                               (pveccasc1[1]+momentum[1])*(pveccasc1[1]+momentum[1])-
                                               (pveccasc1[2]+momentum[2])*(pveccasc1[2]+momentum[2]));
                    lMassOm = TMath::Sqrt((e0+e2)*(e0+e2)-
                                               (pveccasc1[0]+momentum[0])*(pveccasc1[0]+momentum[0])-
                                               (pveccasc1[1]+momentum[1])*(pveccasc1[1]+momentum[1])-
                                               (pveccasc1[2]+momentum[2])*(pveccasc1[2]+momentum[2]));
                    
                    lPt = TMath::Sqrt( (pveccasc1[0]+momentum[0])*(pveccasc1[0]+momentum[0])+
                                           (pveccasc1[1]+momentum[1])*(pveccasc1[1]+momentum[1]) );
                    
                    lV0Radius = TMath::Sqrt(v0vtx[0]*v0vtx[0] + v0vtx[1]*v0vtx[1] );
                    lXiRadius = TMath::Sqrt(vtxcasc[0]*vtxcasc[0] + vtxcasc[1]*vtxcasc[1] );
                    
                    if( casc.bachelor().charge()<0 ) hMassXiMinus -> Fill(lMassXi);
                    if( casc.bachelor().charge()>0 ) hMassXiPlus -> Fill(lMassXi);
                    if( casc.bachelor().charge()<0 ) hMassOmegaMinus -> Fill(lMassOm);
                    if( casc.bachelor().charge()>0 ) hMassOmegaPlus -> Fill(lMassOm);
                }
            }
            
            cascdata(dcaXYneg, dcaXYpos, dcaXYbach,
                     lV0Radius, lXiRadius, fitterV0.getChi2AtPCACandidate(),
                     fitterCasc.getChi2AtPCACandidate(), 0.1,v0cosPA,
                     XicosPA, 1.116, lMassXi,
                     lMassOm, casc.bachelor().charge(), lPt);
        }
        
    }
};

//aod::Cascades const& Cascades

struct cascadeconsumer {
    OutputObj<TH2D> h2dMassXiMinus   {TH2D("h2dMassXiMinus"   , "", 200,0,10, 200, 1.321-0.100,1.321+0.100)};
    OutputObj<TH2D> h2dMassXiPlus    {TH2D("h2dMassXiPlus"    , "", 200,0,10, 200, 1.321-0.100,1.321+0.100)};
    OutputObj<TH2D> h2dMassOmegaMinus{TH2D("h2dMassOmegaMinus", "", 200,0,10, 200, 1.672-0.100,1.672+0.100)};
    OutputObj<TH2D> h2dMassOmegaPlus {TH2D("h2dMassOmegaPlus" , "", 200,0,10, 200, 1.672-0.100,1.672+0.100)};
    
    void process(aod::Collision const& collision, soa::Join<aod::Cascades, aod::CascData> const& fullCascades)
    {
        for (auto& cascade : fullCascades) {
            if(cascade.Charges() < 0){
                h2dMassXiMinus->Fill(cascade.Pts(),cascade.MassAsXis());
                h2dMassOmegaMinus->Fill(cascade.Pts(),cascade.MassAsOmegas());
            }
            if(cascade.Charges() > 0){
                h2dMassXiPlus->Fill(cascade.Pts(),cascade.MassAsXis());
                h2dMassOmegaPlus->Fill(cascade.Pts(),cascade.MassAsOmegas());
            }
        }
    }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
    return WorkflowSpec{
        adaptAnalysisTask<TaskIndexer>("lf-TaskIndexer"),
        adaptAnalysisTask<cascadebuilder>("lf-cascadebuilder"),
        adaptAnalysisTask<cascadeconsumer>("lf-cascadeconsumer")
    };
}
