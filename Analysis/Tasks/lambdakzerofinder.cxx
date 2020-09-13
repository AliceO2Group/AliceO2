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
// This task re-reconstructs the V0s and cascades

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Analysis/SecondaryVertexHF.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Analysis/RecoDecay.h"
#include "Analysis/trackUtilities.h"
#include "PID/PIDResponse.h"

#include <TFile.h>
#include <TLorentzVector.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;
using namespace ROOT::Math;

struct wdvtask {
    Produces<aod::V0Data> v0data;
    
    OutputObj<TH1F> hCandPerEvent  {TH1F("hCandPerEvent"   , "", 1000, 0, 1000)};
    
    OutputObj<TH1F> hV0PosCrossedRows   {TH1F("hV0PosCrossedRows"   , "", 160, 0, 160)};
    OutputObj<TH1F> hV0NegCrossedRows   {TH1F("hV0NegCrossedRows"   , "", 160, 0, 160)};
    OutputObj<TH1F> hV0CosPA            {TH1F("hV0CosPA"   , "", 2000, 0.9, 1)};
    OutputObj<TH1F> hV0Radius           {TH1F("hV0Radius"   , "", 2000, 0, 200)};
    OutputObj<TH1F> hV0DCADaughters     {TH1F("hV0DCADaughters"   , "", 200, 0, 2)};
    OutputObj<TH1F> hV0PosDCAxy         {TH1F("hV0PosDCAxy"   , "", 200, 0, 5)};
    OutputObj<TH1F> hV0NegDCAxy         {TH1F("hV0NegDCAxy"   , "", 200, 0, 5)};
    
    OutputObj<TH1F> hMassK0Short   {TH1F("hMassK0Short"   , "", 200, 0.450, 0.550)};
    OutputObj<TH1F> hMassLambda    {TH1F("hMassLambda"    , "", 200, 1.115-0.100,1.115+0.100)};
    OutputObj<TH1F> hMassAntiLambda{TH1F("hMassAntiLambda"    , "", 200, 1.115-0.100,1.115+0.100)};

    //Configurables
    Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
    Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};
    
    //Selection criteria
    Configurable<double> v0cospa{"v0cospa", 0.995, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
    Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
    Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
    Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
    Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};
    
    //using myTracks = soa::Filtered<aod::Tracks>;
    //using myTracks = soa::Filtered<aod::fullTracks>;
    
    Partition<aod::FullTracks> goodPosTracks = aod::track::signed1Pt > 0;
    Partition<aod::FullTracks> goodNegTracks = aod::track::signed1Pt < 0;
    
    /// Extracts dca in the XY plane
    /// \return dcaXY
    template <typename T, typename U>
    auto getdcaXY(const T& track, const U& coll)
    {
        //Calculate DCAs
         auto sinAlpha = sin(track.alpha());
         auto cosAlpha = cos(track.alpha());
         auto globalX = track.x() * cosAlpha - track.y() * sinAlpha;
         auto globalY = track.x() * sinAlpha + track.y() * cosAlpha;
         return sqrt(pow((globalX - coll[0]), 2) +
                                pow((globalY - coll[1]), 2));
    }
    
    void process(aod::Collision const& collision,
                 aod::FullTracks const& tracks)
    {
        double massPi = TDatabasePDG::Instance()->GetParticle(kPiPlus)->Mass();
        double massKa = TDatabasePDG::Instance()->GetParticle(kKPlus)->Mass();
        double massPr = TDatabasePDG::Instance()->GetParticle(kProton)->Mass();
        
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
        
        Long_t lNCand = 0;
        
        //V0 selections: crossed rows 70
        //radius > 0.2, DCA > 1.5, cosPA > 0.95, track DCA > 0.05
        
        auto& goodpostracks = goodPosTracks.getPartition();
        auto& goodnegtracks = goodNegTracks.getPartition();
        std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};
        
        for (auto& t0 : goodpostracks(tracks)) {
            for (auto& t1 : goodnegtracks(tracks)) {
                if( t0.tpcNClsCrossedRows() < 70 ) continue;
                if( t1.tpcNClsCrossedRows() < 70 ) continue;
                
                if(getdcaXY(t0, pVtx)<dcapostopv) continue;
                if(getdcaXY(t1, pVtx)<dcanegtopv) continue;
                
                auto Track1 = getTrackParCov(t0);
                auto Track2 = getTrackParCov(t1);
                
                //Try to progate to dca
                int nCand = fitter.process(Track1, Track2);
                if (nCand == 0)
                    continue;
                const auto& vtx = fitter.getPCACandidate();
                
                //Fiducial: min radius
                if( TMath::Sqrt(TMath::Power(vtx[0],2) + TMath::Power(vtx[1],2)) < v0radius ) continue;
                
                //DCA V0 daughters
                if( fitter.getChi2AtPCACandidate() > dcav0dau ) continue;
                
                std::array<float, 3> pvec0;
                std::array<float, 3> pvec1;
                fitter.getTrack(0).getPxPyPzGlo(pvec0);
                fitter.getTrack(1).getPxPyPzGlo(pvec1);
                
                
            }
        }
        hCandPerEvent->Fill( lNCand ) ;
    }
};


WorkflowSpec defineDataProcessing(ConfigContext const&)
{
    return WorkflowSpec{
        adaptAnalysisTask<taskQA>("vertexerlf-qa"),
        adaptAnalysisTask<wdvtask>("vertexerlf-wdvtask")};
}
