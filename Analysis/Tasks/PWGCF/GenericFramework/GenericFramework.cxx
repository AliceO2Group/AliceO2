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
/// \brief Implementation of the Generic Framework for flow measurements in O2
/// \author Emil Gorm Nielsen
/// \since 19-07-2021

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/RunningWorkflowInfo.h"

#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/Centrality.h"

#include <CCDB/BasicCCDBManager.h>
#include <chrono>

#include "Framework/HistogramRegistry.h"
#include "GenericFramework/GFW.h"
#include "GenericFramework/GFWCumulant.h"
#include "GenericFramework/FlowContainer.h"
#include "GenericFramework/GFWWeights.h"
#include <TProfile.h>
#include <TRandom3.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

struct GenericFramework {

  O2_DEFINE_CONFIGURABLE(cfgCutVertex, float, 10.0f, "Accepted z-vertex range")
  O2_DEFINE_CONFIGURABLE(cfgCutPtMin, float, 0.2f, "Minimal pT for tracks")
  O2_DEFINE_CONFIGURABLE(cfgCutPtMax, float, 3.0f, "Maximal pT for tracks")
  O2_DEFINE_CONFIGURABLE(cfgCutEta, float, 0.8f, "Eta range for tracks")

  O2_DEFINE_CONFIGURABLE(cfgEfficiency, std::string, "", "CCDB path to efficiency object")
  O2_DEFINE_CONFIGURABLE(cfgAcceptance, std::string, "", "CCDB path to acceptance object")

  ConfigurableAxis axisVertex{"axisVertex", {20, -10, 10}, "vertex axis for histograms"};
  ConfigurableAxis axisPhi{"axisPhi", {60, 0.0, 2.*M_PI}, "phi axis for histograms"};
  ConfigurableAxis axisEta{"axisEta", {40, -1., 1.}, "eta axis for histograms"};
  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.2, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 
                        0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90,
                        2.00, 2.20, 2.40, 2.60, 2.80, 3.00}, "pt axis for histograms"};
  ConfigurableAxis axisMultiplicity{"axisMultiplicity", {VARIABLE_WIDTH, 0, 5, 10, 20, 30, 40, 50, 60, 70, 
                        80, 90, 100.1}, "multiplicity / centrality axis for histograms"};

  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::pt > cfgCutPtMin) 
  && (aod::track::pt < cfgCutPtMax) && ((aod::track::isGlobalTrack == (uint8_t) true) 
  || (aod::track::isGlobalTrackSDD == (uint8_t) true));
  using myTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>>;

  //Connect to ccdb
  Service<ccdb::BasicCCDBManager> ccdb;
  Configurable<long> nolaterthan{"ccdb-no-later-than", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object"};
  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};

  struct Config {
    TH1D* mEfficiency = nullptr;
    GFWWeights* mAcceptance = nullptr;
  } cfg;

  //Define output
  OutputObj<FlowContainer> fFC{FlowContainer("FlowContainer")};
  HistogramRegistry registry{"registry"};


  // define global variables
  GFW *fGFW = new GFW();
  std::vector<GFW::CorrConfig> corrconfigs;
  TRandom3* fRndm = new TRandom3(0);   

  void init(InitContext const&)
  {

    ccdb->setURL(url.value);
    ccdb->setCaching(true);
    ccdb->setCreatedNotAfter(nolaterthan.value);

    //Global effiencies
    if (cfgEfficiency.value.empty() == false) {
      cfg.mEfficiency = ccdb->getForTimeStamp<TH1D>(cfgEfficiency.value, nolaterthan.value);
      if(cfg.mEfficiency) LOGF(info, "Loaded efficiency histogram %s (%p)", cfgEfficiency.value.c_str(), (void*)cfg.mEfficiency);
      else LOGF(info, "Loaded efficiency histogram %s (%p)", cfgEfficiency.value.c_str(), (void*)cfg.mEfficiency);
    }


    registry.add("hPhi","",{HistType::kTH1D,{axisPhi}});
    registry.add("hEta","",{HistType::kTH1D,{axisEta}});
    registry.add("hVtxZ","",{HistType::kTH1D,{axisVertex}});
    Int_t pows[7] = {3,0,2,2,3,3,3}; 
    Int_t powsFull[7] = {5,0,4,4,3,3,3};

    fGFW->AddRegion("refN",7,pows,-0.8,-0.4,1,1);
    fGFW->AddRegion("refP",7,pows,0.4,0.8,1,1);
    fGFW->AddRegion("full",7,powsFull,-0.8,0.8,1,2);
    corrconfigs.push_back(fGFW->GetCorrelatorConfig("refP {2} refN {-2}","ChGap22",kFALSE));
    corrconfigs.push_back(fGFW->GetCorrelatorConfig("refP {2 2} refN {-2 -2}","ChGap24",kFALSE));
    corrconfigs.push_back(fGFW->GetCorrelatorConfig("full {2 -2}","ChFull22",kFALSE));
    corrconfigs.push_back(fGFW->GetCorrelatorConfig("full {2 2 -2 -2}","ChFull24",kFALSE));
    corrconfigs.push_back(fGFW->GetCorrelatorConfig("refP {3} refN {-3}","ChGap32",kFALSE));
    corrconfigs.push_back(fGFW->GetCorrelatorConfig("refP {4} refN {-4}","ChGap42",kFALSE));

    TObjArray *oba = new TObjArray();
    oba->Add(new TNamed("ChGap22","ChGap22")); //for gap (|eta|>0.4) case
    oba->Add(new TNamed("ChGap24","ChGap24")); //for gap (|eta|>0.4) case
    oba->Add(new TNamed("ChFull22","ChFull22")); //no-gap case
    oba->Add(new TNamed("ChFull24","ChFull24")); //no-gap case
    oba->Add(new TNamed("ChGap32","ChGap32")); //gap-case
    oba->Add(new TNamed("ChGap42","ChGap42")); //gap case
    fFC->SetName("FlowContainer");
    fFC->Initialize(oba,axisMultiplicity,1);
    delete oba;

  }

  void FillFC(const GFW::CorrConfig &corconf, const double &cent, const double &rndm)
  {
    double dnx,val;
    dnx = fGFW->Calculate(corrconfigs.at(0),0,kTRUE).Re();
    if(dnx==0) return;
    if(!corconf.pTDif) {
      val = fGFW->Calculate(corrconfigs.at(0),0,kFALSE).Re()/dnx;
      if(TMath::Abs(val)<1) fFC->FillProfile(corrconfigs.at(0).Head.Data(),cent,val,1,rndm);
      return;
    }
    return;
  }

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents>>::iterator const& collision, aod::BCsWithTimestamps const&, myTracks const& tracks)
  {
    auto bc = collision.bc_as<aod::BCsWithTimestamps>();

    if (cfgAcceptance.value.empty() == false) {
      cfg.mAcceptance = ccdb->getForTimeStamp<GFWWeights>(cfgAcceptance.value, bc.timestamp());
      if(cfg.mAcceptance) LOGF(info, "Loaded acceptance histogram from %s (%p)", cfgAcceptance.value.c_str(), (void*)cfg.mAcceptance);
      else LOGF(warning,"Could not load acceptance histogram from %s (%p)", cfgAcceptance.value.c_str(), (void*)cfg.mAcceptance);
    }

    if(tracks.size()<1) return;
    double vtxz = collision.posZ();
    registry.fill(HIST("hVtxZ"),vtxz);
    fGFW->Clear();
    const auto centrality = collision.centV0M();
    double dnx, val;
    double l_Random = fRndm->Rndm();
    double weff, wacc;
    for (auto track = tracks.begin(); track != tracks.end(); ++track) {
      registry.fill(HIST("hPhi"),track.phi());
      registry.fill(HIST("hEta"),track.eta());
      if(cfg.mEfficiency) weff = cfg.mEfficiency->GetBinContent(cfg.mEfficiency->FindBin(track.pt()));
      else weff = 1.0;
      if(weff==0) continue;
      weff = 1./weff;
      if(cfg.mAcceptance) wacc = cfg.mAcceptance->GetNUA(track.phi(),track.eta(),vtxz);
      else wacc = 1;

      fGFW->Fill(track.eta(),1,track.phi(),wacc*weff,3);
    } 
    for(int l_ind=0; l_ind<corrconfigs.size(); l_ind++) {
      FillFC(corrconfigs.at(l_ind),centrality,l_Random);
    };
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<GenericFramework>(cfgc),
  };
}