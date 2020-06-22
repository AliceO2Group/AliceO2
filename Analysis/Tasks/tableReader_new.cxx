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
#include "Analysis/ReducedInfoTables.h"
#include "Analysis/VarManager.h"
#include "Analysis/HistogramManager.h"
#include <TH1F.h>
#include <TMath.h>
#include <iostream>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;


struct TableReader {
  
  HistogramManager fHistMan;
  float fValues[VarManager::kNVars];
  
  void init(o2::framework::InitContext&)
  {
    VarManager::SetDefaultVarNames();
    fHistMan.SetUseDefaultVariableNames(kTRUE);
    fHistMan.SetDefaultVarNames(VarManager::fgVariableNames,VarManager::fgVariableUnits);
    
    DefineHistograms();    // define all histograms 
    VarManager::SetUseVars(fHistMan.GetUsedVars());   // provide the list of required variables so that VarManager knows what to fill
  }

  void process(soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>::iterator event, 
               soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel> tracks)
  {
    // Reset the fValues array
    // TODO: reseting will have to be done selectively, for example run-wise variables don't need to be reset every event, but just updated if the run changes
    for(Int_t i=0; i<VarManager::kNVars; ++i) fValues[i]=-9999.;
    
    VarManager::Fill(event, fValues);     // extract event information and place it in the fValues array
    fHistMan.FillHistClass("Event", fValues);    // automatically fill all the histograms in the class Event
    
    for (auto& track : tracks) {
      VarManager::Fill(track, fValues);
      fHistMan.FillHistClass("Track", fValues);
    }
  }
  
  void DefineHistograms()
  {
    fHistMan.AddHistClass("Event");
    fHistMan.AddHistogram("Event", "VtxZ", "Vtx Z", kFALSE, 60, -15.0, 15.0, VarManager::kVtxZ);      // TH1F histogram 
    fHistMan.AddHistogram("Event", "CentVZERO", "CentVZERO", kFALSE, 100, 0.0, 100.0, VarManager::kCentVZERO);   // TH1F histogram
    fHistMan.AddHistogram("Event", "CentVZERO_VtxZ_prof", "CentVZERO vs vtxZ", kTRUE, 60, -15.0, 15.0, VarManager::kVtxZ, 
                                     10, 0.0, 0.0, VarManager::kCentVZERO);   // TProfile with <CentVZERO> vs vtxZ
    
    fHistMan.AddHistClass("Track");
    fHistMan.AddHistogram("Track", "Pt", "p_{T} distribution", kFALSE, 200, 0.0, 20.0, VarManager::kPt);      // TH1F histogram
    fHistMan.AddHistogram("Track", "TPCdedx_pIN", "TPC dE/dx vs pIN", kFALSE, 100, 0.0, 20.0, VarManager::kPin, 
                                 200, 0.0, 200., VarManager::kTPCsignal);   // TH2F histogram
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableReader>("table-reader")};
}
