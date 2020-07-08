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
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/ReducedInfoTables.h"
#include "Analysis/VarManager.h"
#include "Analysis/HistogramManager.h"
#include <TH1F.h>
#include <TMath.h>
#include <THashList.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct TableReader {

  OutputObj<HistogramManager> fHistMan{"output"};

  void DefineHistograms()
  {
    const int kNRuns = 2;
    int runs[kNRuns] = {244918, 244919};
    TString runsStr;
    for (int i = 0; i < kNRuns; i++)
      runsStr += Form("%d;", runs[i]);
    VarManager::SetRunNumbers(kNRuns, runs);

    fHistMan->AddHistClass("Event");
    fHistMan->AddHistogram("Event", "VtxZ", "Vtx Z", kFALSE, 60, -15.0, 15.0, VarManager::kVtxZ); // TH1F histogram
    fHistMan->AddHistogram("Event", "VtxZ_Run", "Vtx Z", kTRUE,
                           kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId, 60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, runsStr.Data());                                 // TH1F histogram
    fHistMan->AddHistogram("Event", "VtxX_VtxY", "Vtx X vs Vtx Y", kFALSE, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY);                                             // TH2F histogram
    fHistMan->AddHistogram("Event", "VtxX_VtxY_VtxZ", "vtx x - y - z", kFALSE, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 60, -15.0, 15.0, VarManager::kVtxZ);     // TH3F histogram
    fHistMan->AddHistogram("Event", "NContrib_vs_VtxZ_prof", "Vtx Z vs ncontrib", kTRUE, 30, -15.0, 15.0, VarManager::kVtxZ, 10, -1., 1., VarManager::kVtxNcontrib);                             // TProfile histogram
    fHistMan->AddHistogram("Event", "VtxZ_vs_VtxX_VtxY_prof", "Vtx Z vs (x,y)", kTRUE, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 10, -1., 1., VarManager::kVtxZ); // TProfile2D histogram
    fHistMan->AddHistogram("Event", "Ncontrib_vs_VtxZ_VtxX_VtxY_prof", "n-contrib vs (x,y,z)", kTRUE,
                           100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 30, -15., 15., VarManager::kVtxZ,
                           "", "", "", VarManager::kVtxNcontrib); // TProfile3D

    double vtxXbinLims[10] = {0.055, 0.06, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.074, 0.08};
    double vtxYbinLims[7] = {0.31, 0.32, 0.325, 0.33, 0.335, 0.34, 0.35};
    double vtxZbinLims[13] = {-15.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0};
    double nContribbinLims[9] = {0.0, 100.0, 200.0, 400.0, 600.0, 1000.0, 1500.0, 2000.0, 4000.0};

    fHistMan->AddHistogram("Event", "VtxX_VtxY_nonEqualBinning", "Vtx X vs Vtx Y", kFALSE, 9, vtxXbinLims, VarManager::kVtxX, 6, vtxYbinLims, VarManager::kVtxY); // TH2F histogram with custom non-equal binning

    fHistMan->AddHistogram("Event", "VtxZ_weights", "Vtx Z", kFALSE,
                           60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, 10, 0., 0., VarManager::kNothing,
                           "", "", "", VarManager::kNothing, VarManager::kVtxNcontrib); // TH1F histogram, filled with weights using the vtx n-contributors

    Int_t vars[4] = {VarManager::kVtxX, VarManager::kVtxY, VarManager::kVtxZ, VarManager::kVtxNcontrib};
    TArrayD binLimits[4];
    binLimits[0] = TArrayD(10, vtxXbinLims);
    binLimits[1] = TArrayD(7, vtxYbinLims);
    binLimits[2] = TArrayD(13, vtxZbinLims);
    binLimits[3] = TArrayD(9, nContribbinLims);
    fHistMan->AddHistogram("Event", "vtxHisto", "n contrib vs (x,y,z)", 4, vars, binLimits);

    //fHistMan.AddHistogram("Event", "CentVZERO", "CentVZERO", kFALSE, 100, 0.0, 100.0, VarManager::kCentVZERO);   // TH1F histogram
    //fHistMan.AddHistogram("Event", "CentVZERO_VtxZ_prof", "CentVZERO vs vtxZ", kTRUE, 60, -15.0, 15.0, VarManager::kVtxZ,
    //                             10, 0.0, 0.0, VarManager::kCentVZERO);   // TProfile with <CentVZERO> vs vtxZ

    fHistMan->AddHistClass("Track");
    fHistMan->AddHistogram("Track", "Pt", "p_{T} distribution", kFALSE, 200, 0.0, 20.0, VarManager::kPt); // TH1F histogram
    //fHistMan.AddHistogram("Track", "TPCdedx_pIN", "TPC dE/dx vs pIN", kFALSE, 100, 0.0, 20.0, VarManager::kPin,
    //                         200, 0.0, 200., VarManager::kTPCsignal);   // TH2F histogram
  }

  void init(o2::framework::InitContext&)
  {
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));

    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms();                              // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
  }

  //void process(soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>::iterator event,
  //             soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel> tracks)
  void process(aod::ReducedEvents::iterator event, aod::ReducedTracks tracks)
  {
    // Reset the fgValues array
    // TODO: reseting will have to be done selectively, for example run-wise variables don't need to be reset every event, but just updated if the run changes
    //       The reset can be done selectively, using arguments in the ResetValues() function
    VarManager::ResetValues();

    std::vector<float> eventInfo = {(float)event.runNumber(), event.posX(), event.posY(), event.posZ(), (float)event.numContrib()};
    VarManager::FillEvent(eventInfo);                       // extract event information and place it in the fgValues array
    fHistMan->FillHistClass("Event", VarManager::fgValues); // automatically fill all the histograms in the class Event

    for (auto& track : tracks) {
      std::vector<float> trackInfo = {track.pt(), track.eta(), track.phi(), (float)track.charge()};
      VarManager::FillTrack(trackInfo);
      fHistMan->FillHistClass("Track", VarManager::fgValues);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableReader>("table-reader")};
}
