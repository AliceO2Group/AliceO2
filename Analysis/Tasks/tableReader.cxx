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
#include "Analysis/AnalysisCut.h"
#include "Analysis/AnalysisCompositeCut.h"
#include <TH1F.h>
#include <TMath.h>
#include <THashList.h>
#include <TString.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

using namespace o2;
using namespace o2::framework;
//using namespace o2::framework::expressions;
using namespace o2::aod;

struct TableReader {

  OutputObj<HistogramManager> fHistMan{"output"};
  AnalysisCompositeCut* fEventCut;
  AnalysisCompositeCut* fTrackCut;

  void init(o2::framework::InitContext&)
  {
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));

    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms("Event_BeforeCuts;Event_AfterCuts;Track_BeforeCuts;Track_AfterCuts"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill

    DefineCuts();
  }

  void DefineCuts()
  {
    fEventCut = new AnalysisCompositeCut(true);

    AnalysisCut* varCut = new AnalysisCut();
    varCut->AddCut(VarManager::kVtxZ, -10.0, 10.0);

    TF1* cutLow = new TF1("cutLow", "pol1", 0., 0.1);
    cutLow->SetParameters(0.2635, 1.0);
    varCut->AddCut(VarManager::kVtxY, cutLow, 0.335, false, VarManager::kVtxX, 0.067, 0.070);

    varCut->AddCut(VarManager::kVtxY, 0.0, 0.335);
    fEventCut->AddCut(varCut);

    fTrackCut = new AnalysisCompositeCut(true); // true: use AND
    AnalysisCut* cut1 = new AnalysisCut();
    cut1->AddCut(VarManager::kPt, 2.0, 4.0);
    AnalysisCut* cut2 = new AnalysisCut();
    cut2->AddCut(VarManager::kPt, 0.5, 3.0);
    fTrackCut->AddCut(cut1);
    fTrackCut->AddCut(cut2);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
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

    fHistMan->FillHistClass("Event_BeforeCuts", VarManager::fgValues); // automatically fill all the histograms in the class Event
    if (!fEventCut->IsSelected(VarManager::fgValues))
      return;

    fHistMan->FillHistClass("Event_AfterCuts", VarManager::fgValues);

    for (auto& track : tracks) {
      std::vector<float> trackInfo = {track.pt(), track.eta(), track.phi(), (float)track.charge()};
      VarManager::FillTrack(trackInfo);
      fHistMan->FillHistClass("Track_BeforeCuts", VarManager::fgValues);
      if (!fTrackCut->IsSelected(VarManager::fgValues))
        continue;
      fHistMan->FillHistClass("Track_AfterCuts", VarManager::fgValues);
    }
  }

  void DefineHistograms(TString histClasses)
  {
    const int kNRuns = 2;
    int runs[kNRuns] = {244918, 244919};
    TString runsStr;
    for (int i = 0; i < kNRuns; i++)
      runsStr += Form("%d;", runs[i]);
    VarManager::SetRunNumbers(kNRuns, runs);

    TObjArray* arr = histClasses.Tokenize(";");
    for (Int_t iclass = 0; iclass < arr->GetEntries(); ++iclass) {
      TString classStr = arr->At(iclass)->GetName();

      if (classStr.Contains("Event")) {
        fHistMan->AddHistClass(classStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "VtxZ", "Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ); // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxZ_Run", "Vtx Z", true,
                               kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId, 60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, runsStr.Data());                                        // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxX_VtxY", "Vtx X vs Vtx Y", false, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY);                                             // TH2F histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxX_VtxY_VtxZ", "vtx x - y - z", false, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 60, -15.0, 15.0, VarManager::kVtxZ);     // TH3F histogram
        fHistMan->AddHistogram(classStr.Data(), "NContrib_vs_VtxZ_prof", "Vtx Z vs ncontrib", true, 30, -15.0, 15.0, VarManager::kVtxZ, 10, -1., 1., VarManager::kVtxNcontrib);                             // TProfile histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxZ_vs_VtxX_VtxY_prof", "Vtx Z vs (x,y)", true, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 10, -1., 1., VarManager::kVtxZ); // TProfile2D histogram
        fHistMan->AddHistogram(classStr.Data(), "Ncontrib_vs_VtxZ_VtxX_VtxY_prof", "n-contrib vs (x,y,z)", true,
                               100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 30, -15., 15., VarManager::kVtxZ,
                               "", "", "", VarManager::kVtxNcontrib); // TProfile3D

        double vtxXbinLims[10] = {0.055, 0.06, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.074, 0.08};
        double vtxYbinLims[7] = {0.31, 0.32, 0.325, 0.33, 0.335, 0.34, 0.35};
        double vtxZbinLims[13] = {-15.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0};
        double nContribbinLims[9] = {0.0, 100.0, 200.0, 400.0, 600.0, 1000.0, 1500.0, 2000.0, 4000.0};

        fHistMan->AddHistogram(classStr.Data(), "VtxX_VtxY_nonEqualBinning", "Vtx X vs Vtx Y", false, 9, vtxXbinLims, VarManager::kVtxX, 6, vtxYbinLims, VarManager::kVtxY); // THnF histogram with custom non-equal binning

        fHistMan->AddHistogram(classStr.Data(), "VtxZ_weights", "Vtx Z", false,
                               60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, 10, 0., 0., VarManager::kNothing,
                               "", "", "", VarManager::kNothing, VarManager::kVtxNcontrib); // TH1F histogram, filled with weights using the vtx n-contributors

        Int_t vars[4] = {VarManager::kVtxX, VarManager::kVtxY, VarManager::kVtxZ, VarManager::kVtxNcontrib};
        TArrayD binLimits[4];
        binLimits[0] = TArrayD(10, vtxXbinLims);
        binLimits[1] = TArrayD(7, vtxYbinLims);
        binLimits[2] = TArrayD(13, vtxZbinLims);
        binLimits[3] = TArrayD(9, nContribbinLims);
        fHistMan->AddHistogram(classStr.Data(), "vtxHisto", "n contrib vs (x,y,z)", 4, vars, binLimits);

        continue;
      } // end if(Event)

      if (classStr.Contains("Track")) {
        fHistMan->AddHistClass(classStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "Pt", "p_{T} distribution", false, 200, 0.0, 20.0, VarManager::kPt);                                               // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "Eta", "#eta distribution", false, 100, -1.0, 1.0, VarManager::kEta);                                              // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "Phi_Eta", "#phi vs #eta distribution", false, 40, -1.0, 1.0, VarManager::kEta, 200, -6.3, 6.3, VarManager::kPhi); // TH2F histogram
        //fHistMan.AddHistogram("Track", "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 100, 0.0, 20.0, VarManager::kPin,
        //                         200, 0.0, 200., VarManager::kTPCsignal);   // TH2F histogram
      }
    } // end loop over histogram classes
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableReader>("table-reader")};
}
