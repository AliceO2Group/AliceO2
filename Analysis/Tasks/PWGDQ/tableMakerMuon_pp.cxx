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
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/TriggerAliases.h"
#include "AnalysisDataModel/ReducedInfoTables.h"
#include "PWGDQCore/VarManager.h"
#include "PWGDQCore/HistogramManager.h"
#include "PWGDQCore/AnalysisCut.h"
#include "PWGDQCore/AnalysisCompositeCut.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include <iostream>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
//using namespace o2::framework::expressions;
using namespace o2::aod;

using MyEvents = soa::Join<aod::Collisions, aod::EvSels, aod::Timestamps>;
using MyMuons = soa::Join<aod::FwdTracks, aod::FwdTracksCov>;
//using MyMuons = aod::Muons;

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager
constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::BC | VarManager::ObjTypes::Collision;

struct TableMakerMuon_pp {

  Produces<ReducedEvents> event;
  Produces<ReducedEventsExtended> eventExtended;
  Produces<ReducedEventsVtxCov> eventVtxCov;
  Produces<ReducedTracks> trackBasic;
  Produces<ReducedMuons> muonBasic;
  Produces<ReducedMuonsExtra> muonExtended;
  Produces<ReducedMuonsCov> muonCov; // TODO: use with fwdtracks

  float* fValues;

  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;

  // TODO: Filters should be used to make lowest level selection. The additional more restrictive cuts should be defined via the AnalysisCuts
  // TODO: Multiple event selections can be applied and decisions stored in the reducedevent::tag
  AnalysisCompositeCut* fEventCut;
  // TODO: Multiple track selections can be applied and decisions stored in the reducedtrack::filteringFlags

  // TODO a few of the important muon variables in the central data model are dynamic columns so not usable in expressions (e.g. eta, phi)
  //        Update the data model to have them as expression columns
  //Partition<MyMuons> muonSelectedTracks = o2::aod::muon::pt >= 0.5f; // For pp collisions a 0.5 GeV/c pp cuts is defined
  Partition<MyMuons> muonSelectedTracks = o2::aod::fwdtrack::pt >= 0.5f; // For pp collisions a 0.5 GeV/c pp cuts is defined

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms("Event_BeforeCuts;Event_AfterCuts;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());       // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
    DefineCuts();
  }

  void DefineCuts()
  {
    fEventCut = new AnalysisCompositeCut(true);
    AnalysisCut* eventVarCut = new AnalysisCut();
    eventVarCut->AddCut(VarManager::kVtxZ, -10.0, 10.0);
    eventVarCut->AddCut(VarManager::kIsINT7, 0.5, 1.5); // require kINT7
    fEventCut->AddCut(eventVarCut);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvents::iterator const& collision, MyMuons const& muonTracks, aod::BCs const& bcs)
  {
    uint64_t tag = 0;
    uint32_t triggerAliases = 0;
    for (int i = 0; i < kNaliases; i++) {
      if (collision.alias()[i] > 0) {
        triggerAliases |= (uint32_t(1) << i);
      }
    }

    VarManager::ResetValues(0, VarManager::kNEventWiseVariables, fValues);
    VarManager::FillEvent<gkEventFillMap>(collision, fValues); // extract event information and place it in the fgValues array
    fHistMan->FillHistClass("Event_BeforeCuts", fValues);      // automatically fill all the histograms in the class Event

    if (!fEventCut->IsSelected(fValues)) {
      return;
    }

    fHistMan->FillHistClass("Event_AfterCuts", fValues);

    event(tag, collision.bc().runNumber(), collision.posX(), collision.posY(), collision.posZ(), collision.numContrib());
    eventExtended(collision.bc().globalBC(), collision.bc().triggerMask(), collision.timestamp(), triggerAliases, 0.0f);
    eventVtxCov(collision.covXX(), collision.covXY(), collision.covXZ(), collision.covYY(), collision.covYZ(), collision.covZZ(), collision.chi2());

    uint64_t trackFilteringTag = 0;

    // TODO: to be used with the fwdtrack tables
    muonBasic.reserve(muonSelectedTracks.size());
    muonExtended.reserve(muonSelectedTracks.size());
    muonCov.reserve(muonSelectedTracks.size());
    for (auto& muon : muonSelectedTracks) {
      muonBasic(event.lastIndex(), trackFilteringTag, muon.pt(), muon.eta(), muon.phi(), muon.sign());
      muonExtended(muon.nClusters(), muon.pDca(), muon.rAtAbsorberEnd(),
                   muon.chi2(), muon.chi2MatchMCHMID(), muon.chi2MatchMCHMFT(),
                   muon.matchScoreMCHMFT(), muon.matchMFTTrackID(), muon.matchMCHTrackID());
      muonCov(muon.x(), muon.y(), muon.z(), muon.phi(), muon.tgl(), muon.signed1Pt(),
              muon.cXX(), muon.cXY(), muon.cYY(), muon.cPhiX(), muon.cPhiY(), muon.cPhiPhi(),
              muon.cTglX(), muon.cTglY(), muon.cTglPhi(), muon.cTglTgl(), muon.c1PtX(), muon.c1PtY(),
              muon.c1PtPhi(), muon.c1PtTgl(), muon.c1Pt21Pt2());
    }
  }

  void DefineHistograms(TString histClasses)
  {
    const int kNRuns = 2;
    int runs[kNRuns] = {244918, 244919};
    TString runsStr;
    for (int i = 0; i < kNRuns; i++) {
      runsStr += Form("%d;", runs[i]);
    }
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
      }
    } // end loop over histogram classes
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableMakerMuon_pp>(cfgc, TaskName{"table-maker-muon-pp"})};
}
