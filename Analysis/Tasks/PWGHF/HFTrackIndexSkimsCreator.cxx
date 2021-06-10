// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFTrackIndexSkimsCreator.cxx
/// \brief Pre-selection of 2-prong and 3-prong secondary vertices of heavy-flavour decay candidates
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisCore/HFConfigurables.h"
#include "AnalysisDataModel/EventSelection.h"
//#include "AnalysisDataModel/Centrality.h"
#include "AnalysisDataModel/StrangenessTables.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "ReconstructionDataFormats/V0.h"
#include "AnalysisTasksUtils/UtilsDebugLcK0Sp.h"

#include <algorithm>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod;
using namespace o2::analysis::hf_cuts_single_track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"do-LcK0Sp", VariantType::Bool, false, {"Skim also Lc --> K0S+p"}};
  ConfigParamSpec optionEvSel{"doEvSel", VariantType::Bool, false, {"Apply event selection"}};
  workflowOptions.push_back(optionDoMC);
  workflowOptions.push_back(optionEvSel);
}

#include "Framework/runDataProcessing.h"

//#define MY_DEBUG

#ifdef MY_DEBUG
using MY_TYPE1 = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::McTrackLabels>;
using MyTracks = soa::Join<aod::FullTracks, aod::HFSelTrack, aod::TracksExtended, aod::McTrackLabels>;
#define MY_DEBUG_MSG(condition, cmd) \
  if (condition) {                   \
    cmd;                             \
  }
#else
using MY_TYPE1 = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra>;
using MyTracks = soa::Join<aod::FullTracks, aod::HFSelTrack, aod::TracksExtended>;
#define MY_DEBUG_MSG(condition, cmd)
#endif

/// Event selection
struct HfTagSelCollisions {

  Produces<aod::HFSelCollision> rowSelectedCollision;

  Configurable<bool> fillHistograms{"fillHistograms", true, "fill histograms"};
  Configurable<std::string> triggerClassName{"triggerClassName", "kINT7", "trigger class"};
  int triggerClass = std::distance(aliasLabels, std::find(aliasLabels, aliasLabels + kNaliases, triggerClassName.value.data()));

  HistogramRegistry registry{
    "registry",
    {{"hEvents", "Events;;entries", {HistType::kTH1F, {{3, 0.5, 3.5}}}}}};

  void init(InitContext const&)
  {
    std::string labels[3] = {"processed collisions", "selected collisions", "rej. trigger class"};
    for (int iBin = 0; iBin < 3; iBin++) {
      registry.get<TH1>(HIST("hEvents"))->GetXaxis()->SetBinLabel(iBin + 1, labels[iBin].data());
    }
  }

  // event selection
  void processEvSel(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision)
  {
    int statusCollision = 0;

    if (fillHistograms) {
      registry.get<TH1>(HIST("hEvents"))->Fill(1);
    }

    if (!collision.alias()[triggerClass]) {
      statusCollision |= BIT(0);
      if (fillHistograms) {
        registry.get<TH1>(HIST("hEvents"))->Fill(3);
      }
    }

    //TODO: add more event selection criteria

    // selected events
    if (fillHistograms && statusCollision == 0) {
      registry.get<TH1>(HIST("hEvents"))->Fill(2);
    }

    // fill table row
    rowSelectedCollision(statusCollision);
  };

  // no event selection in case of no event-selection task attached
  void processNoEvSel(aod::Collision const&)
  {
    int statusCollision = 0;

    if (fillHistograms) {
      registry.get<TH1>(HIST("hEvents"))->Fill(1);
      registry.get<TH1>(HIST("hEvents"))->Fill(2);
    }

    // fill table row
    rowSelectedCollision(statusCollision);
  };
};

/// Track selection
struct HfTagSelTracks {

  // enum for candidate type
  enum CandidateType {
    Cand2Prong = 0,
    Cand3Prong
  };

  Produces<aod::HFSelTrack> rowSelectedTrack;

  Configurable<bool> fillHistograms{"fillHistograms", true, "fill histograms"};
  Configurable<double> d_bz{"d_bz", 5., "bz field"};
  // quality cut
  Configurable<bool> doCutQuality{"doCutQuality", true, "apply quality cuts"};
  Configurable<int> d_tpcnclsfound{"d_tpcnclsfound", 70, ">= min. number of TPC clusters needed"};
  // pT bins for single-track cuts
  Configurable<std::vector<double>> pTBinsTrack{"ptbins_singletrack", std::vector<double>{hf_cuts_single_track::pTBinsTrack_v}, "track pT bin limits for 2-prong DCAXY pT-depentend cut"};
  // 2-prong cuts
  Configurable<double> ptmintrack_2prong{"ptmintrack_2prong", -1., "min. track pT for 2 prong candidate"};
  Configurable<LabeledArray<double>> cutsTrack2Prong{"cuts_singletrack_2prong", {hf_cuts_single_track::cutsTrack[0], npTBinsTrack, nCutVarsTrack, pTBinLabelsTrack, cutVarLabelsTrack}, "Single-track selections per pT bin for 2-prong candidates"};
  Configurable<double> etamax_2prong{"etamax_2prong", 4., "max. pseudorapidity for 2 prong candidate"};
  // 3-prong cuts
  Configurable<double> ptmintrack_3prong{"ptmintrack_3prong", -1., "min. track pT for 3 prong candidate"};
  Configurable<LabeledArray<double>> cutsTrack3Prong{"cuts_singletrack_3prong", {hf_cuts_single_track::cutsTrack[0], npTBinsTrack, nCutVarsTrack, pTBinLabelsTrack, cutVarLabelsTrack}, "Single-track selections per pT bin for 3-prong candidates"};
  Configurable<double> etamax_3prong{"etamax_3prong", 4., "max. pseudorapidity for 3 prong candidate"};
  // bachelor cuts (when using cascades)
  Configurable<double> ptMinTrackBach{"ptMinTrackBach", 0.3, "min. track pT for bachelor in cascade candidate"}; // 0.5 for PbPb 2015?
  Configurable<double> dcaToPrimXYMaxPtBach{"dcaToPrimXYMaxPtBach", 2., "max pt cut for min. DCAXY to prim. vtx. for bachelor in cascade candidate"};
  Configurable<double> dcaToPrimXYMinBach{"dcaToPrimXYMinBach", 0., "min. DCAXY to prim. vtx. for bachelor in cascade candidate"}; // for PbPb 2018, the cut should be 0.0025
  Configurable<double> dcaToPrimXYMaxBach{"dcaToPrimXYMaxBach", 1.0, "max. DCAXY to prim. vtx. for bachelor in cascade candidate"};
  Configurable<double> etaMaxBach{"etaMaxBach", 0.8, "max. pseudorapidity for bachelor in cascade candidate"};

  // for debugging
#ifdef MY_DEBUG
  Configurable<std::vector<int>> indexK0Spos{"indexK0Spos", {729, 2866, 4754, 5457, 6891, 7824, 9243, 9810}, "indices of K0S positive daughters, for debug"};
  Configurable<std::vector<int>> indexK0Sneg{"indexK0Sneg", {730, 2867, 4755, 5458, 6892, 7825, 9244, 9811}, "indices of K0S negative daughters, for debug"};
  Configurable<std::vector<int>> indexProton{"indexProton", {717, 2810, 4393, 5442, 6769, 7793, 9002, 9789}, "indices of protons, for debug"};
#endif

  HistogramRegistry registry{
    "registry",
    {{"hpt_nocuts", "all tracks;#it{p}_{T}^{track} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     // 2-prong histograms
     {"hpt_cuts_2prong", "tracks selected for 2-prong vertexing;#it{p}_{T}^{track} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hdcatoprimxy_cuts_2prong", "tracks selected for 2-prong vertexing;DCAxy to prim. vtx. (cm);entries", {HistType::kTH1F, {{400, -2., 2.}}}},
     {"heta_cuts_2prong", "tracks selected for 2-prong vertexing;#it{#eta};entries", {HistType::kTH1F, {{static_cast<int>(1.2 * etamax_2prong * 100), -1.2 * etamax_2prong, 1.2 * etamax_2prong}}}},
     // 3-prong histograms
     {"hpt_cuts_3prong", "tracks selected for 3-prong vertexing;#it{p}_{T}^{track} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hdcatoprimxy_cuts_3prong", "tracks selected for 3-prong vertexing;DCAxy to prim. vtx. (cm);entries", {HistType::kTH1F, {{400, -2., 2.}}}},
     {"heta_cuts_3prong", "tracks selected for 3-prong vertexing;#it{#eta};entries", {HistType::kTH1F, {{static_cast<int>(1.2 * etamax_3prong * 100), -1.2 * etamax_3prong, 1.2 * etamax_3prong}}}},
     // bachelor (for cascades) histograms
     {"hpt_cuts_bach", "bachelor tracks selected for cascade vertexing;#it{p}_{T}^{track} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hdcatoprimxy_cuts_bach", "bachelor tracks selected for cascade vertexing;DCAxy to prim. vtx. (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"heta_cuts_bach", "bachelortracks selected for cascade vertexing;#it{#eta};entries", {HistType::kTH1F, {{100, -1., 1.}}}}

    }};

  // array of 2-prong and 3-prong single-track cuts
  std::array<LabeledArray<double>, 2> cutsSingleTrack;

  void init(InitContext const&)
  {
    cutsSingleTrack = {cutsTrack2Prong, cutsTrack3Prong};
  }

  /// Single-track cuts for 2-prongs or 3-prongs
  /// \param hfTrack is a track
  /// \param dca is a 2-element array with dca in transverse and longitudinal directions
  /// \return true if track passes all cuts
  template <typename T>
  bool isSelectedTrack(const T& hfTrack, const array<float, 2>& dca, const int candType)
  {
    auto pTBinTrack = findBin(pTBinsTrack, hfTrack.pt());
    if (pTBinTrack == -1) {
      return false;
    }

    if (std::abs(dca[0]) < cutsSingleTrack[candType].get(pTBinTrack, "min_dcaxytoprimary")) {
      return false; //minimum DCAxy
    }
    if (std::abs(dca[0]) > cutsSingleTrack[candType].get(pTBinTrack, "max_dcaxytoprimary")) {
      return false; //maximum DCAxy
    }
    return true;
  }

  void process(aod::Collision const& collision,
               MY_TYPE1 const& tracks
#ifdef MY_DEBUG
               ,
               aod::McParticles& mcParticles
#endif
  )
  {
    math_utils::Point3D<float> vtxXYZ(collision.posX(), collision.posY(), collision.posZ());
    for (auto& track : tracks) {

#ifdef MY_DEBUG
      auto indexBach = track.mcParticleId();
      //      LOG(INFO) << "Checking label " << indexBach;
      bool isProtonFromLc = isProtonFromLcFunc(indexBach, indexProton);

#endif

      MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "\nWe found the proton " << indexBach);

      int status_prong = 7; // selection flag , 3 bits on

      auto trackPt = track.pt();
      if (fillHistograms.value) {
        registry.get<TH1>(HIST("hpt_nocuts"))->Fill(trackPt);
      }

      // pT cut
      if (trackPt < ptmintrack_2prong) {
        status_prong = status_prong & ~(1 << 0); // the bitwise operation & ~(1 << n) will set the nth bit to 0
      }
      if (trackPt < ptmintrack_3prong) {
        status_prong = status_prong & ~(1 << 1);
      }
      MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "proton " << indexBach << " pt = " << trackPt << " (cut " << ptMinTrackBach << ")");

      if (trackPt < ptMinTrackBach) {
        status_prong = status_prong & ~(1 << 2);
      }

      auto trackEta = track.eta();
      // eta cut
      if ((status_prong & (1 << 0)) && std::abs(trackEta) > etamax_2prong) {
        status_prong = status_prong & ~(1 << 0);
      }
      if ((status_prong & (1 << 1)) && std::abs(trackEta) > etamax_3prong) {
        status_prong = status_prong & ~(1 << 1);
      }
      MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "proton " << indexBach << " eta = " << trackEta << " (cut " << etaMaxBach << ")");

      if ((status_prong & (1 << 2)) && std::abs(trackEta) > etaMaxBach) {
        status_prong = status_prong & ~(1 << 2);
      }

      // quality cut
      MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "proton " << indexBach << " tpcNClsFound = " << track.tpcNClsFound() << " (cut " << tpcnclsfound.value << ")");

      if (doCutQuality.value && status_prong > 0) { // FIXME to make a more complete selection e.g track.flags() & o2::aod::track::TPCrefit && track.flags() & o2::aod::track::GoldenChi2 &&
        UChar_t clustermap = track.itsClusterMap();
        if (!(track.tpcNClsFound() >= d_tpcnclsfound.value &&
              track.flags() & o2::aod::track::ITSrefit &&
              (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1)))) {
          status_prong = 0;
          MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "proton " << indexBach << " did not pass clusters cut");
        }
      }

      // DCA cut
      array<float, 2> dca;
      if (status_prong > 0) {
        double dcaToPrimXYMinBach_ptDep = dcaToPrimXYMinBach * TMath::Max(0., (1 - TMath::Floor(trackPt / dcaToPrimXYMaxPtBach)));
        auto trackparvar0 = getTrackParCov(track);
        if (!trackparvar0.propagateParamToDCA(vtxXYZ, d_bz, &dca, 100.)) { // get impact parameters
          status_prong = 0;
        }
        if ((status_prong & (1 << 0)) && !isSelectedTrack(track, dca, Cand2Prong)) {
          status_prong = status_prong & ~(1 << 0);
        }
        if ((status_prong & (1 << 1)) && !isSelectedTrack(track, dca, Cand3Prong)) {
          status_prong = status_prong & ~(1 << 1);
        }
        if ((status_prong & (1 << 2)) && (std::abs(dca[0]) < dcaToPrimXYMinBach_ptDep || std::abs(dca[0]) > dcaToPrimXYMaxBach)) {
          MY_DEBUG_MSG(isProtonFromLc,
                       LOG(INFO) << "proton " << indexBach << " did not pass DCA cut";
                       LOG(INFO) << "dca[0] = " << dca[0] << " (lower cut " << dcaToPrimXYMinBach_ptDep << ", upper cut " << dcaToPrimXYMaxBach << ")";);
          status_prong = status_prong & ~(1 << 2);
        }
      }
      MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "status_prong = " << status_prong; printf("\n"));

      // fill histograms
      if (fillHistograms) {
        if (status_prong & (1 << 0)) {
          registry.get<TH1>(HIST("hpt_cuts_2prong"))->Fill(trackPt);
          registry.get<TH1>(HIST("hdcatoprimxy_cuts_2prong"))->Fill(dca[0]);
          registry.get<TH1>(HIST("heta_cuts_2prong"))->Fill(trackEta);
        }
        if (status_prong & (1 << 1)) {
          registry.get<TH1>(HIST("hpt_cuts_3prong"))->Fill(trackPt);
          registry.get<TH1>(HIST("hdcatoprimxy_cuts_3prong"))->Fill(dca[0]);
          registry.get<TH1>(HIST("heta_cuts_3prong"))->Fill(trackEta);
        }
        if (status_prong & (1 << 2)) {
          MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "Will be kept: Proton from Lc " << indexBach);
          registry.get<TH1>(HIST("hpt_cuts_bach"))->Fill(trackPt);
          registry.get<TH1>(HIST("hdcatoprimxy_cuts_bach"))->Fill(dca[0]);
          registry.get<TH1>(HIST("heta_cuts_bach"))->Fill(trackEta);
        }
      }

      // fill table row
      rowSelectedTrack(status_prong, dca[0], dca[1]);
    }
  }
};

//____________________________________________________________________________________________________________________________________________

/// Pre-selection of 2-prong and 3-prong secondary vertices
struct HfTrackIndexSkimsCreator {
  Produces<aod::HfTrackIndexProng2> rowTrackIndexProng2;
  Produces<aod::HfCutStatusProng2> rowProng2CutStatus;
  Produces<aod::HfTrackIndexProng3> rowTrackIndexProng3;
  Produces<aod::HfCutStatusProng3> rowProng3CutStatus;

  //Configurable<int> nCollsMax{"nCollsMax", -1, "Max collisions per file"}; //can be added to run over limited collisions per file - for tesing purposes
  Configurable<bool> fillHistograms{"fillHistograms", true, "fill histograms"};
  Configurable<int> do3prong{"do3prong", 0, "do 3 prong"};
  // vertexing parameters
  Configurable<double> d_bz{"d_bz", 5., "magnetic field kG"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<bool> useAbsDCA{"useAbsDCA", true, "Minimise abs. distance rather than chi2"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations if chi2/chi2old > this"};
  Configurable<HFTrackIndexSkimsCreatorConfigs> configs{"configs", {}, "configurables"};
  Configurable<bool> b_debug{"b_debug", false, "debug mode"};

  HistogramRegistry registry{
    "registry",
    {{"hNTracks", ";# of tracks;entries", {HistType::kTH1F, {{2500, 0., 25000.}}}},
     // 2-prong histograms
     {"hvtx2_x", "2-prong candidates;#it{x}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -2., 2.}}}},
     {"hvtx2_y", "2-prong candidates;#it{y}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -2., 2.}}}},
     {"hvtx2_z", "2-prong candidates;#it{z}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -20., 20.}}}},
     {"hNCand2Prong", "2-prong candidates preselected;# of candidates;entries", {HistType::kTH1F, {{2000, 0., 200000.}}}},
     {"hNCand2ProngVsNTracks", "2-prong candidates preselected;# of selected tracks;# of candidates;entries", {HistType::kTH2F, {{2500, 0., 25000.}, {2000, 0., 200000.}}}},
     {"hmassD0ToPiK", "D^{0} candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmassJpsiToEE", "J/#psi candidates;inv. mass (e^{#plus} e^{#minus}) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     // 3-prong histograms
     {"hvtx3_x", "3-prong candidates;#it{x}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -2., 2.}}}},
     {"hvtx3_y", "3-prong candidates;#it{y}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -2., 2.}}}},
     {"hvtx3_z", "3-prong candidates;#it{z}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -20., 20.}}}},
     {"hNCand3Prong", "3-prong candidates preselected;# of candidates;entries", {HistType::kTH1F, {{5000, 0., 500000.}}}},
     {"hNCand3ProngVsNTracks", "3-prong candidates preselected;# of selected tracks;# of candidates;entries", {HistType::kTH2F, {{2500, 0., 25000.}, {5000, 0., 500000.}}}},
     {"hmassDPlusToPiKPi", "D^{#plus} candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmassLcToPKPi", "#Lambda_{c} candidates;inv. mass (p K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmassDsToPiKK", "D_{s} candidates;inv. mass (K K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmassXicToPKPi", "#Xi_{c} candidates;inv. mass (p K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}}}};

  Filter filterSelectCollisions = (aod::hf_selcollision::whyRejectColl == 0);
  Filter filterSelectTracks = aod::hf_seltrack::isSelProng > 0;

  using SelectedCollisions = soa::Filtered<soa::Join<aod::Collisions, aod::HFSelCollision>>;
  using SelectedTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::HFSelTrack>>;

  // FIXME
  //Partition<SelectedTracks> tracksPos = aod::track::signed1Pt > 0.f;
  //Partition<SelectedTracks> tracksNeg = aod::track::signed1Pt < 0.f;

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double massProton = RecoDecay::getMassPDG(kProton);
  double massElectron = RecoDecay::getMassPDG(kElectron);

  // int nColls{0}; //can be added to run over limited collisions per file - for tesing purposes

  void process( //soa::Join<aod::Collisions, aod::Cents>::iterator const& collision, //FIXME add centrality when option for variations to the process function appears
    SelectedCollisions::iterator const& collision,
    aod::BCs const& bcs,
    SelectedTracks const& tracks)
  {

    //can be added to run over limited collisions per file - for tesing purposes
    /*
    if (nCollsMax > -1){
      if (nColls == nCollMax){
        return;
        //can be added to run over limited collisions per file - for tesing purposes
      }
      nColls++;
    }
    */

    //auto centrality = collision.centV0M(); //FIXME add centrality when option for variations to the process function appears

    //FIXME move above process function
    const int n2ProngDecays = hf_cand_prong2::DecayType::N2ProngDecays; // number of 2-prong hadron types
    const int n3ProngDecays = hf_cand_prong3::DecayType::N3ProngDecays; // number of 3-prong hadron types
    int n2ProngBit = (1 << n2ProngDecays) - 1;                          // bit value for 2-prong candidates where each candidiate is one bit and they are all set to 1
    int n3ProngBit = (1 << n3ProngDecays) - 1;                          // bit value for 3-prong candidates where each candidiate is one bit and they are all set to 1

    //retrieve cuts from json - to be made pT dependent when option appears in json
    const int nCuts2Prong = 4; // how many different selections are made on 2-prongs
    double cut2ProngPtCandMin[n2ProngDecays];
    double cut2ProngInvMassCandMin[n2ProngDecays];
    double cut2ProngInvMassCandMax[n2ProngDecays];
    double cut2ProngCPACandMin[n2ProngDecays];
    double cut2ProngImpParProductCandMax[n2ProngDecays];

    cut2ProngPtCandMin[hf_cand_prong2::DecayType::D0ToPiK] = configs->mPtD0ToPiKMin;
    cut2ProngInvMassCandMin[hf_cand_prong2::DecayType::D0ToPiK] = configs->mInvMassD0ToPiKMin;
    cut2ProngInvMassCandMax[hf_cand_prong2::DecayType::D0ToPiK] = configs->mInvMassD0ToPiKMax;
    cut2ProngCPACandMin[hf_cand_prong2::DecayType::D0ToPiK] = configs->mCPAD0ToPiKMin;
    cut2ProngImpParProductCandMax[hf_cand_prong2::DecayType::D0ToPiK] = configs->mImpParProductD0ToPiKMax;

    cut2ProngPtCandMin[hf_cand_prong2::DecayType::JpsiToEE] = configs->mPtJpsiToEEMin;
    cut2ProngInvMassCandMin[hf_cand_prong2::DecayType::JpsiToEE] = configs->mInvMassJpsiToEEMin;
    cut2ProngInvMassCandMax[hf_cand_prong2::DecayType::JpsiToEE] = configs->mInvMassJpsiToEEMax;
    cut2ProngCPACandMin[hf_cand_prong2::DecayType::JpsiToEE] = configs->mCPAJpsiToEEMin;
    cut2ProngImpParProductCandMax[hf_cand_prong2::DecayType::JpsiToEE] = configs->mImpParProductJpsiToEEMax;

    const int nCuts3Prong = 4; // how many different selections are made on 3-prongs
    double cut3ProngPtCandMin[n3ProngDecays];
    double cut3ProngInvMassCandMin[n3ProngDecays];
    double cut3ProngInvMassCandMax[n3ProngDecays];
    double cut3ProngCPACandMin[n3ProngDecays];
    double cut3ProngDecLenCandMin[n3ProngDecays];

    cut3ProngPtCandMin[hf_cand_prong3::DecayType::DPlusToPiKPi] = configs->mPtDPlusToPiKPiMin;
    cut3ProngInvMassCandMin[hf_cand_prong3::DecayType::DPlusToPiKPi] = configs->mInvMassDPlusToPiKPiMin;
    cut3ProngInvMassCandMax[hf_cand_prong3::DecayType::DPlusToPiKPi] = configs->mInvMassDPlusToPiKPiMax;
    cut3ProngCPACandMin[hf_cand_prong3::DecayType::DPlusToPiKPi] = configs->mCPADPlusToPiKPiMin;
    cut3ProngDecLenCandMin[hf_cand_prong3::DecayType::DPlusToPiKPi] = configs->mDecLenDPlusToPiKPiMin;

    cut3ProngPtCandMin[hf_cand_prong3::DecayType::LcToPKPi] = configs->mPtLcToPKPiMin;
    cut3ProngInvMassCandMin[hf_cand_prong3::DecayType::LcToPKPi] = configs->mInvMassLcToPKPiMin;
    cut3ProngInvMassCandMax[hf_cand_prong3::DecayType::LcToPKPi] = configs->mInvMassLcToPKPiMax;
    cut3ProngCPACandMin[hf_cand_prong3::DecayType::LcToPKPi] = configs->mCPALcToPKPiMin;
    cut3ProngDecLenCandMin[hf_cand_prong3::DecayType::LcToPKPi] = configs->mDecLenLcToPKPiMin;

    cut3ProngPtCandMin[hf_cand_prong3::DecayType::DsToPiKK] = configs->mPtDsToPiKKMin;
    cut3ProngInvMassCandMin[hf_cand_prong3::DecayType::DsToPiKK] = configs->mInvMassDsToPiKKMin;
    cut3ProngInvMassCandMax[hf_cand_prong3::DecayType::DsToPiKK] = configs->mInvMassDsToPiKKMax;
    cut3ProngCPACandMin[hf_cand_prong3::DecayType::DsToPiKK] = configs->mCPADsToPiKKMin;
    cut3ProngDecLenCandMin[hf_cand_prong3::DecayType::DsToPiKK] = configs->mDecLenDsToPiKKMin;

    cut3ProngPtCandMin[hf_cand_prong3::DecayType::XicToPKPi] = configs->mPtXicToPKPiMin;
    cut3ProngInvMassCandMin[hf_cand_prong3::DecayType::XicToPKPi] = configs->mInvMassXicToPKPiMin;
    cut3ProngInvMassCandMax[hf_cand_prong3::DecayType::XicToPKPi] = configs->mInvMassXicToPKPiMax;
    cut3ProngCPACandMin[hf_cand_prong3::DecayType::XicToPKPi] = configs->mCPAXicToPKPiMin;
    cut3ProngDecLenCandMin[hf_cand_prong3::DecayType::XicToPKPi] = configs->mDecLenXicToPKPiMin;

    bool cutStatus2Prong[n2ProngDecays][nCuts2Prong];
    bool cutStatus3Prong[n3ProngDecays][nCuts3Prong];
    int nCutStatus2ProngBit = (1 << nCuts2Prong) - 1; // bit value for selection status for each 2-prong candidate where each selection is one bit and they are all set to 1
    int nCutStatus3ProngBit = (1 << nCuts3Prong) - 1; // bit value for selection status for each 3-prong candidate where each selection is one bit and they are all set to 1

    array<array<double, 2>, n2ProngDecays> arr2Mass1;
    arr2Mass1[hf_cand_prong2::DecayType::D0ToPiK] = array{massPi, massK};
    arr2Mass1[hf_cand_prong2::DecayType::JpsiToEE] = array{massElectron, massElectron};

    array<array<double, 2>, n2ProngDecays> arr2Mass2;
    arr2Mass2[hf_cand_prong2::DecayType::D0ToPiK] = array{massK, massPi};
    arr2Mass2[hf_cand_prong2::DecayType::JpsiToEE] = array{massElectron, massElectron};

    array<array<double, 3>, n3ProngDecays> arr3Mass1;
    arr3Mass1[hf_cand_prong3::DecayType::DPlusToPiKPi] = array{massPi, massK, massPi};
    arr3Mass1[hf_cand_prong3::DecayType::LcToPKPi] = array{massProton, massK, massPi};
    arr3Mass1[hf_cand_prong3::DecayType::DsToPiKK] = array{massK, massK, massPi};
    arr3Mass1[hf_cand_prong3::DecayType::XicToPKPi] = array{massProton, massK, massPi};

    array<array<double, 3>, n3ProngDecays> arr3Mass2;
    arr3Mass2[hf_cand_prong3::DecayType::DPlusToPiKPi] = array{massPi, massK, massPi};
    arr3Mass2[hf_cand_prong3::DecayType::LcToPKPi] = array{massPi, massK, massProton};
    arr3Mass2[hf_cand_prong3::DecayType::DsToPiKK] = array{massPi, massK, massK};
    arr3Mass2[hf_cand_prong3::DecayType::XicToPKPi] = array{massPi, massK, massProton};

    double mass2ProngHypo1[n2ProngDecays];
    double mass2ProngHypo2[n2ProngDecays];

    double mass3ProngHypo1[n3ProngDecays];
    double mass3ProngHypo2[n3ProngDecays];

    // 2-prong vertex fitter
    o2::vertexing::DCAFitterN<2> df2;
    df2.setBz(d_bz);
    df2.setPropagateToPCA(b_propdca);
    df2.setMaxR(d_maxr);
    df2.setMaxDZIni(d_maxdzini);
    df2.setMinParamChange(d_minparamchange);
    df2.setMinRelChi2Change(d_minrelchi2change);
    df2.setUseAbsDCA(useAbsDCA);

    // 3-prong vertex fitter
    o2::vertexing::DCAFitterN<3> df3;
    df3.setBz(d_bz);
    df3.setPropagateToPCA(b_propdca);
    df3.setMaxR(d_maxr);
    df3.setMaxDZIni(d_maxdzini);
    df3.setMinParamChange(d_minparamchange);
    df3.setMinRelChi2Change(d_minrelchi2change);
    df3.setUseAbsDCA(useAbsDCA);

    // used to calculate number of candidiates per event
    auto nCand2 = rowTrackIndexProng2.lastIndex();
    auto nCand3 = rowTrackIndexProng3.lastIndex();

    // first loop over positive tracks
    //for (auto trackPos1 = tracksPos.begin(); trackPos1 != tracksPos.end(); ++trackPos1) {
    for (auto trackPos1 = tracks.begin(); trackPos1 != tracks.end(); ++trackPos1) {
      if (trackPos1.signed1Pt() < 0) {
        continue;
      }
      bool sel2ProngStatusPos = trackPos1.isSelProng() & (1 << 0);
      bool sel3ProngStatusPos1 = trackPos1.isSelProng() & (1 << 1);
      if (!sel2ProngStatusPos && !sel3ProngStatusPos1) {
        continue;
      }

      auto trackParVarPos1 = getTrackParCov(trackPos1);

      // first loop over negative tracks
      //for (auto trackNeg1 = tracksNeg.begin(); trackNeg1 != tracksNeg.end(); ++trackNeg1) {
      for (auto trackNeg1 = tracks.begin(); trackNeg1 != tracks.end(); ++trackNeg1) {
        if (trackNeg1.signed1Pt() > 0) {
          continue;
        }
        bool sel2ProngStatusNeg = trackNeg1.isSelProng() & (1 << 0);
        bool sel3ProngStatusNeg1 = trackNeg1.isSelProng() & (1 << 1);
        if (!sel2ProngStatusNeg && !sel3ProngStatusNeg1) {
          continue;
        }

        auto trackParVarNeg1 = getTrackParCov(trackNeg1);

        int isSelected2ProngCand = n2ProngBit; //bitmap for checking status of two-prong candidates (1 is true, 0 is rejected)

        if (b_debug) {
          for (int n2 = 0; n2 < n2ProngDecays; n2++) {
            for (int n2cut = 0; n2cut < nCuts2Prong; n2cut++) {
              cutStatus2Prong[n2][n2cut] = true;
            }
          }
        }
        int iDebugCut = 0;

        // 2-prong vertex reconstruction
        if (sel2ProngStatusPos && sel2ProngStatusNeg) {

          // invariant-mass cut
          auto arrMom = array{
            array{trackPos1.px(), trackPos1.py(), trackPos1.pz()},
            array{trackNeg1.px(), trackNeg1.py(), trackNeg1.pz()}};
          for (int n2 = 0; n2 < n2ProngDecays; n2++) {
            mass2ProngHypo1[n2] = RecoDecay::M(arrMom, arr2Mass1[n2]);
            mass2ProngHypo2[n2] = RecoDecay::M(arrMom, arr2Mass2[n2]);
            if ((b_debug || (isSelected2ProngCand & 1 << n2)) && cut2ProngInvMassCandMin[n2] >= 0. && cut2ProngInvMassCandMax[n2] > 0.) { //no need to check isSelected2Prong but to avoid mistakes
              if ((mass2ProngHypo1[n2] < cut2ProngInvMassCandMin[n2] || mass2ProngHypo1[n2] >= cut2ProngInvMassCandMax[n2]) &&
                  (mass2ProngHypo2[n2] < cut2ProngInvMassCandMin[n2] || mass2ProngHypo2[n2] >= cut2ProngInvMassCandMax[n2])) {
                isSelected2ProngCand = isSelected2ProngCand & ~(1 << n2);
                if (b_debug) {
                  cutStatus2Prong[n2][iDebugCut] = false;
                }
              }
            }
          }
          iDebugCut++;

          // secondary vertex reconstruction and further 2-prong selections
          if (isSelected2ProngCand > 0 && df2.process(trackParVarPos1, trackParVarNeg1) > 0) { //should it be this or > 0 or are they equivalent
            // get secondary vertex
            const auto& secondaryVertex2 = df2.getPCACandidate();
            // get track momenta
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            df2.getTrack(0).getPxPyPzGlo(pvec0);
            df2.getTrack(1).getPxPyPzGlo(pvec1);

            auto pVecCandProng2 = RecoDecay::PVec(pvec0, pvec1);

            // candidate pT cut
            if ((b_debug || isSelected2ProngCand > 0) && (std::count_if(std::begin(cut2ProngPtCandMin), std::end(cut2ProngPtCandMin), [](double d) { return d >= 0.; }) > 0)) {
              double cand2ProngPt = RecoDecay::Pt(pVecCandProng2);
              for (int n2 = 0; n2 < n2ProngDecays; n2++) {
                if ((b_debug || (isSelected2ProngCand & 1 << n2)) && cand2ProngPt < cut2ProngPtCandMin[n2]) {
                  isSelected2ProngCand = isSelected2ProngCand & ~(1 << n2);
                  if (b_debug) {
                    cutStatus2Prong[n2][iDebugCut] = false;
                  }
                }
              }
            }
            iDebugCut++;

            // imp. par. product cut
            if ((b_debug || isSelected2ProngCand > 0) && (std::count_if(std::begin(cut2ProngImpParProductCandMax), std::end(cut2ProngImpParProductCandMax), [](double d) { return d < 100.; }) > 0)) {
              auto impParProduct = trackPos1.dcaPrim0() * trackNeg1.dcaPrim0();
              for (int n2 = 0; n2 < n2ProngDecays; n2++) {
                if ((b_debug || (isSelected2ProngCand & 1 << n2)) && impParProduct > cut2ProngImpParProductCandMax[n2]) {
                  isSelected2ProngCand = isSelected2ProngCand & ~(1 << n2);
                  if (b_debug) {
                    cutStatus2Prong[n2][iDebugCut] = false;
                  }
                }
              }
            }
            iDebugCut++;

            // CPA cut
            if ((b_debug || isSelected2ProngCand > 0) && (std::count_if(std::begin(cut2ProngCPACandMin), std::end(cut2ProngCPACandMin), [](double d) { return d > -2.; }) > 0)) {
              auto cpa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex2, pVecCandProng2);
              for (int n2 = 0; n2 < n2ProngDecays; n2++) {
                if ((b_debug || (isSelected2ProngCand & 1 << n2)) && cpa < cut2ProngCPACandMin[n2]) {
                  isSelected2ProngCand = isSelected2ProngCand & ~(1 << n2);
                  if (b_debug) {
                    cutStatus2Prong[n2][iDebugCut] = false;
                  }
                }
              }
            }
            iDebugCut++;

            if (isSelected2ProngCand > 0) {
              // fill table row
              rowTrackIndexProng2(trackPos1.globalIndex(),
                                  trackNeg1.globalIndex(), isSelected2ProngCand);
              if (b_debug) {
                int Prong2CutStatus[n2ProngDecays];
                for (int n2 = 0; n2 < n2ProngDecays; n2++) {
                  Prong2CutStatus[n2] = nCutStatus2ProngBit;
                  for (int n2cut = 0; n2cut < nCuts2Prong; n2cut++) {
                    if (!cutStatus2Prong[n2][n2cut]) {
                      Prong2CutStatus[n2] = Prong2CutStatus[n2] & ~(1 << n2cut);
                    }
                  }
                }
                rowProng2CutStatus(Prong2CutStatus[0], Prong2CutStatus[1]); //FIXME when we can do this by looping over n2ProngDecays
              }

              // fill histograms
              if (fillHistograms) {

                registry.get<TH1>(HIST("hvtx2_x"))->Fill(secondaryVertex2[0]);
                registry.get<TH1>(HIST("hvtx2_y"))->Fill(secondaryVertex2[1]);
                registry.get<TH1>(HIST("hvtx2_z"))->Fill(secondaryVertex2[2]);
                arrMom = array{pvec0, pvec1};
                for (int n2 = 0; n2 < n2ProngDecays; n2++) {
                  if (isSelected2ProngCand & 1 << n2) {
                    if ((cut2ProngInvMassCandMin[n2] < 0. && cut2ProngInvMassCandMax[n2] <= 0.) || (mass2ProngHypo1[n2] >= cut2ProngInvMassCandMin[n2] && mass2ProngHypo1[n2] < cut2ProngInvMassCandMax[n2])) {
                      mass2ProngHypo1[n2] = RecoDecay::M(arrMom, arr2Mass1[n2]);
                      if (n2 == hf_cand_prong2::DecayType::D0ToPiK) {
                        registry.get<TH1>(HIST("hmassD0ToPiK"))->Fill(mass2ProngHypo1[n2]);
                      }
                      if (n2 == hf_cand_prong2::DecayType::JpsiToEE) {
                        registry.get<TH1>(HIST("hmassJpsiToEE"))->Fill(mass2ProngHypo1[n2]);
                      }
                    }
                    if ((cut2ProngInvMassCandMin[n2] < 0. && cut2ProngInvMassCandMax[n2] <= 0.) || (mass2ProngHypo2[n2] >= cut2ProngInvMassCandMin[n2] && mass2ProngHypo2[n2] < cut2ProngInvMassCandMax[n2])) {
                      mass2ProngHypo2[n2] = RecoDecay::M(arrMom, arr2Mass2[n2]);
                      if (n2 == hf_cand_prong2::DecayType::D0ToPiK) {
                        registry.get<TH1>(HIST("hmassD0ToPiK"))->Fill(mass2ProngHypo1[n2]);
                      }
                    }
                  }
                }
              }
            }
          }
        }

        // 3-prong vertex reconstruction
        if (do3prong == 1) {
          if (!sel3ProngStatusPos1 || !sel3ProngStatusNeg1) {
            continue;
          }

          // second loop over positive tracks
          //for (auto trackPos2 = trackPos1 + 1; trackPos2 != tracksPos.end(); ++trackPos2) {
          for (auto trackPos2 = trackPos1 + 1; trackPos2 != tracks.end(); ++trackPos2) {
            if (trackPos2.signed1Pt() < 0) {
              continue;
            }
            if (!(trackPos2.isSelProng() & (1 << 1))) {
              continue;
            }

            int isSelected3ProngCand = n3ProngBit;

            if (b_debug) {
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                for (int n3cut = 0; n3cut < nCuts3Prong; n3cut++) {
                  cutStatus3Prong[n3][n3cut] = true;
                }
              }
            }
            int iDebugCut = 0;

            // 3-prong invariant-mass cut
            auto arr3Mom = array{
              array{trackPos1.px(), trackPos1.py(), trackPos1.pz()},
              array{trackNeg1.px(), trackNeg1.py(), trackNeg1.pz()},
              array{trackPos2.px(), trackPos2.py(), trackPos2.pz()}};

            for (int n3 = 0; n3 < n3ProngDecays; n3++) {
              mass3ProngHypo1[n3] = RecoDecay::M(arr3Mom, arr3Mass1[n3]);
              mass3ProngHypo2[n3] = RecoDecay::M(arr3Mom, arr3Mass2[n3]);
              if ((isSelected3ProngCand & 1 << n3) && cut3ProngInvMassCandMin[n3] >= 0. && cut3ProngInvMassCandMax[n3] > 0.) {
                if ((mass3ProngHypo1[n3] < cut3ProngInvMassCandMin[n3] || mass3ProngHypo1[n3] >= cut3ProngInvMassCandMax[n3]) &&
                    (mass3ProngHypo2[n3] < cut3ProngInvMassCandMin[n3] || mass3ProngHypo2[n3] >= cut3ProngInvMassCandMax[n3])) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                  if (b_debug) {
                    cutStatus3Prong[n3][iDebugCut] = false;
                  }
                }
              }
            }
            if (!b_debug && isSelected3ProngCand == 0) {
              continue;
            }
            iDebugCut++;

            // reconstruct the 3-prong secondary vertex
            auto trackParVarPos2 = getTrackParCov(trackPos2);
            if (df3.process(trackParVarPos1, trackParVarNeg1, trackParVarPos2) == 0) {
              continue;
            }
            // get secondary vertex
            const auto& secondaryVertex3 = df3.getPCACandidate();
            // get track momenta
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);

            auto pVecCandProng3Pos = RecoDecay::PVec(pvec0, pvec1, pvec2);

            // candidate pT cut
            if (std::count_if(std::begin(cut3ProngPtCandMin), std::end(cut3ProngPtCandMin), [](double d) { return d >= 0.; }) > 0) {
              double cand3ProngPt = RecoDecay::Pt(pVecCandProng3Pos);
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if (cand3ProngPt < cut3ProngPtCandMin[n3]) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                }
                if (b_debug) {
                  cutStatus3Prong[n3][iDebugCut] = false;
                }
              }
              if (!b_debug && isSelected3ProngCand == 0) {
                continue; //this and all further instances should be changed if 4 track loop is added
              }
            }
            iDebugCut++;

            // CPA cut
            if (std::count_if(std::begin(cut3ProngCPACandMin), std::end(cut3ProngCPACandMin), [](double d) { return d > -2.; }) > 0) {
              auto cpa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3, pVecCandProng3Pos);
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if ((isSelected3ProngCand & 1 << n3) && cpa < cut3ProngCPACandMin[n3]) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                }
                if (b_debug) {
                  cutStatus3Prong[n3][iDebugCut] = false;
                }
              }
              if (!b_debug && isSelected3ProngCand == 0) {
                continue;
              }
            }
            iDebugCut++;

            // decay length cut
            if (std::count_if(std::begin(cut3ProngDecLenCandMin), std::end(cut3ProngDecLenCandMin), [](double d) { return d > 0.; }) > 0) {
              auto decayLength = RecoDecay::distance(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3);
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if ((isSelected3ProngCand & 1 << n3) && decayLength < cut3ProngDecLenCandMin[n3]) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                  if (b_debug) {
                    cutStatus3Prong[n3][iDebugCut] = false;
                  }
                }
              }
              if (!b_debug && isSelected3ProngCand == 0) {
                continue;
              }
            }
            iDebugCut++;

            // fill table row
            rowTrackIndexProng3(trackPos1.globalIndex(),
                                trackNeg1.globalIndex(),
                                trackPos2.globalIndex(), isSelected3ProngCand);

            if (b_debug) {
              int Prong3CutStatus[n3ProngDecays];
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                Prong3CutStatus[n3] = nCutStatus3ProngBit;
                for (int n3cut = 0; n3cut < nCuts3Prong; n3cut++) {
                  if (!cutStatus3Prong[n3][n3cut]) {
                    Prong3CutStatus[n3] = Prong3CutStatus[n3] & ~(1 << n3cut);
                  }
                }
              }
              rowProng3CutStatus(Prong3CutStatus[0], Prong3CutStatus[1], Prong3CutStatus[2], Prong3CutStatus[3]); //FIXME when we can do this by looping over n3ProngDecays
            }

            // fill histograms
            if (fillHistograms) {

              registry.get<TH1>(HIST("hvtx3_x"))->Fill(secondaryVertex3[0]);
              registry.get<TH1>(HIST("hvtx3_y"))->Fill(secondaryVertex3[1]);
              registry.get<TH1>(HIST("hvtx3_z"))->Fill(secondaryVertex3[2]);
              arr3Mom = array{pvec0, pvec1, pvec2};
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if (isSelected3ProngCand & 1 << n3) {
                  if ((cut3ProngInvMassCandMin[n3] < 0. && cut3ProngInvMassCandMax[n3] <= 0.) || (mass3ProngHypo1[n3] >= cut3ProngInvMassCandMin[n3] && mass3ProngHypo1[n3] < cut3ProngInvMassCandMax[n3])) {
                    mass3ProngHypo1[n3] = RecoDecay::M(arr3Mom, arr3Mass1[n3]);
                    if (n3 == hf_cand_prong3::DecayType::DPlusToPiKPi) {
                      registry.get<TH1>(HIST("hmassDPlusToPiKPi"))->Fill(mass3ProngHypo1[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::LcToPKPi) {
                      registry.get<TH1>(HIST("hmassLcToPKPi"))->Fill(mass3ProngHypo1[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::DsToPiKK) {
                      registry.get<TH1>(HIST("hmassDsToPiKK"))->Fill(mass3ProngHypo1[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::XicToPKPi) {
                      registry.get<TH1>(HIST("hmassXicToPKPi"))->Fill(mass3ProngHypo1[n3]);
                    }
                  }
                  if ((cut3ProngInvMassCandMin[n3] < 0. && cut3ProngInvMassCandMax[n3] <= 0.) || (mass3ProngHypo2[n3] >= cut3ProngInvMassCandMin[n3] && mass3ProngHypo2[n3] < cut3ProngInvMassCandMax[n3])) {
                    mass3ProngHypo2[n3] = RecoDecay::M(arr3Mom, arr3Mass2[n3]);
                    if (n3 == hf_cand_prong3::DecayType::LcToPKPi) {
                      registry.get<TH1>(HIST("hmassLcToPKPi"))->Fill(mass3ProngHypo2[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::DsToPiKK) {
                      registry.get<TH1>(HIST("hmassDsToPiKK"))->Fill(mass3ProngHypo2[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::XicToPKPi) {
                      registry.get<TH1>(HIST("hmassXicToPKPi"))->Fill(mass3ProngHypo2[n3]);
                    }
                  }
                }
              }
            }
          }

          // second loop over negative tracks
          //for (auto trackNeg2 = trackNeg1 + 1; trackNeg2 != tracksNeg.end(); ++trackNeg2) {
          for (auto trackNeg2 = trackNeg1 + 1; trackNeg2 != tracks.end(); ++trackNeg2) {
            if (trackNeg2.signed1Pt() > 0) {
              continue;
            }
            if (!(trackNeg2.isSelProng() & (1 << 1))) {
              continue;
            }

            int isSelected3ProngCand = n3ProngBit;

            if (b_debug) {
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                for (int n3cut = 0; n3cut < nCuts3Prong; n3cut++) {
                  cutStatus3Prong[n3][n3cut] = true;
                }
              }
            }
            int iDebugCut = 0;

            // 3-prong invariant-mass cut
            auto arr3Mom = array{
              array{trackNeg1.px(), trackNeg1.py(), trackNeg1.pz()},
              array{trackPos1.px(), trackPos1.py(), trackPos1.pz()},
              array{trackNeg2.px(), trackNeg2.py(), trackNeg2.pz()}};

            for (int n3 = 0; n3 < n3ProngDecays; n3++) {
              mass3ProngHypo1[n3] = RecoDecay::M(arr3Mom, arr3Mass1[n3]);
              mass3ProngHypo2[n3] = RecoDecay::M(arr3Mom, arr3Mass2[n3]);
              if ((isSelected3ProngCand & 1 << n3) && cut3ProngInvMassCandMin[n3] >= 0. && cut3ProngInvMassCandMax[n3] > 0.) {
                if ((mass3ProngHypo1[n3] < cut3ProngInvMassCandMin[n3] || mass3ProngHypo1[n3] >= cut3ProngInvMassCandMax[n3]) &&
                    (mass3ProngHypo2[n3] < cut3ProngInvMassCandMin[n3] || mass3ProngHypo2[n3] >= cut3ProngInvMassCandMax[n3])) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                  if (b_debug) {
                    cutStatus3Prong[n3][iDebugCut] = false;
                  }
                }
              }
            }
            if (!b_debug && isSelected3ProngCand == 0) {
              continue;
            }
            iDebugCut++;

            // reconstruct the 3-prong secondary vertex
            auto trackParVarNeg2 = getTrackParCov(trackNeg2);
            if (df3.process(trackParVarNeg1, trackParVarPos1, trackParVarNeg2) == 0) {
              continue;
            }

            // get secondary vertex
            const auto& secondaryVertex3 = df3.getPCACandidate();
            // get track momenta
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);

            auto pVecCandProng3Neg = RecoDecay::PVec(pvec0, pvec1, pvec2);

            // candidate pT cut
            if (std::count_if(std::begin(cut3ProngPtCandMin), std::end(cut3ProngPtCandMin), [](double d) { return d >= 0.; }) > 0) {
              double cand3ProngPt = RecoDecay::Pt(pVecCandProng3Neg);
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if (cand3ProngPt < cut3ProngPtCandMin[n3]) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                  if (b_debug) {
                    cutStatus3Prong[n3][iDebugCut] = false;
                  }
                }
              }
              if (!b_debug && isSelected3ProngCand == 0) {
                continue;
              }
            }
            iDebugCut++;

            // CPA cut
            if (std::count_if(std::begin(cut3ProngCPACandMin), std::end(cut3ProngCPACandMin), [](double d) { return d > -2.; }) > 0) {
              auto cpa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3, pVecCandProng3Neg);
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if ((isSelected3ProngCand & 1 << n3) && cpa < cut3ProngCPACandMin[n3]) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                  if (b_debug) {
                    cutStatus3Prong[n3][iDebugCut] = false;
                  }
                }
              }
              if (!b_debug && isSelected3ProngCand == 0) {
                continue;
              }
            }
            iDebugCut++;

            // decay length cut
            if (std::count_if(std::begin(cut3ProngDecLenCandMin), std::end(cut3ProngDecLenCandMin), [](double d) { return d > 0.; }) > 0) {
              auto decayLength = RecoDecay::distance(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3);
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if ((isSelected3ProngCand & 1 << n3) && decayLength < cut3ProngDecLenCandMin[n3]) {
                  isSelected3ProngCand = isSelected3ProngCand & ~(1 << n3);
                  if (b_debug) {
                    cutStatus3Prong[n3][iDebugCut] = false;
                  }
                }
              }
              if (!b_debug && isSelected3ProngCand == 0) {
                continue;
              }
            }
            iDebugCut++;

            // fill table row
            rowTrackIndexProng3(trackNeg1.globalIndex(),
                                trackPos1.globalIndex(),
                                trackNeg2.globalIndex(), isSelected3ProngCand);

            if (b_debug) {
              int Prong3CutStatus[n3ProngDecays];
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                Prong3CutStatus[n3] = nCutStatus3ProngBit;
                for (int n3cut = 0; n3cut < nCuts3Prong; n3cut++) {
                  if (!cutStatus3Prong[n3][n3cut]) {
                    Prong3CutStatus[n3] = Prong3CutStatus[n3] & ~(1 << n3cut);
                  }
                }
              }
              rowProng3CutStatus(Prong3CutStatus[0], Prong3CutStatus[1], Prong3CutStatus[2], Prong3CutStatus[3]); //FIXME when we can do this by looping over n3ProngDecays
            }

            // fill histograms
            if (fillHistograms) {

              registry.get<TH1>(HIST("hvtx3_x"))->Fill(secondaryVertex3[0]);
              registry.get<TH1>(HIST("hvtx3_y"))->Fill(secondaryVertex3[1]);
              registry.get<TH1>(HIST("hvtx3_z"))->Fill(secondaryVertex3[2]);
              arr3Mom = array{pvec0, pvec1, pvec2};
              for (int n3 = 0; n3 < n3ProngDecays; n3++) {
                if (isSelected3ProngCand & 1 << n3) {
                  if ((cut3ProngInvMassCandMin[n3] < 0. && cut3ProngInvMassCandMax[n3] <= 0.) || (mass3ProngHypo1[n3] >= cut3ProngInvMassCandMin[n3] && mass3ProngHypo1[n3] < cut3ProngInvMassCandMax[n3])) {
                    mass3ProngHypo1[n3] = RecoDecay::M(arr3Mom, arr3Mass1[n3]);
                    if (n3 == hf_cand_prong3::DecayType::DPlusToPiKPi) {
                      registry.get<TH1>(HIST("hmassDPlusToPiKPi"))->Fill(mass3ProngHypo1[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::LcToPKPi) {
                      registry.get<TH1>(HIST("hmassLcToPKPi"))->Fill(mass3ProngHypo1[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::DsToPiKK) {
                      registry.get<TH1>(HIST("hmassDsToPiKK"))->Fill(mass3ProngHypo1[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::XicToPKPi) {
                      registry.get<TH1>(HIST("hmassXicToPKPi"))->Fill(mass3ProngHypo1[n3]);
                    }
                  }
                  if ((cut3ProngInvMassCandMin[n3] < 0. && cut3ProngInvMassCandMax[n3] <= 0.) || (mass3ProngHypo2[n3] >= cut3ProngInvMassCandMin[n3] && mass3ProngHypo2[n3] < cut3ProngInvMassCandMax[n3])) {
                    mass3ProngHypo2[n3] = RecoDecay::M(arr3Mom, arr3Mass2[n3]);
                    if (n3 == hf_cand_prong3::DecayType::LcToPKPi) {
                      registry.get<TH1>(HIST("hmassLcToPKPi"))->Fill(mass3ProngHypo2[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::DsToPiKK) {
                      registry.get<TH1>(HIST("hmassDsToPiKK"))->Fill(mass3ProngHypo2[n3]);
                    }
                    if (n3 == hf_cand_prong3::DecayType::XicToPKPi) {
                      registry.get<TH1>(HIST("hmassXicToPKPi"))->Fill(mass3ProngHypo2[n3]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    auto nTracks = tracks.size();                      // number of tracks passing 2 and 3 prong selection in this collision
    nCand2 = rowTrackIndexProng2.lastIndex() - nCand2; // number of 2-prong candidates in this collision
    nCand3 = rowTrackIndexProng3.lastIndex() - nCand3; // number of 3-prong candidates in this collision

    registry.get<TH1>(HIST("hNTracks"))->Fill(nTracks);
    registry.get<TH1>(HIST("hNCand2Prong"))->Fill(nCand2);
    registry.get<TH1>(HIST("hNCand3Prong"))->Fill(nCand3);
    registry.get<TH2>(HIST("hNCand2ProngVsNTracks"))->Fill(nTracks, nCand2);
    registry.get<TH2>(HIST("hNCand3ProngVsNTracks"))->Fill(nTracks, nCand3);
  }
};

//________________________________________________________________________________________________________________________

/// Pre-selection of cascade secondary vertices
/// It will produce in any case a HfTrackIndexProng2 object, but mixing a V0
/// with a track, instead of 2 tracks

/// to run: o2-analysis-weak-decay-indices --aod-file AO2D.root -b | o2-analysis-lambdakzerobuilder -b |
///         o2-analysis-trackextension -b | o2-analysis-hf-track-index-skims-creator -b

struct HfTrackIndexSkimsCreatorCascades {
  Produces<aod::HfTrackIndexCasc> rowTrackIndexCasc;
  //  Produces<aod::HfTrackIndexProng2> rowTrackIndexCasc;

  // whether to do or not validation plots
  Configurable<bool> doValPlots{"doValPlots", true, "fill histograms"};

  // event selection
  //Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};

  // vertexing parameters
  Configurable<double> bZ{"bZ", 5., "magnetic field"};
  Configurable<bool> propDCA{"propDCA", true, "create tracks version propagated to PCA"};
  Configurable<double> maxR{"maxR", 200., "reject PCA's above this radius"};
  Configurable<double> maxDZIni{"maxDZIni", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> minParamChange{"minParamChange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> minRelChi2Change{"minRelChi2Change", 0.9, "stop iterations if chi2/chi2old > this"};
  Configurable<bool> UseAbsDCA{"UseAbsDCA", true, "Use Abs DCAs"};

  // quality cut
  Configurable<bool> doCutQuality{"doCutQuality", true, "apply quality cuts"};

  // track cuts for V0 daughters
  Configurable<bool> TPCRefit{"TPCRefit", true, "request TPC refit V0 daughters"};
  Configurable<int> i_minCrossedRows{"i_minCrossedRows", 50, "min crossed rows V0 daughters"};
  Configurable<double> etaMax{"etaMax", 1.1, "max. pseudorapidity V0 daughters"};
  Configurable<double> ptMin{"ptMin", 0.05, "min. pT V0 daughters"};

  // bachelor cuts
  //  Configurable<float> dcabachtopv{"dcabachtopv", .1, "DCA Bach To PV"};
  //  Configurable<double> ptminbach{"ptminbach", -1., "min. track pT bachelor"};

  // v0 cuts
  Configurable<double> cosPAV0{"cosPAV0", .995, "CosPA V0"};                  // as in the task that create the V0s
  Configurable<double> dcaXYNegToPV{"dcaXYNegToPV", .1, "DCA_XY Neg To PV"};  // check: in HF Run 2, it was 0 at filtering
  Configurable<double> dcaXYPosToPV{"dcaXYPosToPVS", .1, "DCA_XY Pos To PV"}; // check: in HF Run 2, it was 0 at filtering
  Configurable<double> cutInvMassV0{"cutInvMassV0", 0.05, "V0 candidate invariant mass difference wrt PDG"};

  // cascade cuts
  Configurable<double> cutCascPtCandMin{"cutCascPtCandMin", -1., "min. pT of the 2-prong candidate"};              // PbPb 2018: use 1
  Configurable<double> cutCascInvMassLc{"cutCascInvMassLc", 1., "Lc candidate invariant mass difference wrt PDG"}; // for PbPb 2018: use 0.2
  //Configurable<double> cutCascDCADaughters{"cutCascDCADaughters", .1, "DCA between V0 and bachelor in cascade"};

  // for debugging
#ifdef MY_DEBUG
  Configurable<std::vector<int>> indexK0Spos{"indexK0Spos", {729, 2866, 4754, 5457, 6891, 7824, 9243, 9810}, "indices of K0S positive daughters, for debug"};
  Configurable<std::vector<int>> indexK0Sneg{"indexK0Sneg", {730, 2867, 4755, 5458, 6892, 7825, 9244, 9811}, "indices of K0S negative daughters, for debug"};
  Configurable<std::vector<int>> indexProton{"indexProton", {717, 2810, 4393, 5442, 6769, 7793, 9002, 9789}, "indices of protons, for debug"};
#endif

  // histograms
  HistogramRegistry registry{
    "registry",
    {{"hvtx2_x", "2-prong candidates;#it{x}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -2., 2.}}}},
     {"hvtx2_y", "2-prong candidates;#it{y}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -2., 2.}}}},
     {"hvtx2_z", "2-prong candidates;#it{z}_{sec. vtx.} (cm);entries", {HistType::kTH1F, {{1000, -20., 20.}}}},
     {"hmass2", "2-prong candidates;inv. mass (K0s p) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}}}};

  // NB: using FullTracks = soa::Join<Tracks, TracksCov, TracksExtra>; defined in Framework/Core/include/Framework/AnalysisDataModel.h
  //using MyTracks = aod::BigTracksMC;
  //Partition<MyTracks> selectedTracks = aod::hf_seltrack::isSelProng >= 4;
  // using SelectedV0s = soa::Filtered<aod::V0Datas>;

  double massP = RecoDecay::getMassPDG(kProton);
  double massK0s = RecoDecay::getMassPDG(kK0Short);
  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massLc = RecoDecay::getMassPDG(pdg::Code::kLambdaCPlus);
  double mass2K0sP{0.}; // WHY HERE?

  Filter filterSelectCollisions = (aod::hf_selcollision::whyRejectColl == 0);

  using SelectedCollisions = soa::Filtered<soa::Join<aod::Collisions, aod::HFSelCollision>>;
  using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtended>;

  void process(SelectedCollisions::iterator const& collision,
               aod::BCs const& bcs,
               //soa::Filtered<aod::V0Datas> const& V0s,
               aod::V0Datas const& V0s,
               MyTracks const& tracks
#ifdef MY_DEBUG
               ,
               aod::McParticles& mcParticles
#endif
               ) // TODO: I am now assuming that the V0s are already filtered with my cuts (David's work to come)
  {

    //Define o2 fitter, 2-prong
    o2::vertexing::DCAFitterN<2> fitter;
    fitter.setBz(bZ);
    fitter.setPropagateToPCA(propDCA);
    fitter.setMaxR(maxR);
    fitter.setMinParamChange(minParamChange);
    fitter.setMinRelChi2Change(minRelChi2Change);
    //fitter.setMaxDZIni(1e9); // used in cascadeproducer.cxx, but not for the 2 prongs
    //fitter.setMaxChi2(1e9);  // used in cascadeproducer.cxx, but not for the 2 prongs
    fitter.setUseAbsDCA(UseAbsDCA);

    // fist we loop over the bachelor candidate

    //for (const auto& bach : selectedTracks) {
    for (const auto& bach : tracks) {

      MY_DEBUG_MSG(1, printf("\n"); LOG(INFO) << "Bachelor loop");
#ifdef MY_DEBUG
      auto indexBach = bach.mcParticleId();
      bool isProtonFromLc = isProtonFromLcFunc(indexBach, indexProton);
#endif
      // selections on the bachelor
      // pT cut
      if (bach.isSelProng() < 4) {
        MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "proton " << indexBach << ": rejected due to HFsel");
        continue;
      }

      if (TPCRefit) {
        if (!(bach.trackType() & o2::aod::track::TPCrefit)) {
          MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "proton " << indexBach << ": rejected due to TPCrefit");
          continue;
        }
      }
      if (bach.tpcNClsCrossedRows() < i_minCrossedRows) {
        MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "proton " << indexBach << ": rejected due to minNUmberOfCrossedRows " << bach.tpcNClsCrossedRows() << " (cut " << i_minCrossedRows << ")");
        continue;
      }
      MY_DEBUG_MSG(isProtonFromLc, LOG(INFO) << "KEPT! proton from Lc with daughters " << indexBach);

      auto trackBach = getTrackParCov(bach);
      // now we loop over the V0s
      for (const auto& v0 : V0s) {
        MY_DEBUG_MSG(1, LOG(INFO) << "*** Checking next K0S");
        // selections on the V0 daughters
        const auto& trackV0DaughPos = v0.posTrack_as<MyTracks>();
        const auto& trackV0DaughNeg = v0.negTrack_as<MyTracks>();
#ifdef MY_DEBUG
        auto indexV0DaughPos = trackV0DaughPos.mcParticleId();
        auto indexV0DaughNeg = trackV0DaughNeg.mcParticleId();
        bool isK0SfromLc = isK0SfromLcFunc(indexV0DaughPos, indexV0DaughNeg, indexK0Spos, indexK0Sneg);

        bool isLc = isLcK0SpFunc(indexBach, indexV0DaughPos, indexV0DaughNeg, indexProton, indexK0Spos, indexK0Sneg);
#endif
        MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "K0S from Lc found, trackV0DaughPos --> " << indexV0DaughPos << ", trackV0DaughNeg --> " << indexV0DaughNeg);

        MY_DEBUG_MSG(isK0SfromLc && isProtonFromLc,
                     LOG(INFO) << "ACCEPTED!!!";
                     LOG(INFO) << "proton belonging to a Lc found: label --> " << indexBach;
                     LOG(INFO) << "K0S belonging to a Lc found: trackV0DaughPos --> " << indexV0DaughPos << ", trackV0DaughNeg --> " << indexV0DaughNeg);

        MY_DEBUG_MSG(isLc, LOG(INFO) << "Combination of K0S and p which correspond to a Lc found!");

        if (TPCRefit) {
          if (!(trackV0DaughPos.trackType() & o2::aod::track::TPCrefit) ||
              !(trackV0DaughNeg.trackType() & o2::aod::track::TPCrefit)) {
            MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "K0S with daughters " << indexV0DaughPos << " and " << indexV0DaughNeg << ": rejected due to TPCrefit");
            continue;
          }
        }
        if (trackV0DaughPos.tpcNClsCrossedRows() < i_minCrossedRows ||
            trackV0DaughNeg.tpcNClsCrossedRows() < i_minCrossedRows) {
          MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "K0S with daughters " << indexV0DaughPos << " and " << indexV0DaughNeg << ": rejected due to minCrossedRows");
          continue;
        }
        //
        // if (trackV0DaughPos.dcaXY() < dcaXYPosToPV ||   // to the filters?
        //     trackV0DaughNeg.dcaXY() < dcaXYNegToPV) {
        //   continue;
        // }
        //
        if (trackV0DaughPos.pt() < ptMin || // to the filters? I can't for now, it is not in the tables
            trackV0DaughNeg.pt() < ptMin) {
          MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "K0S with daughters " << indexV0DaughPos << " and " << indexV0DaughNeg << ": rejected due to minPt --> pos " << trackV0DaughPos.pt() << ", neg " << trackV0DaughNeg.pt() << " (cut " << ptMin << ")");
          continue;
        }
        if (std::abs(trackV0DaughPos.eta()) > etaMax || // to the filters? I can't for now, it is not in the tables
            std::abs(trackV0DaughNeg.eta()) > etaMax) {
          MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "K0S with daughters " << indexV0DaughPos << " and " << indexV0DaughNeg << ": rejected due to eta --> pos " << trackV0DaughPos.eta() << ", neg " << trackV0DaughNeg.eta() << " (cut " << etaMax << ")");
          continue;
        }

        // V0 invariant mass selection
        if (std::abs(v0.mK0Short() - massK0s) > cutInvMassV0) {
          MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "K0S with daughters " << indexV0DaughPos << " and " << indexV0DaughNeg << ": rejected due to invMass --> " << v0.mK0Short() - massK0s << " (cut " << cutInvMassV0 << ")");
          continue; // should go to the filter, but since it is a dynamic column, I cannot use it there
        }

        // V0 cosPointingAngle selection
        if (v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) < cosPAV0) {
          MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "K0S with daughters " << indexV0DaughPos << " and " << indexV0DaughNeg << ": rejected due to cosPA --> " << v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) << " (cut " << cosPAV0 << ")");
          continue;
        }

        const std::array<float, 3> momentumV0 = {v0.px(), v0.py(), v0.pz()};

        // invariant-mass cut: we do it here, before updating the momenta of bach and V0 during the fitting to save CPU
        // TODO: but one should better check that the value here and after the fitter do not change significantly!!!
        mass2K0sP = RecoDecay::M(array{array{bach.px(), bach.py(), bach.pz()}, momentumV0}, array{massP, massK0s});
        if ((cutCascInvMassLc >= 0.) && (std::abs(mass2K0sP - massLc) > cutCascInvMassLc)) {
          MY_DEBUG_MSG(isK0SfromLc && isProtonFromLc, LOG(INFO) << "True Lc from proton " << indexBach << " and K0S pos " << indexV0DaughPos << " and neg " << indexV0DaughNeg << " rejected due to invMass cut: " << mass2K0sP << ", mass Lc " << massLc << " (cut " << cutCascInvMassLc << ")");
          continue;
        }

        MY_DEBUG_MSG(isK0SfromLc, LOG(INFO) << "KEPT! K0S from Lc with daughters " << indexV0DaughPos << " and " << indexV0DaughNeg);

        auto trackParCovV0DaughPos = getTrackParCov(trackV0DaughPos);
        trackParCovV0DaughPos.propagateTo(v0.posX(), bZ); // propagate the track to the X closest to the V0 vertex
        auto trackParCovV0DaughNeg = getTrackParCov(trackV0DaughNeg);
        trackParCovV0DaughNeg.propagateTo(v0.negX(), bZ); // propagate the track to the X closest to the V0 vertex
        std::array<float, 3> pVecV0 = {0., 0., 0.};
        std::array<float, 3> pVecBach = {0., 0., 0.};

        const std::array<float, 3> vertexV0 = {v0.x(), v0.y(), v0.z()};
        // we build the neutral track to then build the cascade
        auto trackV0 = o2::dataformats::V0(vertexV0, momentumV0, {0, 0, 0, 0, 0, 0}, trackParCovV0DaughPos, trackParCovV0DaughNeg, {0, 0}, {0, 0}); // build the V0 track

        // now we find the DCA between the V0 and the bachelor, for the cascade
        int nCand2 = fitter.process(trackV0, trackBach);
        MY_DEBUG_MSG(isK0SfromLc && isProtonFromLc, LOG(INFO) << "Fitter result = " << nCand2 << " proton = " << indexBach << " and K0S pos " << indexV0DaughPos << " and neg " << indexV0DaughNeg);
        MY_DEBUG_MSG(isLc, LOG(INFO) << "Fitter result for true Lc = " << nCand2);
        if (nCand2 == 0) {
          continue;
        }
        fitter.propagateTracksToVertex();        // propagate the bach and V0 to the Lc vertex
        fitter.getTrack(0).getPxPyPzGlo(pVecV0); // take the momentum at the Lc vertex
        fitter.getTrack(1).getPxPyPzGlo(pVecBach);

        // cascade candidate pT cut
        auto ptCascCand = RecoDecay::Pt(pVecBach, pVecV0);
        if (ptCascCand < cutCascPtCandMin) {
          MY_DEBUG_MSG(isK0SfromLc && isProtonFromLc, LOG(INFO) << "True Lc from proton " << indexBach << " and K0S pos " << indexV0DaughPos << " and neg " << indexV0DaughNeg << " rejected due to pt cut: " << ptCascCand << " (cut " << cutCascPtCandMin << ")");
          continue;
        }

        // invariant mass
        // re-calculate invariant masses with updated momenta, to fill the histogram
        mass2K0sP = RecoDecay::M(array{pVecBach, pVecV0}, array{massP, massK0s});

        std::array<float, 3> posCasc = {0., 0., 0.};
        const auto& cascVtx = fitter.getPCACandidate();
        for (int i = 0; i < 3; i++) {
          posCasc[i] = cascVtx[i];
        }

        // fill table row
        rowTrackIndexCasc(bach.globalIndex(),
                          v0.globalIndex(),
                          1); // 1 should be the value for the Lc
        // fill histograms
        if (doValPlots) {
          MY_DEBUG_MSG(isK0SfromLc && isProtonFromLc && isLc, LOG(INFO) << "KEPT! True Lc from proton " << indexBach << " and K0S pos " << indexV0DaughPos << " and neg " << indexV0DaughNeg);
          registry.get<TH1>(HIST("hvtx2_x"))->Fill(posCasc[0]);
          registry.get<TH1>(HIST("hvtx2_y"))->Fill(posCasc[1]);
          registry.get<TH1>(HIST("hvtx2_z"))->Fill(posCasc[2]);
          registry.get<TH1>(HIST("hmass2"))->Fill(mass2K0sP);
        }

      } // loop over V0s

    } // loop over tracks
  }   // process
};

//________________________________________________________________________________________________________________________

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};

  const bool doEvSel = cfgc.options().get<bool>("doEvSel");
  if (doEvSel) {
    workflow.push_back(adaptAnalysisTask<HfTagSelCollisions>(cfgc, Processes{&HfTagSelCollisions::processEvSel}));
  } else {
    workflow.push_back(adaptAnalysisTask<HfTagSelCollisions>(cfgc, Processes{&HfTagSelCollisions::processNoEvSel}));
  }

  workflow.push_back(adaptAnalysisTask<HfTagSelTracks>(cfgc));
  workflow.push_back(adaptAnalysisTask<HfTrackIndexSkimsCreator>(cfgc));

  const bool doLcK0Sp = cfgc.options().get<bool>("do-LcK0Sp");
  if (doLcK0Sp) {
    workflow.push_back(adaptAnalysisTask<HfTrackIndexSkimsCreatorCascades>(cfgc));
  }

  return workflow;
}
