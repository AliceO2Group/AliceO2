// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
// O2 includes

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/HistogramRegistry.h"

#include "filterTables.h"

#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/TrackSelectorPID.h"

#include <cmath>
#include <string>

namespace {

  enum HfTriggers {
    kHighPt = 0,
    kBeauty,
    kFemto,
    kNtriggersHF
  };

  static const std::vector<std::string> HfTriggerNames{"highPt", "beauty", "femto"};

  static const float massPi = RecoDecay::getMassPDG(211);
  static const float massK = RecoDecay::getMassPDG(321);
  static const float massProton = RecoDecay::getMassPDG(2212);
  static const float massD0 = RecoDecay::getMassPDG(421);
  static const float massDPlus = RecoDecay::getMassPDG(411);
  static const float massDs = RecoDecay::getMassPDG(431);
  static const float massLc = RecoDecay::getMassPDG(4122);
  static const float massBPlus = RecoDecay::getMassPDG(511);
  static const float massB0 = RecoDecay::getMassPDG(521);
  static const float massBs = RecoDecay::getMassPDG(531);
  static const float massLb = RecoDecay::getMassPDG(5122);

} // namespace


using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::aod::hf_cand_prong3;

struct HfFilter {

  Produces<aod::HfFilters> tags;

  Configurable<float> ptThreshold2Prong{"ptThreshold2Prong", 5., "pT treshold for high pT 2-prong candidates for kHighPt triggers in GeV/c"};
  Configurable<float> ptThreshold3Prong{"ptThreshold3Prong", 5., "pT treshold for high pT 3-prong candidates for kHighPt triggers in GeV/c"};
  Configurable<float> deltaMassBPlus{"deltaMassBPlus", 0.3, "invariant-mass delta with respect to the B+ mass"};
  Configurable<float> deltaMassB0{"deltaMassB0", 0.3, "invariant-mass delta with respect to the B0 mass"};
  Configurable<float> deltaMassBs{"deltaMassBs", 0.3, "invariant-mass delta with respect to the Bs mass"};
  Configurable<float> deltaMassLb{"deltaMassLb", 0.3, "invariant-mass delta with respect to the Lb mass"};

  HistogramRegistry registry{"registry", {}, OutputObjHandlingPolicy::AnalysisObject, true, true};

  void init(o2::framework::InitContext&)
  {
    registry.add("fProcessedEvents", "HF - event filtered", HistType::kTH1F, {{4, -0.5, 4.5, "Event counter"}});
    std::array<std::string, 4> eventTitles = {"rejected", "w/ high-#it{p}_{T} candidate", "w/ beauty candidate", "w/ femto candidate"};
    for(size_t iBin=0; iBin<eventTitles.size(); iBin++) {
      registry.get<TH1>(HIST("fProcessedEvents"))->GetXaxis()->SetBinLabel(iBin+1, eventTitles[iBin].data());
    }
  }

  void process(aod::Collision const& collision,
               aod::HfTrackIndexProng2 const& cand2Prongs,
               aod::HfTrackIndexProng3 const& cand3Prongs,
               aod::BigTracks const& tracks)
  {
    // collision process loop
    bool keepEvent[kNtriggersHF]{false};
    //

    for (auto& cand2Prong : cand2Prongs) { // start loop over 2 prongs

      if(!TESTBIT(cand2Prong.hfflag(), o2::aod::hf_cand_prong2::DecayType::D0ToPiK)) { // check if it's a D0
        continue;
      }

      auto trackPos = cand2Prong.index0_as<aod::BigTracks>(); // positive daughter
      auto trackNeg = cand2Prong.index1_as<aod::BigTracks>(); // negative daughter
      std::array<float, 3> pVecPos = {trackPos.px(), trackPos.py(), trackPos.pz()};
      std::array<float, 3> pVecNeg = {trackNeg.px(), trackNeg.py(), trackNeg.pz()};

      auto pVec2Prong = RecoDecay::PVec(pVecPos, pVecNeg);
      auto pt2Prong = RecoDecay::Pt(pVec2Prong);
      if(pt2Prong >= ptThreshold2Prong) {
        keepEvent[kHighPt] = true;
      }

      for (auto track : tracks) { // start loop over tracks

        if(track.globalIndex() == trackPos.globalIndex() || track.globalIndex() == trackNeg.globalIndex()) {
          continue;
        }

        std::array<float, 3> pVecThird = {track.px(), track.py(), track.pz()};

        if(!keepEvent[kBeauty]) {
          auto massCandB = RecoDecay::M(std::array{pVec2Prong, pVecThird}, std::array{massD0, massPi}); // TODO: retrieve D0-D0bar hypothesis to pair with proper signed track
          if(abs(massCandB - massBPlus) <= deltaMassBPlus) {
            keepEvent[kBeauty] = true;
          }
        }

        // TODO: add momentum correlation with a track for femto

      } // end loop over tracks
    } // end loop over 2-prong candidates

    for (auto& cand3Prong : cand3Prongs) { // start loop over 2 prongs

      bool isDPlus = TESTBIT(cand3Prong.hfflag(), o2::aod::hf_cand_prong3::DecayType::DPlusToPiKPi);
      bool isDs = TESTBIT(cand3Prong.hfflag(), o2::aod::hf_cand_prong3::DecayType::DsToPiKK);
      bool isLc = TESTBIT(cand3Prong.hfflag(), o2::aod::hf_cand_prong3::DecayType::LcToPKPi);
      if(!isDPlus && !isDs && !isLc) { // check if it's a D+, Ds+ or Lc+ 
        continue;
      }

      auto trackFirst = cand3Prong.index0_as<aod::BigTracks>();
      auto trackSecond = cand3Prong.index1_as<aod::BigTracks>();
      auto trackThird = cand3Prong.index2_as<aod::BigTracks>();
      std::array<float, 3> pVecFirst = {trackFirst.px(), trackFirst.py(), trackFirst.pz()};
      std::array<float, 3> pVecSecond = {trackSecond.px(), trackSecond.py(), trackSecond.pz()};
      std::array<float, 3> pVecThird = {trackThird.px(), trackThird.py(), trackThird.pz()};

      float sign3Prong = trackFirst.signed1Pt() * trackSecond.signed1Pt() * trackThird.signed1Pt();

      auto pVec3Prong = RecoDecay::PVec(pVecFirst, pVecSecond, pVecThird);
      auto pt3Prong = RecoDecay::Pt(pVec3Prong);
      if(pt3Prong >= ptThreshold3Prong) {
        keepEvent[kHighPt] = true;
      }

      for (auto track : tracks) { // start loop over tracks

        if(track.globalIndex() == trackFirst.globalIndex() || track.globalIndex() == trackSecond.globalIndex() || track.globalIndex() == trackThird.globalIndex()) {
          continue;
        }

        std::array<float, 3> pVecFourth = {track.px(), track.py(), track.pz()};

        int iHypo=0;
        bool specieCharmHypos[3] = {isDPlus, isDs, isLc};
        float massCharmHypos[3] = {massDPlus, massDs, massLc};
        float massBeautyHypos[3] = {massB0, massBs, massLb};
        float deltaMassHypos[3] = {deltaMassB0, deltaMassBs, deltaMassLb};
        if(track.signed1Pt() * sign3Prong < 0) {
          while(!keepEvent[kBeauty] && iHypo < 3) {
            if(specieCharmHypos[iHypo]) {
              auto massCandB = RecoDecay::M(std::array{pVec3Prong, pVecFourth}, std::array{massCharmHypos[iHypo], massPi});
              if(abs(massCandB - massBeautyHypos[iHypo]) <= deltaMassHypos[iHypo]) {
                keepEvent[kBeauty] = true;
              }
            }
            iHypo++;
          }
        }

        // TODO: add momentum correlation with a track for femto

      } // end loop over tracks
    } // end loop over 3-prong candidates

    tags(keepEvent[kHighPt], keepEvent[kBeauty], keepEvent[kFemto]);

    if(!keepEvent[kHighPt] && !keepEvent[kBeauty] && !keepEvent[kFemto]) {
      registry.get<TH1>(HIST("fProcessedEvents"))->Fill(1);
    }
    else {
      for(int iTrigger=0; iTrigger<kNtriggersHF; iTrigger++) {
        if(keepEvent[iTrigger]) {
          registry.get<TH1>(HIST("fProcessedEvents"))->Fill(iTrigger+2);
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  return WorkflowSpec{
    adaptAnalysisTask<HfFilter>(cfg)};
}
