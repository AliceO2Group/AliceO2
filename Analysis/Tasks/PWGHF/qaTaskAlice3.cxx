// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Peter Hristov <Peter.Hristov@cern.ch>, CERN
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Henrique J C Zanoli <henrique.zanoli@cern.ch>, Utrecht University
/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

// O2 inlcudes
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/TrackSelectorPID.h"
#include "ALICE3Analysis/RICH.h"
#include "ALICE3Analysis/MID.h"

using namespace o2;

using namespace o2::framework;

namespace o2::aod
{

namespace hf_track_index_alice3_pid
{
DECLARE_SOA_INDEX_COLUMN(Track, track); //!
DECLARE_SOA_INDEX_COLUMN(RICH, rich);   //!
DECLARE_SOA_INDEX_COLUMN(MID, mid);     //!
} // namespace hf_track_index_alice3_pid

DECLARE_SOA_INDEX_TABLE_USER(HfTrackIndexALICE3PID, Tracks, "HFTRKIDXA3PID", //!
                             hf_track_index_alice3_pid::TrackId,
                             hf_track_index_alice3_pid::RICHId,
                             hf_track_index_alice3_pid::MIDId);
} // namespace o2::aod

struct Alice3PidIndexBuilder {
  Builds<o2::aod::HfTrackIndexALICE3PID> index;
  void init(o2::framework::InitContext&) {}
};

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"rej-el", VariantType::Int, 1, {"Efficiency for the Electron PDG code"}},
    {"rej-mu", VariantType::Int, 1, {"Efficiency for the Muon PDG code"}},
    {"rej-pi", VariantType::Int, 1, {"Efficiency for the Pion PDG code"}},
    {"rej-ka", VariantType::Int, 0, {"Efficiency for the Kaon PDG code"}},
    {"rej-pr", VariantType::Int, 0, {"Efficiency for the Proton PDG code"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

// ROOT includes
#include "TPDGCode.h"
#include "TEfficiency.h"
#include "TList.h"


/// Task to QA the efficiency of a particular particle defined by particlePDG
template <o2::track::pid_constants::ID particle>
struct QaTrackingRejection {
  static constexpr PDG_t PDGs[5] = {kElectron, kMuonMinus, kPiPlus, kKPlus, kProton};
  static_assert(particle < 5 && "Maximum of particles reached");
  static constexpr int particlePDG = PDGs[particle];
  // Particle selection
  Configurable<int> ptBins{"ptBins", 100, "Number of pT bins"};
  Configurable<float> ptMin{"ptMin", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"ptMax", 5.f, "Upper limit in pT"};
  HistogramRegistry histos{"HistogramsRejection"};

  void init(InitContext&)
  {
    AxisSpec ptAxis{ptBins, ptMin, ptMax};

    TString commonTitle = "";
    if (particlePDG != 0) {
      commonTitle += Form("PDG %i", particlePDG);
    }

    const TString pt = "#it{p}_{T} [GeV/#it{c}]";
    const TString eta = "#it{#eta}";
    const TString phi = "#it{#varphi} [rad]";

    histos.add("tracking/pt", commonTitle + ";" + pt, kTH1D, {ptAxis});
    histos.add("trackingPrm/pt", commonTitle + " Primary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingSec/pt", commonTitle + " Secondary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingTOFsel/pt", commonTitle + ";" + pt, kTH1D, {ptAxis});
    histos.add("trackingPrmTOFsel/pt", commonTitle + " Primary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingSecTOFsel/pt", commonTitle + " Secondary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingRICHsel/pt", commonTitle + ";" + pt, kTH1D, {ptAxis});
    histos.add("trackingPrmRICHsel/pt", commonTitle + " Primary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingSecRICHsel/pt", commonTitle + " Secondary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingMIDsel/pt", commonTitle + ";" + pt, kTH1D, {ptAxis});
    histos.add("trackingPrmMIDsel/pt", commonTitle + " Primary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingSecMIDsel/pt", commonTitle + " Secondary;" + pt, kTH1D, {ptAxis});
  }
  
  using TracksPID = soa::Join<aod::BigTracksPID, aod::HfTrackIndexALICE3PID>;

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>& collisions,
               const o2::soa::Join<TracksPID, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McCollisions& mcCollisions,
               const o2::aod::McParticles& mcParticles, aod::RICHs const&, aod::MIDs const&)
  {
    TrackSelectorPID selector(particlePDG);
    selector.setRangePtTOF(0.15, 1.);
    selector.setRangeNSigmaTOF(-3.,3.);
    selector.setRangeNSigmaTOFCondTPC(0., 0.);
    selector.setRangePtRICH(1., 5.);
    selector.setRangeNSigmaRICH(-3., 3.);
    selector.setRangeNSigmaRICHCondTOF(0.,0.);

    TrackSelectorPID selectorMuon(particlePDG);
 
    std::vector<int64_t> recoEvt(collisions.size());
    std::vector<int64_t> recoTracks(tracks.size());
    LOGF(info, "%d", particlePDG);
    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if (particlePDG != 0 && mcParticle.pdgCode() != particlePDG) { // Checking PDG code
        continue;
      }
      bool isTOF = selector.getStatusTrackPIDTOF(track) == TrackSelectorPID::Status::PIDRejected;
      bool isRICH = selector.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected;
      bool isMID = selectorMuon.getStatusTrackPIDMID(track) == TrackSelectorPID::Status::PIDRejected;
    
      if (MC::isPhysicalPrimary(mcParticles, mcParticle)) {
        histos.fill(HIST("trackingPrm/pt"), track.pt());
        if (!isTOF) {histos.fill(HIST("trackingPrmTOFsel/pt"), track.pt());}
        if (!isRICH) {histos.fill(HIST("trackingPrmRICHsel/pt"), track.pt());}
        if (!isMID) {histos.fill(HIST("trackingPrmMIDsel/pt"), track.pt());}
      } else {
        histos.fill(HIST("trackingSec/pt"), track.pt());
        if (!isTOF) {histos.fill(HIST("trackingSecTOFsel/pt"), track.pt());}
        if (!isRICH) {histos.fill(HIST("trackingSecRICHsel/pt"), track.pt());}
        if (!isMID) {histos.fill(HIST("trackingSecMIDsel/pt"), track.pt());}
      }
      histos.fill(HIST("tracking/pt"), track.pt());
      if (!isTOF) {histos.fill(HIST("trackingTOFsel/pt"), track.pt());}
      if (!isRICH) {histos.fill(HIST("trackingRICHsel/pt"), track.pt());}
      if (!isMID) {histos.fill(HIST("trackingMIDsel/pt"), track.pt());}
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec w; 
  w.push_back(adaptAnalysisTask<Alice3PidIndexBuilder>(cfgc));
  if (cfgc.options().get<int>("rej-el")) {
  w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Electron>>(cfgc, TaskName{"qa-tracking-rejection-electron"}));
  }
  if (cfgc.options().get<int>("rej-mu")) {
  w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Muon>>(cfgc, TaskName{"qa-tracking-rejection-mu"}));
  }
  if (cfgc.options().get<int>("rej-pi")) {
  w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Pion>>(cfgc, TaskName{"qa-tracking-rejection-pion"}));
  }
  return w;
}
