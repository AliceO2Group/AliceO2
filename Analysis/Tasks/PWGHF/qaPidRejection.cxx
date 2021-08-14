// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author Peter Hristov <Peter.Hristov@cern.ch>, CERN
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Henrique J C Zanoli <henrique.zanoli@cern.ch>, Utrecht University
/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

// O2 includes
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
    {"rej-ka", VariantType::Int, 1, {"Efficiency for the Kaon PDG code"}},
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
  Configurable<int> etaBins{"etaBins", 40, "Number of eta bins"};
  Configurable<float> etaMin{"etaMin", -2.f, "Lower limit in eta"};
  Configurable<float> etaMax{"etaMax", 2.f, "Upper limit in eta"};
  Configurable<int> ptBins{"ptBins", 400, "Number of pT bins"};
  Configurable<float> ptMin{"ptMin", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"ptMax", 20.f, "Upper limit in pT"};
  // TPC
  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 9999., "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 999999., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 99999, "Nsigma cut on TPC only"};
  // TOF
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 5., "Upper bound of track pT for TOF PID"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 0., "Nsigma cut on TOF combined with TPC"};
  // RICH
  Configurable<double> d_pidRICHMinpT{"d_pidRICHMinpT", 0.15, "Lower bound of track pT for RICH PID"};
  Configurable<double> d_pidRICHMaxpT{"d_pidRICHMaxpT", 10., "Upper bound of track pT for RICH PID"};
  Configurable<double> d_nSigmaRICH{"d_nSigmaRICH", 3., "Nsigma cut on RICH only"};
  Configurable<double> d_nSigmaRICHCombinedTOF{"d_nSigmaRICHCombinedTOF", 0., "Nsigma cut on RICH combined with TOF"};
  HistogramRegistry histos{"HistogramsRejection"};

  void init(InitContext&)
  {
    AxisSpec ptAxis{ptBins, ptMin, ptMax};
    AxisSpec etaAxis{etaBins, etaMin, etaMax};

    TString commonTitle = "";
    if (particlePDG != 0) {
      commonTitle += Form("PDG %i", particlePDG);
    }

    const TString pt = "#it{p}_{T} [GeV/#it{c}]";
    const TString p = "#it{p} [GeV/#it{c}]";
    const TString eta = "#it{#eta}";
    const TString phi = "#it{#varphi} [rad]";

    histos.add("tracking/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselElectron/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselPion/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselKaon/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselProton/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselElectron/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselPion/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselKaon/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselProton/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingMIDselMuon/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});

    histos.add("tracking/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselElectron/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselPion/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselKaon/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingTOFselProton/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselElectron/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselPion/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselKaon/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingRICHselProton/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("trackingMIDselMuon/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
  }

  using TracksPID = soa::Join<aod::BigTracksPID, aod::HfTrackIndexALICE3PID>;

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>& collisions,
               const o2::soa::Join<TracksPID, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McCollisions& mcCollisions,
               const o2::aod::McParticles& mcParticles, aod::RICHs const&, aod::MIDs const&)
  {
    TrackSelectorPID selectorElectron(kElectron);
    selectorElectron.setRangePtTPC(d_pidTPCMinpT, d_pidTPCMaxpT);
    selectorElectron.setRangeNSigmaTPC(-d_nSigmaTPC, d_nSigmaTPC);
    selectorElectron.setRangePtTOF(d_pidTOFMinpT, d_pidTOFMaxpT);
    selectorElectron.setRangeNSigmaTOF(-d_nSigmaTOF, d_nSigmaTOF);
    selectorElectron.setRangeNSigmaTOFCondTPC(-d_nSigmaTOFCombined, d_nSigmaTOFCombined);
    selectorElectron.setRangePtRICH(d_pidRICHMinpT, d_pidRICHMaxpT);
    selectorElectron.setRangeNSigmaRICH(-d_nSigmaRICH, d_nSigmaRICH);
    selectorElectron.setRangeNSigmaRICHCondTOF(-d_nSigmaRICHCombinedTOF, d_nSigmaRICHCombinedTOF);

    auto selectorPion(selectorElectron);
    selectorPion.setPDG(kPiPlus);

    auto selectorKaon(selectorElectron);
    selectorKaon.setPDG(kKPlus);

    auto selectorProton(selectorElectron);
    selectorProton.setPDG(kProton);

    TrackSelectorPID selectorMuon(kMuonPlus);

    std::vector<int64_t> recoEvt(collisions.size());
    std::vector<int64_t> recoTracks(tracks.size());
    LOGF(info, "%d", particlePDG);
    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if (particlePDG != 0 && mcParticle.pdgCode() != particlePDG) { // Checking PDG code
        continue;
      }
      bool isTOFhpElectron = !(selectorElectron.getStatusTrackPIDTOF(track) == TrackSelectorPID::Status::PIDRejected);
      bool isRICHhpElectron = !(selectorElectron.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isTOFhpPion = !(selectorPion.getStatusTrackPIDTOF(track) == TrackSelectorPID::Status::PIDRejected);
      bool isRICHhpPion = !(selectorPion.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isTOFhpKaon = !(selectorKaon.getStatusTrackPIDTOF(track) == TrackSelectorPID::Status::PIDRejected);
      bool isRICHhpKaon = !(selectorKaon.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isTOFhpProton = !(selectorProton.getStatusTrackPIDTOF(track) == TrackSelectorPID::Status::PIDRejected);
      bool isRICHhpProton = !(selectorProton.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isMIDhpMuon = (selectorMuon.getStatusTrackPIDMID(track) == TrackSelectorPID::Status::PIDAccepted);

      if (MC::isPhysicalPrimary(mcParticle)) {
        histos.fill(HIST("tracking/pteta"), track.pt(), track.eta());
        if (isTOFhpElectron) {
          histos.fill(HIST("trackingTOFselElectron/pteta"), track.pt(), track.eta());
        }
        if (isTOFhpPion) {
          histos.fill(HIST("trackingTOFselPion/pteta"), track.pt(), track.eta());
        }
        if (isTOFhpKaon) {
          histos.fill(HIST("trackingTOFselKaon/pteta"), track.pt(), track.eta());
        }
        if (isTOFhpProton) {
          histos.fill(HIST("trackingTOFselProton/pteta"), track.pt(), track.eta());
        }
        if (isRICHhpElectron) {
          histos.fill(HIST("trackingRICHselElectron/pteta"), track.pt(), track.eta());
        }
        if (isRICHhpPion) {
          histos.fill(HIST("trackingRICHselPion/pteta"), track.pt(), track.eta());
        }
        if (isRICHhpKaon) {
          histos.fill(HIST("trackingRICHselKaon/pteta"), track.pt(), track.eta());
        }
        if (isRICHhpProton) {
          histos.fill(HIST("trackingRICHselProton/pteta"), track.pt(), track.eta());
        }
        if (isMIDhpMuon) {
          histos.fill(HIST("trackingMIDselMuon/pteta"), track.pt(), track.eta());
        }

        histos.fill(HIST("tracking/peta"), track.p(), track.eta());
        if (isTOFhpElectron) {
          histos.fill(HIST("trackingTOFselElectron/peta"), track.p(), track.eta());
        }
        if (isTOFhpPion) {
          histos.fill(HIST("trackingTOFselPion/peta"), track.p(), track.eta());
        }
        if (isTOFhpKaon) {
          histos.fill(HIST("trackingTOFselKaon/peta"), track.p(), track.eta());
        }
        if (isTOFhpProton) {
          histos.fill(HIST("trackingTOFselProton/peta"), track.p(), track.eta());
        }
        if (isRICHhpElectron) {
          histos.fill(HIST("trackingRICHselElectron/peta"), track.p(), track.eta());
        }
        if (isRICHhpPion) {
          histos.fill(HIST("trackingRICHselPion/peta"), track.p(), track.eta());
        }
        if (isRICHhpKaon) {
          histos.fill(HIST("trackingRICHselKaon/peta"), track.p(), track.eta());
        }
        if (isRICHhpProton) {
          histos.fill(HIST("trackingRICHselProton/peta"), track.p(), track.eta());
        }
        if (isMIDhpMuon) {
          histos.fill(HIST("trackingMIDselMuon/peta"), track.p(), track.eta());
        }
      }
    }
  }
};

struct QaRejectionGeneral {
  static constexpr PDG_t PDGs[5] = {kElectron, kMuonMinus, kPiPlus, kKPlus, kProton};
  // Cuts
  Configurable<float> etaMaxSel{"etaMaxSel", 1.44, "Max #eta single track"};
  Configurable<float> ptMinSel{"ptMinSel", 0.6, "p_{T} min single track"};
  // Particle selection
  Configurable<int> etaBins{"etaBins", 40, "Number of eta bins"};
  Configurable<float> etaMin{"etaMin", -2.f, "Lower limit in eta"};
  Configurable<float> etaMax{"etaMax", 2.f, "Upper limit in eta"};
  Configurable<int> ptBins{"ptBins", 400, "Number of pT bins"};
  Configurable<float> ptMin{"ptMin", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"ptMax", 20.f, "Upper limit in pT"};
  // TPC
  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 9999., "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 999999., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 99999, "Nsigma cut on TPC only"};
  // TOF
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 5., "Upper bound of track pT for TOF PID"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 0., "Nsigma cut on TOF combined with TPC"};
  // RICH
  Configurable<double> d_pidRICHMinpT{"d_pidRICHMinpT", 0.15, "Lower bound of track pT for RICH PID"};
  Configurable<double> d_pidRICHMaxpT{"d_pidRICHMaxpT", 10., "Upper bound of track pT for RICH PID"};
  Configurable<double> d_nSigmaRICH{"d_nSigmaRICH", 3., "Nsigma cut on RICH only"};
  Configurable<double> d_nSigmaRICHCombinedTOF{"d_nSigmaRICHCombinedTOF", 0., "Nsigma cut on RICH combined with TOF"};
  HistogramRegistry histos{"HistogramsRejection"};

  void init(InitContext&)
  {
    AxisSpec ptAxis{ptBins, ptMin, ptMax};
    AxisSpec etaAxis{etaBins, etaMin, etaMax};

    TString commonTitle = "";

    const TString pt = "#it{p}_{T} [GeV/#it{c}]";
    const TString p = "#it{p} [GeV/#it{c}]";
    const TString eta = "#it{#eta}";
    const TString phi = "#it{#varphi} [rad]";

    histos.add("hAllNoSel/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hPionNoSel/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hElectronNoSel/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hMuonNoSel/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hKaonNoSel/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});

    histos.add("hAllRICHSelHpElTight/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hPionRICHSelHpElTight/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hElectronRICHSelHpElTight/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hMuonRICHSelHpElTight/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hKaonRICHSelHpElTight/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});

    histos.add("hAllRICHSelHpElTightAlt/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hPionRICHSelHpElTightAlt/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hElectronRICHSelHpElTightAlt/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hMuonRICHSelHpElTightAlt/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hKaonRICHSelHpElTightAlt/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});

    histos.add("hAllRICHSelHpElTightAltDiff/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hPionRICHSelHpElTightAltDiff/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hElectronRICHSelHpElTightAltDiff/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hMuonRICHSelHpElTightAltDiff/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hKaonRICHSelHpElTightAltDiff/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});

    histos.add("hAllRICHSelHpElLoose/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hPionRICHSelHpElLoose/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hElectronRICHSelHpElLoose/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hMuonRICHSelHpElLoose/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hKaonRICHSelHpElLoose/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});

    histos.add("hAllMID/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hElectronMID/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hMuonMID/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hPionMID/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});
    histos.add("hKaonMID/pteta", commonTitle + " Primary;" + pt, kTH2D, {ptAxis, etaAxis});

    histos.add("hAllMID/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("hElectronMID/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("hMuonMID/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("hPionMID/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
    histos.add("hKaonMID/peta", commonTitle + " Primary;" + p, kTH2D, {ptAxis, etaAxis});
  }

  using TracksPID = soa::Join<aod::BigTracksPID, aod::HfTrackIndexALICE3PID>;

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>& collisions,
               const o2::soa::Join<TracksPID, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McCollisions& mcCollisions,
               const o2::aod::McParticles& mcParticles, aod::RICHs const&, aod::MIDs const&)
  {
    TrackSelectorPID selectorElectron(kElectron);
    selectorElectron.setRangePtTPC(d_pidTPCMinpT, d_pidTPCMaxpT);
    selectorElectron.setRangeNSigmaTPC(-d_nSigmaTPC, d_nSigmaTPC);
    selectorElectron.setRangePtTOF(d_pidTOFMinpT, d_pidTOFMaxpT);
    selectorElectron.setRangeNSigmaTOF(-d_nSigmaTOF, d_nSigmaTOF);
    selectorElectron.setRangeNSigmaTOFCondTPC(-d_nSigmaTOFCombined, d_nSigmaTOFCombined);
    selectorElectron.setRangePtRICH(d_pidRICHMinpT, d_pidRICHMaxpT);
    selectorElectron.setRangeNSigmaRICH(-d_nSigmaRICH, d_nSigmaRICH);
    selectorElectron.setRangeNSigmaRICHCondTOF(-d_nSigmaRICHCombinedTOF, d_nSigmaRICHCombinedTOF);

    auto selectorPion(selectorElectron);
    selectorPion.setPDG(kPiPlus);

    auto selectorKaon(selectorElectron);
    selectorKaon.setPDG(kKPlus);

    auto selectorProton(selectorElectron);
    selectorProton.setPDG(kProton);

    TrackSelectorPID selectorMuon(kMuonPlus);

    for (const auto& track : tracks) {

      if (std::abs(track.eta()) > etaMaxSel || track.pt() < ptMinSel) {
        continue;
      }
      const auto mcParticle = track.mcParticle();
      histos.fill(HIST("hAllNoSel/pteta"), track.pt(), track.eta());
      if (mcParticle.pdgCode() == kElectron) {
        histos.fill(HIST("hElectronNoSel/pteta"), track.pt(), track.eta());
      }
      if (mcParticle.pdgCode() == kMuonPlus) {
        histos.fill(HIST("hMuonNoSel/pteta"), track.pt(), track.eta());
      }
      if (mcParticle.pdgCode() == kPiPlus) {
        histos.fill(HIST("hPionNoSel/pteta"), track.pt(), track.eta());
      }
      if (mcParticle.pdgCode() == kKPlus) {
        histos.fill(HIST("hKaonNoSel/pteta"), track.pt(), track.eta());
      }

      bool isRICHhpElectron = !(selectorElectron.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isRICHhpPion = !(selectorPion.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isRICHhpKaon = !(selectorKaon.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isRICHhpProton = !(selectorProton.getStatusTrackPIDRICH(track) == TrackSelectorPID::Status::PIDRejected);
      bool isMIDhpMuon = (selectorMuon.getStatusTrackPIDMID(track) == TrackSelectorPID::Status::PIDAccepted);

      bool isRICHElLoose = isRICHhpElectron;

      bool isRICHElTight = true;
      bool isrichel = true;
      bool istofel = true;
      bool istofpi = false;
      bool isrichpi = false;

      if (track.p() < 0.6) {
        istofel = std::abs(track.tofNSigmaEl()) < d_pidTOFMaxpT;
        istofpi = std::abs(track.tofNSigmaPi()) < d_pidTOFMaxpT;
        isrichel = std::abs(track.rich().richNsigmaEl()) < d_pidRICHMaxpT;
      } else if (track.p() > 0.6 && track.p() < 1.0) {
        isrichel = std::abs(track.rich().richNsigmaEl()) < d_pidRICHMaxpT;
      } else if (track.p() > 1.0 && track.p() < 2.0) {
        isrichel = std::abs(track.rich().richNsigmaEl()) < d_pidRICHMaxpT;
        isrichpi = std::abs(track.rich().richNsigmaPi()) < d_pidRICHMaxpT;
      } else {
        isrichel = std::abs(track.rich().richNsigmaEl()) < d_pidRICHMaxpT;
      }
      isRICHElTight = isrichel && !isrichpi && istofel && !istofpi;

      auto isRICHElTightAlt = selectorElectron.isElectronAndNotPion(track);

      if (isRICHElLoose) {
        histos.fill(HIST("hAllRICHSelHpElLoose/pteta"), track.pt(), track.eta());
        if (mcParticle.pdgCode() == kElectron) {
          histos.fill(HIST("hElectronRICHSelHpElLoose/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kMuonPlus) {
          histos.fill(HIST("hMuonRICHSelHpElLoose/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kPiPlus) {
          histos.fill(HIST("hPionRICHSelHpElLoose/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kKPlus) {
          histos.fill(HIST("hKaonRICHSelHpElLoose/pteta"), track.pt(), track.eta());
        }
      }
      if (isRICHElTight) {
        histos.fill(HIST("hAllRICHSelHpElTight/pteta"), track.pt(), track.eta());
        if (mcParticle.pdgCode() == kElectron) {
          histos.fill(HIST("hElectronRICHSelHpElTight/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kMuonPlus) {
          histos.fill(HIST("hMuonRICHSelHpElTight/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kPiPlus) {
          histos.fill(HIST("hPionRICHSelHpElTight/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kKPlus) {
          histos.fill(HIST("hKaonRICHSelHpElTight/pteta"), track.pt(), track.eta());
        }
      }
      if (isRICHElTightAlt) {
        histos.fill(HIST("hAllRICHSelHpElTightAlt/pteta"), track.pt(), track.eta());
        if (mcParticle.pdgCode() == kElectron) {
          histos.fill(HIST("hElectronRICHSelHpElTightAlt/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kMuonPlus) {
          histos.fill(HIST("hMuonRICHSelHpElTightAlt/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kPiPlus) {
          histos.fill(HIST("hPionRICHSelHpElTightAlt/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kKPlus) {
          histos.fill(HIST("hKaonRICHSelHpElTightAlt/pteta"), track.pt(), track.eta());
        }
      }
      if (isRICHElTightAlt != isRICHElTight) {
        histos.fill(HIST("hAllRICHSelHpElTightAltDiff/pteta"), track.pt(), track.eta());
        if (mcParticle.pdgCode() == kElectron) {
          histos.fill(HIST("hElectronRICHSelHpElTightAltDiff/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kMuonPlus) {
          histos.fill(HIST("hMuonRICHSelHpElTightAltDiff/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kPiPlus) {
          histos.fill(HIST("hPionRICHSelHpElTightAltDiff/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kKPlus) {
          histos.fill(HIST("hKaonRICHSelHpElTightAltDiff/pteta"), track.pt(), track.eta());
        }
      }
      if (isMIDhpMuon) {
        histos.fill(HIST("hAllMID/pteta"), track.pt(), track.eta());
        if (mcParticle.pdgCode() == kElectron) {
          histos.fill(HIST("hElectronMID/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kMuonPlus) {
          histos.fill(HIST("hMuonMID/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kPiPlus) {
          histos.fill(HIST("hPionMID/pteta"), track.pt(), track.eta());
        }
        if (mcParticle.pdgCode() == kKPlus) {
          histos.fill(HIST("hKaonMID/pteta"), track.pt(), track.eta());
        }

        histos.fill(HIST("hAllMID/peta"), track.p(), track.eta());
        if (mcParticle.pdgCode() == kElectron) {
          histos.fill(HIST("hElectronMID/peta"), track.p(), track.eta());
        }
        if (mcParticle.pdgCode() == kMuonPlus) {
          histos.fill(HIST("hMuonMID/peta"), track.p(), track.eta());
        }
        if (mcParticle.pdgCode() == kPiPlus) {
          histos.fill(HIST("hPionMID/peta"), track.p(), track.eta());
        }
        if (mcParticle.pdgCode() == kKPlus) {
          histos.fill(HIST("hKaonMID/peta"), track.p(), track.eta());
        }
      }
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
  if (cfgc.options().get<int>("rej-ka")) {
    w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Kaon>>(cfgc, TaskName{"qa-tracking-rejection-kaon"}));
  }
  if (cfgc.options().get<int>("rej-pr")) {
    w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Proton>>(cfgc, TaskName{"qa-tracking-rejection-proton"}));
  }
  if (cfgc.options().get<int>("rej-mu")) {
    w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Muon>>(cfgc, TaskName{"qa-tracking-rejection-mu"}));
  }
  if (cfgc.options().get<int>("rej-pi")) {
    w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Pion>>(cfgc, TaskName{"qa-tracking-rejection-pion"}));
  }
  w.push_back(adaptAnalysisTask<QaRejectionGeneral>(cfgc));
  return w;
}
