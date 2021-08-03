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
//

/// \brief Measurement of the nuclear modification factors R_AA.
/// \author Jan Herdieckerhoff
/// \since June 2021

/// Tasks included for Data:
/// o2-analysis-trackextension, o2-analysis-centrality-table,
/// o2-analysis-multiplicity-table, o2-analysis-timestamp,
/// o2-analysis-event-selection, o2-analysis-pid-tpc-full,
/// o2-analysis-pid-tof-full, o2-analysis-id-raa

/// Tasks for Mc:
/// o2-analysis-pid-tpc-full,  o2-analysis-trackextension,
/// o2-analysis-pid-tof-full, o2-analysis-id-raa

#include "AnalysisCore/MC.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisCore/TrackSelectionDefaults.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// Switch for the input of monte-carlo data or not
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"MC", VariantType::Int, 0, {"1 for MC, 0 for data"}}};
  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

struct identifiedraaTask {
  TrackSelection globalTrackswoPrim; // Track without cut for primaries
  TrackSelection globalTracks;       // Track with cut for primaries
  // Histograms for pions, kaons and protons: TOF, TPC, DCA_xy, DCA_z
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};
  static constexpr std::string_view num_events = "num_events";
  static constexpr std::string_view pt_num[6] = {"pi+/pt_num", "pi-/pt_num", "K+/pt_num", "K-/pt_num", "p+/pt_num", "p-/pt_num"};
  static constexpr std::string_view pt_den[6] = {"pi+/pt_den", "pi-/pt_den", "K+/pt_den", "K-/pt_den", "p+/pt_den", "p-/pt_den"};
  static constexpr std::string_view matching_num[6] = {"pi+/matching_num", "pi-/matching_num", "K+/matching_num", "K-/matching_num", "p+/matching_num", "p-/matching_num"};
  static constexpr std::string_view wo_track_sel[6] = {"pi+/wo_track_sel", "pi-/wo_track_sel", "K+/wo_track_sel", "K-/wo_track_sel", "p+/wo_track_sel", "p-/wo_track_sel"};
  static constexpr std::string_view w_track_sel[6] = {"pi+/w_track_sel", "pi-/w_track_sel", "K+/w_track_sel", "K-/w_track_sel", "p+/w_track_sel", "p-/w_track_sel"};
  static constexpr std::string_view tof[6] = {"pi+/tof", "pi-/tof", "K+/tof", "K-/tof", "p+/tof", "p-/tof"};
  static constexpr std::string_view tof_MC[6] = {"pi+/tof_MC", "pi-/tof_MC", "K+/tof_MC", "K-/tof_MC", "p+/tof_MC", "p-/tof_MC"};
  static constexpr std::string_view tof_MC_full[6] = {"pi+/tof_MC_full", "pi-/tof_MC_full", "K+/tof_MC_full", "K-/tof_MC_full", "p+/tof_MC_full", "p-/tof_MC_full"};
  static constexpr std::string_view tpc[6] = {"pi+/tpc", "pi-/tpc", "K+/tpc", "K-/tpc", "p+/tpc", "p-/tpc"};
  static constexpr std::string_view dca_xy[6] = {"pi+/dca_xy", "pi-/dca_xy", "K+/dca_xy", "K-/dca_xy", "p+/dca_xy", "p-/dca_xy"};
  static constexpr std::string_view dca_z[6] = {"pi+/dca_z", "pi-/dca_z", "K+/dca_z", "K-/dca_z", "p+/dca_z", "p-/dca_z"};
  static constexpr std::string_view dca_xy_prm[6] = {"pi+/dca_xy_prm", "pi-/dca_xy_prm", "K+/dca_xy_prm", "K-/dca_xy_prm", "p+/dca_xy_prm", "p-/dca_xy_prm"};
  static constexpr std::string_view dca_z_prm[6] = {"pi+/dca_z_prm", "pi-/dca_z_prm", "K+/dca_z_prm", "K-/dca_z_prm", "p+/dca_z_prm", "p-/dca_z_prm"};
  static constexpr std::string_view dca_xy_sec[6] = {"pi+/dca_xy_sec", "pi-/dca_xy_sec", "K+/dca_xy_sec", "K-/dca_xy_sec", "p+/dca_xy_sec", "p-/dca_xy_sec"};
  static constexpr std::string_view dca_z_sec[6] = {"pi+/dca_z_sec", "pi-/dca_z_sec", "K+/dca_z_sec", "K-/dca_z_sec", "p+/dca_z_sec", "p-/dca_z_sec"};
  static constexpr std::array<int, 6> pdg_num = {211, -211, 321, -321, 2212, -2212};

  void init(InitContext&)
  {
    globalTracks = getGlobalTrackSelection();
    globalTrackswoPrim = getGlobalTrackSelection();
    globalTrackswoPrim.SetMaxDcaXYPtDep([](float pt) { return 3.f + pt; });
    globalTrackswoPrim.SetRequireGoldenChi2(false);
    // globalTracks.SetRequireITSRefit(false);
    // globalTracks.SetMaxChi2PerClusterTPC(35.f);
    // globalTracks.SetRequireTPCRefit(false);
    // globalTracks.SetMaxChi2PerClusterITS(60.f);

    std::vector<double> ptBin = {0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0};
    AxisSpec axisPt{ptBin, "#it{p}_{T} [GeV/c]"};
    // AxisSpec axisPt{100, 0, 10, "#it{p}_{T} [GeV/c]"};
    const AxisSpec axisNSigma{200, -10., 10.};
    const AxisSpec axisDca{1200, -3., 3.};
    const AxisSpec axisNum{2, 0.5, 2.5};
    const AxisSpec axisCent{1010, -1, 100, "V0M centrality percentile"};
    histos.add(num_events.data(), "Number of Events", kTH1D, {axisNum});
    // histos.add("MCEvents", "MCEvents", kTH1D, {axisNum});
    histos.add("Centrality", "Centrality", kTH1D, {axisCent});
    histos.add("CentralityAfterEvSel", "CentralityAfterEvSel", kTH1D, {axisCent});
    histos.add("VtxZ", "VtxZ", kTH1D, {{60, -30, 30}});
    histos.add("TrackCut", "TrackCut", kTH1D, {{20, -0.5, 19.5}});
    for (int i = 0; i < 6; i++) {
      histos.add(pt_num[i].data(), "Efficiency Numerator", kTH1D, {axisPt});
      histos.add(pt_den[i].data(), "Efficiency Denominator", kTH1D, {axisPt});
      histos.add(matching_num[i].data(), "Matching Numerator", kTH1D, {axisPt});
      histos.add(wo_track_sel[i].data(), "Without Track Selection", kTH1D, {axisPt});
      histos.add(w_track_sel[i].data(), "With Track Selection", kTH1D, {axisPt});
      histos.add(tof[i].data(), ";#it{p}_{T} (GeV/#it{c});N/#sigma^{TOF}(#pi);Tracks", kTH2D, {axisPt, axisNSigma});
      histos.add(tof_MC[i].data(), "TOF MC", kTH2D, {axisPt, axisNSigma});
      histos.add(tof_MC_full[i].data(), "TOF MC Full", kTH2D, {axisPt, axisNSigma});
      histos.add(tpc[i].data(), "TPC", kTH2D, {axisPt, axisNSigma});
      histos.add(dca_xy[i].data(), "dca_xy", kTH2D, {axisPt, axisDca});
      histos.add(dca_z[i].data(), "dca_z", kTH2D, {axisPt, axisDca});
      histos.add(dca_xy_prm[i].data(), "dca_xy primary", kTH2D, {axisPt, axisDca});
      histos.add(dca_z_prm[i].data(), "dca_z primary", kTH2D, {axisPt, axisDca});
      histos.add(dca_xy_sec[i].data(), "dca_xy secondary", kTH2D, {axisPt, axisDca});
      histos.add(dca_z_sec[i].data(), "dca_z secondary", kTH2D, {axisPt, axisDca});
    }
  }

  template <std::size_t i, typename T1, typename T2>
  void fillHistograms_MC(T1 const& tracks, T2 const& mcParticles)
  {
    for (auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      for (int j = 0; j < 16; j++) {
        if (globalTrackswoPrim.IsSelected(track, static_cast<TrackSelection::TrackCuts>(j))) {
          histos.fill(HIST("TrackCut"), j);
        }
      }
      if (mcParticle.pdgCode() * pdg_num[i] <= 0) {
        continue;
      }
      if (std::abs(mcParticle.eta()) > 0.8) {
        continue;
      }
      float mass;
      if ((i == 0) || (i == 1)) {
        mass = 0.13957; // pion mass
      }
      if ((i == 2) || (i == 3)) {
        mass = 0.49367; // kaon mass
      }
      if ((i == 4) || (i == 5)) {
        mass = 0.93827; // proton mass
      }
      const float energy = std::sqrt(mass * mass + track.p() * track.p());
      if (std::abs(0.5f * std::log((energy + track.pz()) / (energy - track.pz()))) > 0.5) {
        continue;
      }
      if (!globalTrackswoPrim.IsSelected(track)) {
        continue;
      }
      if (!MC::isPhysicalPrimary(mcParticle)) {
        if (mcParticle.pdgCode() == pdg_num[i]) {
          histos.fill(HIST(dca_xy_sec[i]), track.pt(), track.dcaXY());
          histos.fill(HIST(dca_z_sec[i]), track.pt(), track.dcaZ());
        }
        continue;
      }
      if (!globalTracks.IsSelected(track)) {
        continue;
      }
      if (mcParticle.pdgCode() != pdg_num[i]) {
        if (track.hasTOF()) {
          if constexpr ((i == 0) || (i == 1)) {
            histos.fill(HIST(tof_MC_full[i]), track.pt(), track.tofNSigmaPi());
          } else if constexpr ((i == 2) || (i == 3)) {
            histos.fill(HIST(tof_MC_full[i]), track.pt(), track.tofNSigmaKa());
          } else if constexpr ((i == 4) || (i == 5)) {
            histos.fill(HIST(tof_MC_full[i]), track.pt(), track.tofNSigmaPr());
          }
        }
        continue;
      }
      histos.fill(HIST(dca_xy_prm[i]), track.pt(), track.dcaXY());
      histos.fill(HIST(dca_z_prm[i]), track.pt(), track.dcaZ());
      if (track.hasTOF()) {
        histos.fill(HIST(matching_num[i]), mcParticle.pt());
      }
      histos.fill(HIST(pt_num[i]), mcParticle.pt());

      if (track.hasTOF()) {
        if constexpr ((i == 0) || (i == 1)) {
          histos.fill(HIST(tof_MC[i]), track.pt(), track.tofNSigmaPi());
        } else if constexpr ((i == 2) || (i == 3)) {
          histos.fill(HIST(tof_MC[i]), track.pt(), track.tofNSigmaKa());
        } else if constexpr ((i == 4) || (i == 5)) {
          histos.fill(HIST(tof_MC[i]), track.pt(), track.tofNSigmaPr());
        }
      }
    }
    for (auto& particle : mcParticles) {
      //if (std::abs(particle.eta()) > 0.8) {
      //   continue;
      // }
      if (std::abs(0.5f * std::log((particle.e() + particle.pz()) / (particle.e() - particle.pz()))) > 0.5) {
        continue;
      }
      if (!MC::isPhysicalPrimary(particle)) {
        continue;
      }
      if (particle.pdgCode() != pdg_num[i]) {
        continue;
      }

      histos.fill(HIST(pt_den[i]), particle.pt());
    }
  }

  void processMC(soa::Join<aod::Tracks, aod::TracksExtra,
                           aod::TracksExtended, aod::McTrackLabels,
                           aod::pidTOFFullPi, aod::pidTOFFullKa,
                           aod::pidTOFFullPr> const& tracks,
                 const aod::McParticles& mcParticles)
  {
    // LOGF(INFO, "Enter processMC!");
    fillHistograms_MC<0>(tracks, mcParticles);
    fillHistograms_MC<1>(tracks, mcParticles);
    fillHistograms_MC<2>(tracks, mcParticles);
    fillHistograms_MC<3>(tracks, mcParticles);
    fillHistograms_MC<4>(tracks, mcParticles);
    fillHistograms_MC<5>(tracks, mcParticles);
  }

  PROCESS_SWITCH(identifiedraaTask, processMC, "Process simulation events", false);

  template <std::size_t i, typename T>
  void fillHistogramsData(T const& tracks)
  {
    for (auto& track : tracks) {
      if (std::abs(track.eta()) > 0.8) {
        continue;
      }
      if constexpr ((i == 0) || (i == 2) || (i == 4)) {
        if (track.sign() < 0) {
          continue;
        }
      } else if constexpr ((i == 1) || (i == 3) || (i == 5)) {
        if (track.sign() > 0) {
          continue;
        }
      }
      float mass;
      if constexpr ((i == 0) || (i == 1)) {
        mass = 0.13957; // pion mass
      } else if constexpr ((i == 2) || (i == 3)) {
        mass = 0.49367; // kaon mass
      } else if constexpr ((i == 4) || (i == 5)) {
        mass = 0.93827; // proton mass
      }
      const float energy = std::sqrt(mass * mass + track.p() * track.p());
      if (std::abs(0.5f * std::log((energy + track.pz()) / (energy - track.pz()))) > 0.5) {
        continue;
      }

      if (!globalTrackswoPrim.IsSelected(track)) {
        histos.fill(HIST(wo_track_sel[i]), track.pt());
        continue;
      }
      histos.fill(HIST(w_track_sel[i]), track.pt());
      float nsigmaTOF = -999.f;
      float nsigmaTPC = -999.f;
      if constexpr ((i == 0) || (i == 1)) {
        nsigmaTOF = track.tofNSigmaPi();
        nsigmaTPC = track.tpcNSigmaPi();
      } else if constexpr ((i == 2) || (i == 3)) {
        nsigmaTOF = track.tofNSigmaKa();
        nsigmaTPC = track.tpcNSigmaKa();
      } else if constexpr ((i == 4) || (i == 5)) {
        nsigmaTOF = track.tofNSigmaPr();
        nsigmaTPC = track.tpcNSigmaPr();
      }
      if (std::abs(nsigmaTOF) < 2.f) {
        histos.fill(HIST(dca_xy[i]), track.pt(), track.dcaXY());
        histos.fill(HIST(dca_z[i]), track.pt(), track.dcaZ());
      }
      if (!globalTracks.IsSelected(track)) {
        continue;
      }
      if (track.hasTOF()) {
        histos.fill(HIST(tof[i]), track.pt(), nsigmaTOF);
      }
      histos.fill(HIST(tpc[i]), track.pt(), nsigmaTPC);
    }
  }

  void processData(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator const& collision,
                   soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended,
                             aod::pidTOFFullPi, aod::pidTOFFullKa,
                             aod::pidTOFFullPr, aod::pidTPCFullPi,
                             aod::pidTPCFullKa, aod::pidTPCFullPr> const& tracks)
  {
    // LOGF(INFO, "Enter processData!");
    histos.fill(HIST(num_events), 1);
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }
    histos.fill(HIST("Centrality"), collision.centV0M());
    if (collision.centV0M() > 5.f || collision.centV0M() < 0.1f) {
      return;
    }
    histos.fill(HIST("CentralityAfterEvSel"), collision.centV0M());
    histos.fill(HIST("VtxZ"), collision.posZ());
    if (std::abs(collision.posZ()) > 10.f) {
      return;
    }
    histos.fill(HIST(num_events), 2);
    fillHistogramsData<0>(tracks);
    fillHistogramsData<1>(tracks);
    fillHistogramsData<2>(tracks);
    fillHistogramsData<3>(tracks);
    fillHistogramsData<4>(tracks);
    fillHistogramsData<5>(tracks);
  }

  PROCESS_SWITCH(identifiedraaTask, processData, "Process data events", true);

  void processMCasData(aod::Collision const& collision,
                       soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended,
                                 aod::pidTOFFullPi, aod::pidTOFFullKa,
                                 aod::pidTOFFullPr, aod::pidTPCFullPi,
                                 aod::pidTPCFullKa, aod::pidTPCFullPr> const& tracks)
  {
    // LOGF(INFO, "Enter processData!");
    histos.fill(HIST(num_events), 1);
    if (std::abs(collision.posZ()) > 10.f) {
      return;
    }
    histos.fill(HIST(num_events), 2);
    fillHistogramsData<0>(tracks);
    fillHistogramsData<1>(tracks);
    fillHistogramsData<2>(tracks);
    fillHistogramsData<3>(tracks);
    fillHistogramsData<4>(tracks);
    fillHistogramsData<5>(tracks);
  }
  PROCESS_SWITCH(identifiedraaTask, processMCasData, "Process simulation events as data events", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{};
  workflow.push_back(adaptAnalysisTask<identifiedraaTask>(cfgc));
  return workflow;
}
