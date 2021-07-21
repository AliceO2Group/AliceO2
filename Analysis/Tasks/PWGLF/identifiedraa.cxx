// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Measurement of the nuclear modification factors R_AA.
/// \author Jan Herdieckerhoff
/// \since June 2021

/// Tasks included for Data:
/// o2-analysis-trackextension, o2-analysis-centrality-table,
/// o2-analysis-multiplicity-table, o2-analysis-timestamp,
/// o2-analysis-event-selection, o2-analysis-pid-tpc-full,
/// o2-analysis-pid-tof-full, o2-analysis-raameasurement

/// Tasks for Mc:
/// o2-analysis-pid-tpc-full,  o2-analysis-trackextension,
/// o2-analysis-pid-tof-full, o2-analysis-raameasurement

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

struct raameasurement {
  TrackSelection globalTracks;
  // Histograms for pions, kaons and protons: TOF, TPC, DCA_xy, DCA_z
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};
  static constexpr std::string_view num_events = "num_events";
  static constexpr std::string_view pt_num[6] = {"pi+/pt_num", "pi-/pt_num", "K+/pt_num", "K-/pt_num", "p+/pt_num", "p-/pt_num"};
  static constexpr std::string_view pt_den[6] = {"pi+/pt_den", "pi-/pt_den", "K+/pt_den", "K-/pt_den", "p+/pt_den", "p-/pt_den"};
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
    globalTracks.SetMaxDcaXYPtDep([](float pt) { return 3.f; });
    // std::vector<double> ptBin = {0.01, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0};
    std::vector<double> ptBin = {0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0};
    AxisSpec axisPt{ptBin, "#it{p}_{T} [GeV/c]"};
    AxisSpec axisNSigma{200, -10., 10.};
    AxisSpec axisDca{1200, -3., 3.};
    AxisSpec axisNum{2, 0.5, 2.5};
    histos.add(num_events.data(), "Number of Events", kTH1D, {axisNum});
    for (int i = 0; i < 6; i++) {
      histos.add(pt_num[i].data(), "Efficiency Numerator", kTH1D, {axisPt});
      histos.add(pt_den[i].data(), "Efficiency Denominator", kTH1D, {axisPt});
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

      if (std::abs(mcParticle.eta()) > 0.8) {
        continue;
      }
      if (!globalTracks.IsSelected(track)) {
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
      float energy = std::sqrt(pow(mass, 2) + pow(track.p(), 2));
      if (std::abs(0.5f * std::log((energy + track.pz()) / (energy - track.pz()))) > 0.5) {
        continue;
      }
      if (!MC::isPhysicalPrimary(mcParticles, mcParticle)) {
        if (mcParticle.pdgCode() == pdg_num[i]) {
          histos.fill(HIST(dca_xy_sec[i]), track.pt(), track.dcaXY());
          histos.fill(HIST(dca_z_sec[i]), track.pt(), track.dcaZ());
        }
        continue;
      }
      if (mcParticle.pdgCode() != pdg_num[i]) {
        if constexpr ((i == 0) || (i == 1)) {
          histos.fill(HIST(tof_MC_full[i]), track.pt(), track.tofNSigmaPi());
        } else if constexpr ((i == 2) || (i == 3)) {
          histos.fill(HIST(tof_MC_full[i]), track.pt(), track.tofNSigmaKa());
        } else if constexpr ((i == 4) || (i == 5)) {
          histos.fill(HIST(tof_MC_full[i]), track.pt(), track.tofNSigmaPr());
        }
        continue;
      }
      histos.fill(HIST(dca_xy_prm[i]), track.pt(), track.dcaXY());
      histos.fill(HIST(dca_z_prm[i]), track.pt(), track.dcaZ());
      histos.fill(HIST(pt_num[i]), mcParticle.pt());
      if constexpr ((i == 0) || (i == 1)) {
        histos.fill(HIST(tof_MC[i]), track.pt(), track.tofNSigmaPi());
      } else if constexpr ((i == 2) || (i == 3)) {
        histos.fill(HIST(tof_MC[i]), track.pt(), track.tofNSigmaKa());
      } else if constexpr ((i == 4) || (i == 5)) {
        histos.fill(HIST(tof_MC[i]), track.pt(), track.tofNSigmaPr());
      }
    }
    for (auto& particle : mcParticles) {
      if (particle.pdgCode() != pdg_num[i]) {
        continue;
      }
      if (!MC::isPhysicalPrimary(mcParticles, particle)) {
        continue;
      }
      if (std::abs(particle.eta()) > 0.8) {
        continue;
      }
      if (std::abs(0.5f * std::log((particle.e() + particle.pz()) / (particle.e() - particle.pz()))) > 0.5) {
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

  template <std::size_t i, typename T>
  void fillHistogramsData(T const& tracks)
  {
    for (auto& track : tracks) {
      if (std::abs(track.eta()) > 0.8) {
        continue;
      }
      // if (!globalTracks.IsSelected(track)) {
      //     continue;
      // }
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
      const float energy = std::sqrt(pow(mass, 2) + pow(track.p(), 2));
      if (std::abs(0.5f * std::log((energy + track.pz()) / (energy - track.pz()))) > 0.5) {
        continue;
      }
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
      histos.fill(HIST(tof[i]), track.pt(), nsigmaTOF);
      histos.fill(HIST(tpc[i]), track.pt(), nsigmaTPC);
      if (std::abs(nsigmaTOF) < 2.f) {
        histos.fill(HIST(dca_xy[i]), track.pt(), track.dcaXY());
        histos.fill(HIST(dca_z[i]), track.pt(), track.dcaZ());
      }
    }
  }

  void processData(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator const& collision,
                   soa::Join<aod::Tracks, aod::TracksExtra,
                             aod::TracksExtended,
                             aod::pidTOFFullEl, aod::pidTOFFullMu,
                             aod::pidTOFFullPi, aod::pidTOFFullKa,
                             aod::pidTOFFullPr, aod::pidTPCFullEl,
                             aod::pidTPCFullMu, aod::pidTPCFullPi,
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
    if (collision.centV0M() > 5.f) {
      return;
    }
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

  void processMCasData(aod::Collision const& collision,
                       soa::Join<aod::Tracks, aod::TracksExtra,
                                 aod::TracksExtended,
                                 aod::pidTOFFullEl, aod::pidTOFFullMu,
                                 aod::pidTOFFullPi, aod::pidTOFFullKa,
                                 aod::pidTOFFullPr, aod::pidTPCFullEl,
                                 aod::pidTPCFullMu, aod::pidTPCFullPi,
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
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{};
  if (cfgc.options().get<int>("MC")) {
    workflow.push_back(adaptAnalysisTask<raameasurement>(cfgc,
                                                         Processes{&raameasurement::processMC,
                                                                   &raameasurement::processMCasData},
                                                         TaskName{"raa-measurement"}));
  } else {
    workflow.push_back(adaptAnalysisTask<raameasurement>(cfgc,
                                                         Processes{&raameasurement::processData},
                                                         TaskName{"raa-measurement"}));
  }
  // workflow.push_back(adaptAnalysisTask<raameasurement>(cfgc,
  //                                                      Processes{&raameasurement::processData}));
  return workflow;
}
