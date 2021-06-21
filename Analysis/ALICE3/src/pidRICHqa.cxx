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
#include "Framework/AnalysisTask.h"
#include "ALICE3Analysis/RICH.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "ReconstructionDataFormats/PID.h"

using namespace o2;
using namespace o2::track;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"qa-el", VariantType::Int, 1, {"Produce PID information for the electron mass hypothesis"}},
    {"qa-mu", VariantType::Int, 1, {"Produce PID information for the muon mass hypothesis"}},
    {"qa-pikapr", VariantType::Int, 1, {"Produce PID information for the Pion, Kaon, Proton mass hypothesis"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

namespace o2::aod
{

namespace indices
{
DECLARE_SOA_INDEX_COLUMN(Track, track);
DECLARE_SOA_INDEX_COLUMN(RICH, rich);
} // namespace indices

DECLARE_SOA_INDEX_TABLE_USER(RICHTracksIndex, Tracks, "RICHTRK", indices::TrackId, indices::RICHId);
} // namespace o2::aod

struct richIndexBuilder {
  Builds<o2::aod::RICHTracksIndex> ind;
  void init(o2::framework::InitContext&)
  {
  }
};

template <o2::track::PID::ID pid_type>
struct richPidQaMc {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};
  Configurable<int> pdgCode{"pdgCode", 0, "pdg code of the particles to accept"};
  Configurable<int> useOnlyPhysicsPrimary{"useOnlyPhysicsPrimary", 1,
                                          "Whether to use only physical primary particles."};
  Configurable<int> useTOF{"useTOF", 0,
                           "Whether to use the TOF information"};
  Configurable<float> minLength{"minLength", 0, "Minimum length of accepted tracks (cm)"};
  Configurable<float> maxLength{"maxLength", 1000, "Maximum length of accepted tracks (cm)"};
  Configurable<float> minEta{"minEta", -1.4, "Minimum eta of accepted tracks"};
  Configurable<float> maxEta{"maxEta", 1.4, "Maximum eta of accepted tracks"};
  Configurable<int> nBinsP{"nBinsP", 500, "Number of momentum bins"};
  Configurable<float> minP{"minP", 0.01, "Minimum momentum plotted (GeV/c)"};
  Configurable<float> maxP{"maxP", 100, "Maximum momentum plotted (GeV/c)"};
  Configurable<int> nBinsNsigma{"nBinsNsigma", 600, "Number of Nsigma bins"};
  Configurable<float> minNsigma{"minNsigma", -100.f, "Minimum Nsigma plotted"};
  Configurable<float> maxNsigma{"maxNsigma", 100.f, "Maximum Nsigma plotted"};
  Configurable<int> nBinsDelta{"nBinsDelta", 600, "Number of delta bins"};
  Configurable<float> minDelta{"minDelta", -0.4f, "Minimum delta plotted (rad)"};
  Configurable<float> maxDelta{"maxDelta", 0.4f, "Maximum delta plotted (rad)"};
  Configurable<int> logAxis{"logAxis", 1, "Flag to use a log momentum axis"};

  template <typename T>
  void makelogaxis(T h)
  {
    if (logAxis == 0) {
      return;
    }
    const int nbins = h->GetNbinsX();
    double binp[nbins + 1];
    double max = h->GetXaxis()->GetBinUpEdge(nbins);
    double min = h->GetXaxis()->GetBinLowEdge(1);
    if (min <= 0) {
      min = 0.00001;
    }
    double lmin = TMath::Log10(min);
    double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins);
    for (int i = 0; i < nbins; i++) {
      binp[i] = TMath::Exp(TMath::Log(10) * (lmin + i * ldelta));
    }
    binp[nbins] = max + 1;
    h->GetXaxis()->Set(nbins, binp);
  }
  static constexpr int Np = 5;
  static constexpr std::string_view hdelta[Np] = {"delta/El", "delta/Mu", "delta/Pi", "delta/Ka", "delta/Pr"};
  static constexpr std::string_view hnsigma[Np] = {"nsigma/El", "nsigma/Mu", "nsigma/Pi", "nsigma/Ka", "nsigma/Pr"};
  static constexpr std::string_view hnsigmaprm[Np] = {"nsigmaprm/El", "nsigmaprm/Mu", "nsigmaprm/Pi", "nsigmaprm/Ka", "nsigmaprm/Pr"};
  static constexpr std::string_view hnsigmasec[Np] = {"nsigmasec/El", "nsigmasec/Mu", "nsigmasec/Pi", "nsigmasec/Ka", "nsigmasec/Pr"};
  static constexpr std::string_view hnsigmaMC[Np] = {"nsigmaMC/El", "nsigmaMC/Mu", "nsigmaMC/Pi", "nsigmaMC/Ka", "nsigmaMC/Pr"};
  static constexpr std::string_view hnsigmaMCsec[Np] = {"nsigmaMCsec/El", "nsigmaMCsec/Mu", "nsigmaMCsec/Pi", "nsigmaMCsec/Ka", "nsigmaMCsec/Pr"};
  static constexpr std::string_view hnsigmaMCprm[Np] = {"nsigmaMCprm/El", "nsigmaMCprm/Mu", "nsigmaMCprm/Pi", "nsigmaMCprm/Ka", "nsigmaMCprm/Pr"};

  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p"};
  static constexpr int PDGs[Np] = {11, 13, 211, 321, 2212};
  template <uint8_t i>
  void addParticleHistos()
  {
    AxisSpec momAxis{nBinsP, minP, maxP};
    AxisSpec nsigmaAxis{nBinsNsigma, minNsigma, maxNsigma};

    const char* ns = Form("N_{#sigma}^{RICH}(%s)", pT[pid_type]);
    const char* pt = "#it{p}_{T} (GeV/#it{c})";
    const char* tit = Form("%s", pT[i]);
    if (useTOF) {
      tit = Form("TOF Selected %s", pT[i]);
    }
    // NSigma
    histos.add(hnsigmaMC[i].data(), Form("True %s;%s;%s", tit, pt, ns), HistType::kTH2F, {momAxis, nsigmaAxis});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaMC[i])));
    histos.add(hnsigmaMCprm[i].data(), Form("True Primary %s;%s;%s", tit, pt, ns), HistType::kTH2F, {momAxis, nsigmaAxis});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaMCprm[i])));
    histos.add(hnsigmaMCsec[i].data(), Form("True Secondary %s;%s;%s", tit, pt, ns), HistType::kTH2F, {momAxis, nsigmaAxis});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaMCsec[i])));
  }
  void init(o2::framework::InitContext&)
  {
    AxisSpec momAxis{nBinsP, minP, maxP};
    AxisSpec nsigmaAxis{nBinsNsigma, minNsigma, maxNsigma};
    AxisSpec deltaAxis{nBinsDelta, minDelta, maxDelta};

    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", kTH1F, {{100, -20, 20}});
    histos.add("p/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("p/Prim", "Primaries;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("p/Sec", "Secondaries;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("pt/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("qa/signal", ";Cherenkov angle (rad)", kTH1F, {{100, 0, 1}});
    histos.add("qa/eta", ";#it{#eta}", kTH1F, {{100, -2, 2}});
    histos.add("qa/signalerror", ";Cherenkov angle (rad)", kTH1F, {{100, 0, 1}});
    histos.add("qa/signalvsP", ";#it{p} (GeV/#it{c});Cherenkov angle (rad)", kTH2F, {momAxis, {1000, 0, 0.3}});
    makelogaxis(histos.get<TH2>(HIST("qa/signalvsP")));
    histos.add("qa/signalvsPPrim", ";#it{p} (GeV/#it{c});Cherenkov angle (rad)", kTH2F, {momAxis, {1000, 0, 0.3}});
    makelogaxis(histos.get<TH2>(HIST("qa/signalvsPPrim")));
    histos.add("qa/signalvsPSec", ";#it{p} (GeV/#it{c});Cherenkov angle (rad)", kTH2F, {momAxis, {1000, 0, 0.3}});
    makelogaxis(histos.get<TH2>(HIST("qa/signalvsPSec")));
    histos.add(hdelta[pid_type].data(), Form(";#it{p} (GeV/#it{c});#Delta(%s) (rad)", pT[pid_type]), kTH2F, {momAxis, deltaAxis});
    makelogaxis(histos.get<TH2>(HIST(hdelta[pid_type])));
    histos.add(hnsigma[pid_type].data(), Form(";#it{p}_{T} (GeV/#it{c});N_{#sigma}^{RICH}(%s)", pT[pid_type]), HistType::kTH2F, {momAxis, nsigmaAxis});
    makelogaxis(histos.get<TH2>(HIST(hnsigma[pid_type])));
    histos.add(hnsigmaprm[pid_type].data(), Form("Primary;#it{p}_{T} (GeV/#it{c});N_{#sigma}^{RICH}(%s)", pT[pid_type]), HistType::kTH2F, {momAxis, nsigmaAxis});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaprm[pid_type])));
    histos.add(hnsigmasec[pid_type].data(), Form("Secondary;#it{p}_{T} (GeV/#it{c});N_{#sigma}^{RICH}(%s)", pT[pid_type]), HistType::kTH2F, {momAxis, nsigmaAxis});
    makelogaxis(histos.get<TH2>(HIST(hnsigmasec[pid_type])));
    addParticleHistos<0>();
    addParticleHistos<1>();
    addParticleHistos<2>();
    addParticleHistos<3>();
    addParticleHistos<4>();
  }

  template <uint8_t pidIndex, typename T, typename TTT, typename TT>
  void fillNsigma(const T& track, const TTT& particle, const TT& mcParticles, const float& nsigma)
  {
    if (abs(particle.pdgCode()) == PDGs[pidIndex]) {
      histos.fill(HIST(hnsigmaMC[pidIndex]), track.pt(), nsigma);

      if (MC::isPhysicalPrimary(mcParticles, particle)) { // Selecting primaries
        histos.fill(HIST(hnsigmaMCprm[pidIndex]), track.pt(), nsigma);
      } else {
        histos.fill(HIST(hnsigmaMCsec[pidIndex]), track.pt(), nsigma);
      }
    }
  }

  using Trks = soa::Join<aod::Tracks, aod::RICHTracksIndex, aod::TracksExtra,
                         aod::pidTOFFullEl, aod::pidTOFFullMu, aod::pidTOFFullPi,
                         aod::pidTOFFullKa, aod::pidTOFFullPr>;
  void process(const Trks& tracks,
               const aod::McTrackLabels& labels,
               const aod::RICHs&,
               const aod::McParticles& mcParticles,
               const aod::Collisions& colls)
  {
    for (const auto& col : colls) {
      histos.fill(HIST("event/vertexz"), col.posZ());
    }
    for (const auto& track : tracks) {
      if (!track.has_rich()) {
        continue;
      }
      if (track.length() < minLength) {
        continue;
      }
      if (track.length() > maxLength) {
        continue;
      }
      if (track.eta() > maxEta || track.eta() < minEta) {
        continue;
      }
      const auto mcParticle = labels.iteratorAt(track.globalIndex()).mcParticle();
      if (pdgCode != 0 && abs(mcParticle.pdgCode()) != pdgCode) {
        continue;
      }
      if (useTOF && !track.hasTOF()) {
        continue;
      }

      histos.fill(HIST("p/Unselected"), track.p());
      histos.fill(HIST("pt/Unselected"), track.pt());
      histos.fill(HIST("qa/eta"), track.eta());
      histos.fill(HIST("qa/signal"), track.rich().richSignal());
      histos.fill(HIST("qa/signalerror"), track.rich().richSignalError());
      histos.fill(HIST("qa/signalvsP"), track.p(), track.rich().richSignal());

      float delta = -999.f;
      float nsigma = -999.f;
      if constexpr (pid_type == 0) {
        delta = track.rich().richDeltaEl();
        nsigma = track.rich().richNsigmaEl();
        if (useTOF && abs(track.tofNSigmaEl()) > 3.f) {
          continue;
        }
      } else if constexpr (pid_type == 1) {
        delta = track.rich().richDeltaMu();
        nsigma = track.rich().richNsigmaMu();
        if (useTOF && abs(track.tofNSigmaMu()) > 3.f) {
          continue;
        }
      } else if constexpr (pid_type == 2) {
        delta = track.rich().richDeltaPi();
        nsigma = track.rich().richNsigmaPi();
        if (useTOF && abs(track.tofNSigmaPi()) > 3.f) {
          continue;
        }
      } else if constexpr (pid_type == 3) {
        delta = track.rich().richDeltaKa();
        nsigma = track.rich().richNsigmaKa();
        if (useTOF && abs(track.tofNSigmaKa()) > 3.f) {
          continue;
        }
      } else if constexpr (pid_type == 4) {
        delta = track.rich().richDeltaPr();
        nsigma = track.rich().richNsigmaPr();
        if (useTOF && abs(track.tofNSigmaPr()) > 3.f) {
          continue;
        }
      }
      histos.fill(HIST(hnsigma[pid_type]), track.pt(), nsigma);
      histos.fill(HIST(hdelta[pid_type]), track.p(), delta);
      if (MC::isPhysicalPrimary(mcParticles, mcParticle)) { // Selecting primaries
        histos.fill(HIST(hnsigmaprm[pid_type]), track.pt(), nsigma);
        histos.fill(HIST("p/Prim"), track.p());
      } else {
        histos.fill(HIST(hnsigmasec[pid_type]), track.pt(), nsigma);
        histos.fill(HIST("p/Sec"), track.p());
      }

      fillNsigma<0>(track, mcParticle, mcParticles, nsigma);
      fillNsigma<1>(track, mcParticle, mcParticles, nsigma);
      fillNsigma<2>(track, mcParticle, mcParticles, nsigma);
      fillNsigma<3>(track, mcParticle, mcParticles, nsigma);
      fillNsigma<4>(track, mcParticle, mcParticles, nsigma);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  auto workflow = WorkflowSpec{adaptAnalysisTask<richIndexBuilder>(cfg)};
  if (cfg.options().get<int>("qa-el")) {
    workflow.push_back(adaptAnalysisTask<richPidQaMc<PID::Electron>>(cfg, TaskName{"pidRICH-qa-El"}));
  }
  if (cfg.options().get<int>("qa-mu")) {
    workflow.push_back(adaptAnalysisTask<richPidQaMc<PID::Muon>>(cfg, TaskName{"pidRICH-qa-Mu"}));
  }
  if (cfg.options().get<int>("qa-pikapr")) {
    workflow.push_back(adaptAnalysisTask<richPidQaMc<PID::Pion>>(cfg, TaskName{"pidRICH-qa-Pi"}));
    workflow.push_back(adaptAnalysisTask<richPidQaMc<PID::Kaon>>(cfg, TaskName{"pidRICH-qa-Ka"}));
    workflow.push_back(adaptAnalysisTask<richPidQaMc<PID::Proton>>(cfg, TaskName{"pidRICH-qa-Pr"}));
  }
  return workflow;
}
