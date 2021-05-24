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
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/Track.h"
#include <CCDB/BasicCCDBManager.h>
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTOF.h"
#include "AnalysisCore/MC.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"qa-el", VariantType::Int, 0, {"Produce PID information for the electron mass hypothesis"}},
    {"qa-mu", VariantType::Int, 0, {"Produce PID information for the muon mass hypothesis"}},
    {"qa-pikapr", VariantType::Int, 1, {"Produce PID information for the Pion, Kaon, Proton mass hypothesis"}},
    {"qa-nuclei", VariantType::Int, 0, {"Produce PID information for the Deuteron, Triton, Alpha mass hypothesis"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

template <o2::track::PID::ID pid_type>
struct pidTOFTaskQA {

  static constexpr int Np = 9;
  static constexpr std::string_view hnsigma[Np] = {"nsigma/El", "nsigma/Mu", "nsigma/Pi",
                                                   "nsigma/Ka", "nsigma/Pr", "nsigma/De",
                                                   "nsigma/Tr", "nsigma/He", "nsigma/Al"};
  static constexpr std::string_view hnsigmaprm[Np] = {"nsigmaprm/El", "nsigmaprm/Mu", "nsigmaprm/Pi",
                                                      "nsigmaprm/Ka", "nsigmaprm/Pr", "nsigmaprm/De",
                                                      "nsigmaprm/Tr", "nsigmaprm/He", "nsigmaprm/Al"};
  static constexpr std::string_view hnsigmasec[Np] = {"nsigmasec/El", "nsigmasec/Mu", "nsigmasec/Pi",
                                                      "nsigmasec/Ka", "nsigmasec/Pr", "nsigmasec/De",
                                                      "nsigmasec/Tr", "nsigmasec/He", "nsigmasec/Al"};
  static constexpr std::string_view hnsigmaMC[Np] = {"nsigmaMC/El", "nsigmaMC/Mu", "nsigmaMC/Pi",
                                                     "nsigmaMC/Ka", "nsigmaMC/Pr", "nsigmaMC/De",
                                                     "nsigmaMC/Tr", "nsigmaMC/He", "nsigmaMC/Al"};
  static constexpr std::string_view hnsigmaMCsec[Np] = {"nsigmaMCsec/El", "nsigmaMCsec/Mu", "nsigmaMCsec/Pi",
                                                        "nsigmaMCsec/Ka", "nsigmaMCsec/Pr", "nsigmaMCsec/De",
                                                        "nsigmaMCsec/Tr", "nsigmaMCsec/He", "nsigmaMCsec/Al"};
  static constexpr std::string_view hnsigmaMCprm[Np] = {"nsigmaMCprm/El", "nsigmaMCprm/Mu", "nsigmaMCprm/Pi",
                                                        "nsigmaMCprm/Ka", "nsigmaMCprm/Pr", "nsigmaMCprm/De",
                                                        "nsigmaMCprm/Tr", "nsigmaMCprm/He", "nsigmaMCprm/Al"};
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr int PDGs[Np] = {11, 13, 211, 321, 2212, 1, 1, 1, 1};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0.1, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 5, "Maximum momentum in range"};
  Configurable<float> MinEta{"MinEta", -0.8, "Minimum eta in range"};
  Configurable<float> MaxEta{"MaxEta", 0.8, "Maximum eta in range"};
  Configurable<int> nMinNumberOfContributors{"nMinNumberOfContributors", 2, "Minimum required number of contributors to the vertex"};

  template <typename T>
  void makelogaxis(T h)
  {
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

  template <uint8_t i>
  void addParticleHistos()
  {
    // NSigma
    histos.add(hnsigmaMC[i].data(), Form("True %s;#it{p}_{T} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[i], pT[pid_type]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaMC[i])));
    histos.add(hnsigmaMCprm[i].data(), Form("True Primary %s;#it{p}_{T} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[i], pT[pid_type]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaMCprm[i])));
    histos.add(hnsigmaMCsec[i].data(), Form("True Secondary %s;#it{p}_{T} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[i], pT[pid_type]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaMCsec[i])));
  }

  void init(o2::framework::InitContext&)
  {
    histos.add("event/T0", ";Tracks with TOF;T0 (ps);Counts", HistType::kTH2F, {{1000, 0, 1000}, {1000, -1000, 1000}});
    histos.add(hnsigma[pid_type].data(), Form(";#it{p}_{T} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[pid_type]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigma[pid_type])));
    histos.add(hnsigmaprm[pid_type].data(), Form("Primary;#it{p}_{T} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[pid_type]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaprm[pid_type])));
    histos.add(hnsigmasec[pid_type].data(), Form("Secondary;#it{p}_{T} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[pid_type]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigmasec[pid_type])));
    addParticleHistos<0>();
    addParticleHistos<1>();
    addParticleHistos<2>();
    addParticleHistos<3>();
    addParticleHistos<4>();
    addParticleHistos<5>();
    addParticleHistos<6>();
    addParticleHistos<7>();
    addParticleHistos<8>();
    histos.add("event/tofbeta", ";#it{p}_{T} (GeV/#it{c});TOF #beta", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 1.2}});
    makelogaxis(histos.get<TH2>(HIST("event/tofbeta")));
    histos.add("event/tofbetaPrm", ";#it{p}_{T} (GeV/#it{c});TOF #beta", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 1.2}});
    makelogaxis(histos.get<TH2>(HIST("event/tofbetaPrm")));
    histos.add("event/tofbetaSec", ";#it{p}_{T} (GeV/#it{c});TOF #beta", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 1.2}});
    makelogaxis(histos.get<TH2>(HIST("event/tofbetaSec")));
  }

  template <uint8_t pidIndex, typename T, typename TT>
  void fillNsigma(const T& track, const TT& mcParticles, const float& nsigma)
  {
    if (abs(track.mcParticle().pdgCode()) == PDGs[pidIndex]) {
      histos.fill(HIST(hnsigmaMC[pidIndex]), track.pt(), nsigma);

      if (MC::isPhysicalPrimary(mcParticles, track.mcParticle())) { // Selecting primaries
        histos.fill(HIST(hnsigmaMCprm[pidIndex]), track.pt(), nsigma);
      } else {
        histos.fill(HIST(hnsigmaMCsec[pidIndex]), track.pt(), nsigma);
      }
    }
  }

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksExtra,
                         aod::pidTOFFullEl, aod::pidTOFFullMu, aod::pidTOFFullPi,
                         aod::pidTOFFullKa, aod::pidTOFFullPr, aod::pidTOFFullDe,
                         aod::pidTOFFullTr, aod::pidTOFFullHe, aod::pidTOFFullAl,
                         aod::McTrackLabels, aod::pidTOFbeta> const& tracks,
               aod::McParticles& mcParticles)
  {
    if (collision.numContrib() < nMinNumberOfContributors) {
      return;
    }
    const float collisionTime_ps = collision.collisionTime() * 1000.f;
    unsigned int nTracksWithTOF = 0;
    for (auto t : tracks) {
      //
      if (t.tofSignal() < 0) { // Skipping tracks without TOF
        continue;
      }
      if (t.eta() < MinEta || t.eta() > MaxEta) {
        continue;
      }
      nTracksWithTOF++;
      float nsigma = -999.f;
      if constexpr (pid_type == 0) {
        nsigma = t.tofNSigmaEl();
      } else if constexpr (pid_type == 1) {
        nsigma = t.tofNSigmaMu();
      } else if constexpr (pid_type == 2) {
        nsigma = t.tofNSigmaPi();
      } else if constexpr (pid_type == 3) {
        nsigma = t.tofNSigmaKa();
      } else if constexpr (pid_type == 4) {
        nsigma = t.tofNSigmaPr();
      } else if constexpr (pid_type == 5) {
        nsigma = t.tofNSigmaDe();
      } else if constexpr (pid_type == 6) {
        nsigma = t.tofNSigmaTr();
      } else if constexpr (pid_type == 7) {
        nsigma = t.tofNSigmaHe();
      } else if constexpr (pid_type == 8) {
        nsigma = t.tofNSigmaAl();
      }

      // Fill for all
      histos.fill(HIST(hnsigma[pid_type]), t.pt(), nsigma);
      histos.fill(HIST("event/tofbeta"), t.p(), t.beta());
      if (MC::isPhysicalPrimary(mcParticles, t.mcParticle())) { // Selecting primaries
        histos.fill(HIST(hnsigmaprm[pid_type]), t.pt(), nsigma);
        histos.fill(HIST("event/tofbetaPrm"), t.p(), t.beta());
      } else {
        histos.fill(HIST(hnsigmasec[pid_type]), t.pt(), nsigma);
        histos.fill(HIST("event/tofbetaSec"), t.p(), t.beta());
      }
      // Fill with PDG codes
      fillNsigma<0>(t, mcParticles, nsigma);
      fillNsigma<1>(t, mcParticles, nsigma);
      fillNsigma<2>(t, mcParticles, nsigma);
      fillNsigma<3>(t, mcParticles, nsigma);
      fillNsigma<4>(t, mcParticles, nsigma);
      fillNsigma<5>(t, mcParticles, nsigma);
      fillNsigma<6>(t, mcParticles, nsigma);
      fillNsigma<7>(t, mcParticles, nsigma);
      fillNsigma<8>(t, mcParticles, nsigma);
    }
    histos.fill(HIST("event/T0"), nTracksWithTOF, collisionTime_ps);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{};
  if (cfgc.options().get<int>("qa-el")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Electron>>(cfgc, TaskName{"pidTOF-qa-El"}));
  }
  if (cfgc.options().get<int>("qa-mu")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Muon>>(cfgc, TaskName{"pidTOF-qa-Mu"}));
  }
  if (cfgc.options().get<int>("qa-pikapr")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Pion>>(cfgc, TaskName{"pidTOF-qa-Pi"}));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Kaon>>(cfgc, TaskName{"pidTOF-qa-Ka"}));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Proton>>(cfgc, TaskName{"pidTOF-qa-Pr"}));
  }
  if (cfgc.options().get<int>("qa-nuclei")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Deuteron>>(cfgc, TaskName{"pidTOF-qa-De"}));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Triton>>(cfgc, TaskName{"pidTOF-qa-Tr"}));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Helium3>>(cfgc, TaskName{"pidTOF-qa-He"}));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Alpha>>(cfgc, TaskName{"pidTOF-qa-Al"}));
  }
  return workflow;
}
