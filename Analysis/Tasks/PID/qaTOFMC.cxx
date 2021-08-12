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

///
/// \file   qaTOFMC.cxx
/// \author Nicolo' Jacazio
/// \brief  Task to produce QA output of the PID with TOF running on the MC.
///

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
  static constexpr int PDGs[Np] = {11, 13, 211, 321, 2212, 1000010020, 1000010030, 1000020030};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> checkPrimaries{"checkPrimaries", 1,
                                   "Whether to check physical primary and secondaries particles for the resolution."};
  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> minP{"minP", 0.1, "Minimum momentum in range"};
  Configurable<float> maxP{"maxP", 5, "Maximum momentum in range"};
  Configurable<int> nBinsNsigma{"nBinsNsigma", 2000, "Number of bins for the momentum"};
  Configurable<float> minNsigma{"minNsigma", -30.f, "Minimum momentum in range"};
  Configurable<float> maxNsigma{"maxNsigma", 30.f, "Maximum momentum in range"};
  Configurable<float> minEta{"minEta", -0.8, "Minimum eta in range"};
  Configurable<float> maxEta{"maxEta", 0.8, "Maximum eta in range"};
  Configurable<int> nMinNumberOfContributors{"nMinNumberOfContributors", 2, "Minimum required number of contributors to the vertex"};
  Configurable<int> logPt{"log-pt", 1, "Flag to use a logarithmic pT axis, in this case the pT limits are the expontents"};

  template <uint8_t i>
  void addParticleHistos()
  {

    AxisSpec ptAxis{nBinsP, minP, maxP, "#it{p}_{T} (GeV/#it{c})"};
    const AxisSpec nSigmaAxis{nBinsNsigma, minNsigma, maxNsigma, Form("N_{#sigma}^{TOF}(%s)", pT[pid_type])};
    if (logPt) {
      ptAxis.makeLogaritmic();
    }

    // NSigma
    histos.add(hnsigmaMC[i].data(), Form("True %s", pT[i]), HistType::kTH2F, {ptAxis, nSigmaAxis});
    if (!checkPrimaries) {
      return;
    }
    histos.add(hnsigmaMCprm[i].data(), Form("True Primary %s", pT[i]), HistType::kTH2F, {ptAxis, nSigmaAxis});
    histos.add(hnsigmaMCsec[i].data(), Form("True Secondary %s", pT[i]), HistType::kTH2F, {ptAxis, nSigmaAxis});
  }

  void init(o2::framework::InitContext&)
  {
    AxisSpec ptAxis{nBinsP, minP, maxP, "#it{p}_{T} (GeV/#it{c})"};
    AxisSpec pAxis{nBinsP, minP, maxP, "#it{p} (GeV/#it{c})"};
    if (logPt) {
      ptAxis.makeLogaritmic();
      pAxis.makeLogaritmic();
    }
    const AxisSpec nSigmaAxis{nBinsNsigma, minNsigma, maxNsigma, Form("N_{#sigma}^{TOF}(%s)", pT[pid_type])};
    const AxisSpec betaAxis{1000, 0, 1.2, "TOF #beta"};

    histos.add("event/T0", ";Tracks with TOF;T0 (ps);Counts", HistType::kTH2F, {{1000, 0, 1000}, {1000, -1000, 1000}});
    histos.add(hnsigma[pid_type].data(), pT[pid_type], HistType::kTH2F, {ptAxis, nSigmaAxis});
    if (checkPrimaries) {
      histos.add(hnsigmaprm[pid_type].data(), Form("Primary %s", pT[pid_type]), HistType::kTH2F, {ptAxis, nSigmaAxis});
      histos.add(hnsigmasec[pid_type].data(), Form("Secondary %s", pT[pid_type]), HistType::kTH2F, {ptAxis, nSigmaAxis});
    }
    addParticleHistos<0>();
    addParticleHistos<1>();
    addParticleHistos<2>();
    addParticleHistos<3>();
    addParticleHistos<4>();
    addParticleHistos<5>();
    addParticleHistos<6>();
    addParticleHistos<7>();
    addParticleHistos<8>();
    histos.add("event/tofbeta", "All", HistType::kTH2F, {pAxis, betaAxis});
    histos.add("event/tofbetaMC", pT[pid_type], HistType::kTH2F, {pAxis, betaAxis});
    if (checkPrimaries) {
      histos.add("event/tofbetaPrm", "Primaries", HistType::kTH2F, {pAxis, betaAxis});
      histos.add("event/tofbetaSec", "Secondaries", HistType::kTH2F, {pAxis, betaAxis});
      histos.add("event/tofbetaMCPrm", Form("Primary %s", pT[pid_type]), HistType::kTH2F, {pAxis, betaAxis});
      histos.add("event/tofbetaMCSec", Form("Secondary %s", pT[pid_type]), HistType::kTH2F, {pAxis, betaAxis});
    }
  }

  template <uint8_t pidIndex, typename T, typename TT>
  void fillNsigma(const T& track, const TT& mcParticles, const float& nsigma)
  {
    const auto particle = track.mcParticle();
    if (abs(particle.pdgCode()) == PDGs[pidIndex]) {

      histos.fill(HIST(hnsigmaMC[pidIndex]), track.pt(), nsigma);
      // Selecting primaries
      if (MC::isPhysicalPrimary(particle)) {
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
      if (!t.hasTOF()) { // Skipping tracks without TOF
        continue;
      }
      if (t.eta() < minEta || t.eta() > maxEta) {
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
      const auto particle = t.mcParticle();
      if (MC::isPhysicalPrimary(particle)) { // Selecting primaries
        histos.fill(HIST(hnsigmaprm[pid_type]), t.pt(), nsigma);
        histos.fill(HIST("event/tofbetaPrm"), t.p(), t.beta());
      } else {
        histos.fill(HIST(hnsigmasec[pid_type]), t.pt(), nsigma);
        histos.fill(HIST("event/tofbetaSec"), t.p(), t.beta());
      }
      if (abs(particle.pdgCode()) == PDGs[pid_type]) { // Checking the PDG code
        histos.fill(HIST("event/tofbetaMC"), t.pt(), t.beta());
        if (MC::isPhysicalPrimary(particle)) {
          histos.fill(HIST("event/tofbetaMCPrm"), t.pt(), t.beta());
        } else {
          histos.fill(HIST("event/tofbetaMCSec"), t.pt(), t.beta());
        }
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
