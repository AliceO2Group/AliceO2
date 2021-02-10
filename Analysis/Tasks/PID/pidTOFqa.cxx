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
    {"pid-el", VariantType::Int, 1, {"Produce PID information for the electron mass hypothesis"}},
    {"pid-mu", VariantType::Int, 1, {"Produce PID information for the muon mass hypothesis"}},
    {"pid-pikapr", VariantType::Int, 1, {"Produce PID information for the Pion, Kaon, Proton mass hypothesis"}},
    {"pid-nuclei", VariantType::Int, 1, {"Produce PID information for the Deuteron, Triton, Alpha mass hypothesis"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

template <o2::track::PID::ID pid_type>
struct pidTOFTaskQA {

  static constexpr int Np = 9;
  static constexpr std::string_view hnsigma[Np] = {"nsigma/El", "nsigma/Mu", "nsigma/Pi",
                                                   "nsigma/Ka", "nsigma/Pr", "nsigma/De",
                                                   "nsigma/Tr", "nsigma/He", "nsigma/Al"};
  static constexpr std::string_view hnsigmaMC[Np] = {"nsigmaMC/El", "nsigmaMC/Mu", "nsigmaMC/Pi",
                                                     "nsigmaMC/Ka", "nsigmaMC/Pr", "nsigmaMC/De",
                                                     "nsigmaMC/Tr", "nsigmaMC/He", "nsigmaMC/Al"};
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr int PDGs[Np] = {11, 13, 211, 321, 2212, 1, 1, 1, 1};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0.1, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 5, "Maximum momentum in range"};

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
    histos.add(hnsigmaMC[i].data(), Form(";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigmaMC[i])));
  }

  void init(o2::framework::InitContext&)
  {
    histos.add(hnsigma[pid_type].data(), Form(";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[pid_type]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {2000, -30, 30}});
    makelogaxis(histos.get<TH2>(HIST(hnsigma[pid_type])));
    addParticleHistos<0>();
    addParticleHistos<1>();
    addParticleHistos<2>();
    addParticleHistos<3>();
    addParticleHistos<4>();
    addParticleHistos<5>();
    addParticleHistos<6>();
    addParticleHistos<7>();
    addParticleHistos<8>();
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::McTrackLabels> const& tracks, aod::McParticles& mcParticles)
  {
    const float collisionTime_ps = collision.collisionTime() * 1000.f;
    for (auto t : tracks) {
      //
      if (t.tofSignal() < 0) { // Skipping tracks without TOF
        continue;
      }
      if (!MC::isPhysicalPrimary(mcParticles, t.label())) {
        continue;
      }
      float nsigma = -999;
      if (pid_type == 0) {
        nsigma = t.tofNSigmaEl();
      } else if (pid_type == 0) {
        nsigma = t.tofNSigmaMu();
      } else if (pid_type == 1) {
        nsigma = t.tofNSigmaPi();
      } else if (pid_type == 2) {
        nsigma = t.tofNSigmaKa();
      } else if (pid_type == 3) {
        nsigma = t.tofNSigmaPr();
      } else if (pid_type == 4) {
        nsigma = t.tofNSigmaDe();
      } else if (pid_type == 5) {
        nsigma = t.tofNSigmaTr();
      } else if (pid_type == 6) {
        nsigma = t.tofNSigmaHe();
      } else if (pid_type == 7) {
        nsigma = t.tofNSigmaAl();
      }

      histos.fill(HIST(hnsigma[pid_type]), t.pt(), nsigma);
      for (int i = 0; i < 9; i++) {
        if (abs(t.label().pdgCode()) == PDGs[pid_type]) {
          histos.fill(HIST(hnsigmaMC[pid_type]), t.pt(), nsigma);
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{};
  if (cfgc.options().get<int>("pid-el")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Electron>>("pidTOF-qa-El"));
  }
  if (cfgc.options().get<int>("pid-mu")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Muon>>("pidTOF-qa-Mu"));
  }
  if (cfgc.options().get<int>("pid-pikapr")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Pion>>("pidTOF-qa-Pi"));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Kaon>>("pidTOF-qa-Ka"));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Proton>>("pidTOF-qa-Pr"));
  }
  if (cfgc.options().get<int>("pid-nuclei")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Deuteron>>("pidTOF-qa-De"));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Triton>>("pidTOF-qa-Tr"));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Helium3>>("pidTOF-qa-He"));
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQA<PID::Alpha>>("pidTOF-qa-Al"));
  }
  return workflow;
}
