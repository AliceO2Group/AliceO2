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
#include "ReconstructionDataFormats/Track.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "DataModel/LFDerived.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"add-tof-histos", VariantType::Int, 0, {"Generate TPC with TOF histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

// FIXME: we should put this function in some common header so it has to be defined only once
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

constexpr int Np = 9;
struct TPCSpectraAnalyserTask {

  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr std::string_view hp[Np] = {"p/El", "p/Mu", "p/Pi", "p/Ka", "p/Pr", "p/De", "p/Tr", "p/He", "p/Al"};
  static constexpr std::string_view hpt[Np] = {"pt/El", "pt/Mu", "pt/Pi", "pt/Ka", "pt/Pr", "pt/De", "pt/Tr", "pt/He", "pt/Al"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(o2::framework::InitContext&)
  {
    histos.add("p/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    histos.add("pt/Unselected", "Unselected;#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    for (int i = 0; i < Np; i++) {
      histos.add(hp[i].data(), Form("%s;#it{p} (GeV/#it{c})", pT[i]), kTH1F, {{100, 0, 20}});
      histos.add(hpt[i].data(), Form("%s;#it{p}_{T} (GeV/#it{c})", pT[i]), kTH1F, {{100, 0, 20}});
    }
  }

  Configurable<float> nsigmacut{"nsigmacut", 3, "Value of the Nsigma cut"};
  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"};
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};

  template <std::size_t i, typename T>
  void fillParticleHistos(const T& track, const float nsigma[])
  {
    if (abs(nsigma[i]) > nsigmacut.value) {
      return;
    }
    histos.fill(HIST(hp[i]), track.p());
    histos.fill(HIST(hpt[i]), track.pt());
  }

  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex; //collision filters not doing anything now?
  Filter trackFilter = nabs(aod::lftrack::eta) < cfgCutEta;

  void process(soa::Filtered<aod::LFTracks>::iterator const& track)
  {
    auto nsigma = track.tpcNSigma();
    histos.fill(HIST("p/Unselected"), track.p());
    histos.fill(HIST("pt/Unselected"), track.pt());

    fillParticleHistos<0>(track, nsigma);
    fillParticleHistos<1>(track, nsigma);
    fillParticleHistos<2>(track, nsigma);
    fillParticleHistos<3>(track, nsigma);
    fillParticleHistos<4>(track, nsigma);
    fillParticleHistos<5>(track, nsigma);
    fillParticleHistos<6>(track, nsigma);
    fillParticleHistos<7>(track, nsigma);
    fillParticleHistos<8>(track, nsigma);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<TPCSpectraAnalyserTask>("tpcspectra-task-skim-analyser")};
  return workflow;
}
