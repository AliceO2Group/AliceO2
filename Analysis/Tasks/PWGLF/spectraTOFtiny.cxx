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
/// \file   spectraTOFtiny.h
/// \author Nicolo' Jacazio
///
/// \brief Task for the analysis of the spectra with the TOF detector using the tiny tables
///

// O2 includes
#include "ReconstructionDataFormats/Track.h"
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// Spectra task
struct tofSpectraTiny {
  static constexpr int Np = 9;
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

  template <std::size_t i, typename T>
  void fillParticleHistos(const T& track, const float& nsigma)
  {
    if (abs(nsigma) > cfgNSigmaCut) {
      return;
    }
    histos.fill(HIST(hp[i]), track.p());
    histos.fill(HIST(hpt[i]), track.pt());
  }

  //Defining filters and input
  Configurable<float> cfgNSigmaCut{"cfgNSigmaCut", 3, "Value of the Nsigma cut"};
  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"};
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::isGlobalTrack == (uint8_t) true);
  Filter trackFilterTOF = (aod::track::tofSignal > 0.f); // Skip tracks without TOF
  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra,
                                                  aod::pidTOFEl, aod::pidTOFMu, aod::pidTOFPi,
                                                  aod::pidTOFKa, aod::pidTOFPr, aod::pidTOFDe,
                                                  aod::pidTOFTr, aod::pidTOFHe, aod::pidTOFAl,
                                                  aod::TrackSelection>>;

  void process(TrackCandidates::iterator const& track)
  {
    histos.fill(HIST("p/Unselected"), track.p());
    histos.fill(HIST("pt/Unselected"), track.pt());

    fillParticleHistos<0>(track, track.tofNSigmaEl());
    fillParticleHistos<1>(track, track.tofNSigmaMu());
    fillParticleHistos<2>(track, track.tofNSigmaPi());
    fillParticleHistos<3>(track, track.tofNSigmaKa());
    fillParticleHistos<4>(track, track.tofNSigmaPr());
    fillParticleHistos<5>(track, track.tofNSigmaDe());
    fillParticleHistos<6>(track, track.tofNSigmaTr());
    fillParticleHistos<7>(track, track.tofNSigmaHe());
    fillParticleHistos<8>(track, track.tofNSigmaAl());
  } // end of the process function
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<tofSpectraTiny>(cfgc)};
}
