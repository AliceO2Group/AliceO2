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
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

#include "AnalysisDataModel/EventSelection.h"
#include "filterTables.h"

#include "Framework/HistogramRegistry.h"

#include <cmath>
#include <string>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace
{
float rapidity(float pt, float eta, float m)
{
  return std::asinh(pt / std::hypot(m, pt) * std::sinh(eta));
}

static constexpr int nNuclei{4};
static constexpr int nCutsPID{5};
static constexpr std::array<float, nNuclei> masses{
  constants::physics::MassDeuteron, constants::physics::MassTriton,
  constants::physics::MassHelium3, constants::physics::MassAlpha};
static constexpr std::array<int, nNuclei> charges{1, 1, 2, 2};
static const std::vector<std::string> nucleiNames{"H2", "H3", "He3", "He4"};
static const std::vector<std::string> cutsNames{
  "TPCnSigmaMin", "TPCnSigmaMax", "TOFnSigmaMin", "TOFnSigmaMax", "TOFpidStartPt"};
static constexpr float cutsPID[nNuclei][nCutsPID]{
  {-3.f, +3.f, -4.f, +4.f, 1.0f},    /*H2*/
  {-3.f, +3.f, -4.f, +4.f, 1.6f},    /*H3*/
  {-5.f, +5.f, -4.f, +4.f, 14000.f}, /*He3*/
  {-5.f, +5.f, -4.f, +4.f, 14000.f}  /*He4*/
};
} // namespace

struct nucleiFilter {

  Produces<aod::NucleiFilters> tags;

  Configurable<float> yMin{"yMin", -0.8, "Maximum rapidity"};
  Configurable<float> yMax{"yMax", 0.8, "Minimum rapidity"};
  Configurable<float> yBeam{"yBeam", 0., "Beam rapidity"};

  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"};
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};

  Configurable<LabeledArray<float>> cfgCutsPID{"nucleiCutsPID", {cutsPID[0], nNuclei, nCutsPID, nucleiNames, cutsNames}, "Nuclei PID selections"};

  HistogramRegistry spectra{"spectra", {}, OutputObjHandlingPolicy::AnalysisObject, true, true};

  void init(o2::framework::InitContext&)
  {
    std::vector<double> ptBinning = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8, 3.2, 3.6, 4., 5.};
    std::vector<double> centBinning = {0., 1., 5., 10., 20., 30., 40., 50., 70., 100.};

    AxisSpec ptAxis = {ptBinning, "#it{p}_{T} (GeV/#it{c})"};
    AxisSpec centAxis = {centBinning, "V0M (%)"};

    spectra.add("fCollZpos", "collision z position", HistType::kTH1F, {{600, -20., +20., "z position (cm)"}});
    spectra.add("fTPCsignal", "Specific energy loss", HistType::kTH2F, {{600, 0., 3, "#it{p} (GeV/#it{c})"}, {1400, 0, 1400, "d#it{E} / d#it{X} (a. u.)"}});
    spectra.add("fTPCcounts", "n-sigma TPC", HistType::kTH2F, {ptAxis, {200, -100., +100., "n#sigma_{He} (a. u.)"}});
    spectra.add("fProcessedEvents", "Nuclei - event filtered", HistType::kTH1F, {{4, -0.5, 3.5, "Event counter"}});
  }

  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::isGlobalTrack == static_cast<uint8_t>(1u));

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCDe, aod::pidTPCTr, aod::pidTPCHe, aod::pidTPCAl, aod::pidTOFDe, aod::pidTOFTr, aod::pidTOFHe, aod::pidTOFAl, aod::TrackSelection>>;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision, aod::BCsWithTimestamps const&, TrackCandidates const& tracks)
  {
    // collision process loop
    bool keepEvent[nNuclei]{false};
    //
    spectra.fill(HIST("fCollZpos"), collision.posZ());
    //

    for (auto track : tracks) { // start loop over tracks

      const float nSigmaTPC[nNuclei]{
        track.tpcNSigmaDe(), track.tpcNSigmaTr(), track.tpcNSigmaHe(), track.tpcNSigmaAl()};
      const float nSigmaTOF[nNuclei]{
        track.tofNSigmaDe(), track.tofNSigmaTr(), track.tofNSigmaHe(), track.tofNSigmaAl()};

      for (int iN{0}; iN < nNuclei; ++iN) {
        float y{rapidity(track.pt() * charges[iN], track.eta(), masses[iN])};
        if (y < yMin + yBeam || y > yMax + yBeam) {
          continue;
        }
        if (nSigmaTPC[iN] < cfgCutsPID->get(iN, 0u) || nSigmaTPC[iN] > cfgCutsPID->get(iN, 1u)) {
          continue;
        }
        if (track.pt() > cfgCutsPID->get(iN, 4u) && (nSigmaTOF[iN] < cfgCutsPID->get(iN, 2u) || nSigmaTOF[iN] > cfgCutsPID->get(iN, 3u))) {
          continue;
        }
        keepEvent[iN] = true;
      }

      //
      // fill QA histograms
      //
      spectra.fill(HIST("fTPCsignal"), track.tpcInnerParam(), track.tpcSignal());
      spectra.fill(HIST("fTPCcounts"), track.tpcInnerParam(), track.tpcNSigmaHe());

    } // end loop over tracks
    //
    for (int iDecision{0}; iDecision < 4; ++iDecision) {
      if (keepEvent[iDecision]) {
        spectra.fill(HIST("fProcessedEvents"), iDecision);
      }
    }
    tags(keepEvent[0], keepEvent[1], keepEvent[2], keepEvent[3]);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  return WorkflowSpec{
    adaptAnalysisTask<nucleiFilter>(cfg)};
}
