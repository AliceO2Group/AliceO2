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

// #include "AnalysisDataModel/EventSelection.h"
// #include "AnalysisDataModel/TrackSelectionTables.h"
// #include "AnalysisDataModel/Centrality.h"

#include "Framework/HistogramRegistry.h"

#include <TLorentzVector.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct NucleiSpecraTask {

  HistogramRegistry spectra{"spectra", {}, OutputObjHandlingPolicy::AnalysisObject, true, true};

  void init(o2::framework::InitContext&)
  {
    std::vector<double> ptBinning = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8, 3.2, 3.6, 4., 5.};
    std::vector<double> centBinning = {0., 1., 5., 10., 20., 30., 40., 50., 70., 100.};

    AxisSpec ptAxis = {ptBinning, "#it{p}_{T} (GeV/#it{c})"};
    AxisSpec centAxis = {centBinning, "V0M (%)"};

    spectra.add("fTPCsignal", "Specific energy loss", HistType::kTH2F, {{600, 0., 3, "#it{p} (GeV/#it{c})"}, {1400, 0, 1400, "d#it{E} / d#it{X} (a. u.)"}});

    spectra.add("fTPCcounts", "n-sigma TPC", HistType::kTH2F, {ptAxis, {200, -5, 5, "n#sigma_{d} (a. u.)"}});
  }

  Configurable<float> yMin{"yMin", -0.5, "Maximum rapidity"};
  Configurable<float> yMax{"yMax", 0.5, "Minimum rapidity"};
  Configurable<float> yBeam{"yBeam", 0., "Beam rapidity"};

  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"};
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};
  Configurable<float> nsigmacut{"nsigmacut", 3, "Value of the Nsigma cut"};

  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::isGlobalTrack == (uint8_t) true);

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::pidRespTOF, aod::pidRespTOFbeta, aod::TrackSelection>>;

  void process(/*soa::Join<aod::Collisions, aod::EvSels, aod::Cents> aod::Collisions::iterator const& col, */ TrackCandidates const& tracks)
  {
    /*
    if (!col.alias()[kINT7])
      return;
    if (!col.sel7())
      return;

    fMultiplicity->Fill(col.centV0M());
    */
    for (auto track : tracks) {

      TLorentzVector cutVector{};
      cutVector.SetPtEtaPhiM(track.pt(), track.eta(), track.phi(), constants::physics::MassDeuteron);
      if (cutVector.Rapidity() < yMin + yBeam || cutVector.Rapidity() > yMax + yBeam) {
        continue;
      }

      spectra.fill(HIST("fTPCsignal"), track.tpcInnerParam(), track.tpcSignal());
      spectra.fill(HIST("fTPCcounts"), fabs(track.pt()), track.tpcNSigmaDe());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<NucleiSpecraTask>("nuclei-spectra")};
}
