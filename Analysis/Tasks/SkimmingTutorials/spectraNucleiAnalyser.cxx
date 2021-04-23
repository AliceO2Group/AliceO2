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
#include "DataModel/LFDerived.h"

#include "Framework/HistogramRegistry.h"

#include <TLorentzVector.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct NucleiSpectraAnalyserTask {

  Configurable<float> yMin{"yMin", -0.8, "Maximum rapidity"};
  Configurable<float> yMax{"yMax", 0.8, "Minimum rapidity"};
  Configurable<float> yBeam{"yBeam", 0., "Beam rapidity"};

  Configurable<float> nSigmaCutLow{"nSigmaCutLow", -30.0, "Value of the Nsigma cut"};
  Configurable<float> nSigmaCutHigh{"nSigmaCutHigh", +3., "Value of the Nsigma cut"};

  HistogramRegistry spectra{"spectra", {}, OutputObjHandlingPolicy::AnalysisObject, true, true};

  void init(o2::framework::InitContext&)
  {
    spectra.add("fCollZpos", "collision z position", HistType::kTH1F, {{600, -20., +20., "z position (cm)"}});
    spectra.add("fHePt", "He pT", HistType::kTH1F, {{60, 0.0, 30., "#it{p}_{T} (GeV/#it{c})"}});
  }
  void process(aod::LFCollision const& collision, aod::LFNucleiTracks const& tracks)
  {

    spectra.fill(HIST("fCollZpos"), collision.posZ());

    for (auto track : tracks) { // start loop over tracks
      TLorentzVector cutVector{};
      cutVector.SetPtEtaPhiM(track.pt() * 2.0, track.eta(), track.phi(), constants::physics::MassHelium3);
      if (cutVector.Rapidity() < yMin + yBeam || cutVector.Rapidity() > yMax + yBeam) {
        continue;
      }
      float nsigma = track.tpcNSigmaHe();
      if (nsigma < nSigmaCutLow || nsigma > nSigmaCutHigh) {
        continue;
      }
      spectra.fill(HIST("fHePt"), track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<NucleiSpectraAnalyserTask>(cfgc, TaskName{"nucleispectra-task-skim-analyser"})};
}
