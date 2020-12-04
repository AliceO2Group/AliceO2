// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"

#include "AnalysisDataModel/PID/PIDResponse.h"

#include <TH1F.h>
#include <TH2F.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct NucleiSpecraTask {

  OutputObj<TH2F> hTPCsignal{TH2F("hTPCsignal", ";#it{p} (GeV/#it{c}); d#it{E} / d#it{X} (a. u.)", 600, 0., 3, 1400, 0, 1400)};
  OutputObj<TH1F> hMomentum{TH1F("hMomentum", ";#it{p} (GeV/#it{c});", 600, 0., 3.)};

  Configurable<float> absEtaMax{"absEtaMax", 0.8, "pseudo-rapidity edges"};
  Configurable<float> absYmax{"absYmax", 0.5, "rapidity edges"};
  Configurable<float> beamRapidity{"yBeam", 0., "beam rapidity"};
  Configurable<float> chi2TPCperNDF{"chi2TPCperNDF", 4., "chi2 per NDF in TPC"};
  Configurable<float> foundFractionTPC{"foundFractionTPC", 0., "TPC clusters / TPC crossed rows"};
  Configurable<int> recPointsTPC{"recPointsTPC", 0, "clusters in TPC"};
  Configurable<int> signalClustersTPC{"signalClustersTPC", 70, "clusters with PID in TPC"};
  Configurable<float> minEnergyLoss{"minEnergyLoss", 0., "energy loss in TPC"};
  Configurable<int> recPointsITS{"recPointsITS", 2, "number of ITS points"};
  Configurable<int> recPointsITSInnerBarrel{"recPointsITSInnerBarrel", 1, "number of points in ITS Inner Barrel"};

  Filter etaFilter = aod::track::eta > -1 * absEtaMax&& aod::track::eta < absEtaMax;
  Filter chi2Filter = aod::track::tpcChi2NCl < chi2TPCperNDF;

  void process(soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra>> const& tracks)
  {
    for (auto& track : tracks) {
      // Part not covered by filters
      if (track.tpcNClsFound() < recPointsTPC) {
        continue;
      }
      if (track.itsNCls() < recPointsITS) {
        continue;
      }
      if (track.itsNClsInnerBarrel() < recPointsITSInnerBarrel) {
        continue;
      }

      hTPCsignal->Fill(track.tpcInnerParam(), track.tpcSignal());
      hMomentum->Fill(track.p());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<NucleiSpecraTask>("nuclei-spectra")};
}
