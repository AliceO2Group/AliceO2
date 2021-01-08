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
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include <TH1F.h>
#include <TH2F.h>
#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct PseudorapidityDensity {

  Configurable<double> etaBinWidth{"etaBinWidth", 0.1, "eta bin width"};
  Configurable<float> etaMax{"etaMax", 1.5, "max eta value"};
  Configurable<float> etaMin{"etaMin", -1.5, "min eta value"};
  Configurable<float> vtxZMax{"vtxZMax", 10, "max z vertex"};
  Configurable<float> vtxZMin{"vtxZMin", -10, "min z vertex"};
  int etaBins = TMath::Nint((etaMax - etaMin) / etaBinWidth);
  int vtxZBins = TMath::Nint(vtxZMax - vtxZMin);

  OutputObj<TH1F> hStat{TH1F("hStat", "TotalEvents", 1, 0.5, 1.5)};
  OutputObj<TH1F> hdNdeta{TH1F("hdNdeta", "dNdeta", 50, -2.5, 2.5)};
  OutputObj<TH2F> vtxZEta{TH2F("vtxZEta", ";#eta;vtxZ", 50, -2.5, 2.5, 60, -30, 30)};
  OutputObj<TH2F> phiEta{TH2F("phiEta", ";#eta;#varphi", 50, -2.5, 2.5, 200, 0., 2 * TMath::Pi())};

  // TODO remove static casts for configurables when fixed
  Filter etaFilter = (aod::track::eta < (float)etaMax) && (aod::track::eta > (float)etaMin);
  Filter trackTypeFilter = (aod::track::trackType == static_cast<uint8_t>(aod::track::TrackTypeEnum::Run2Tracklet));
  Filter posZFilter = (aod::collision::posZ < (float)vtxZMax) && (aod::collision::posZ > (float)vtxZMin);

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collisions, soa::Filtered<aod::Tracks> const& tracklets)
  {
    // TODO change to sel7 filter expression when implemented
    if (!collisions.sel7())
      return;
    hStat->Fill(collisions.size());
    auto vtxZ = collisions.posZ();
    for (auto& track : tracklets) {
      vtxZEta->Fill(track.eta(), vtxZ);
      phiEta->Fill(track.eta(), track.phi());
      hdNdeta->Fill(track.eta());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<PseudorapidityDensity>("dNdetaRun2Tracklets-analysis")};
}
