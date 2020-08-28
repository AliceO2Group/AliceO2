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
#include "Analysis/Multiplicity.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Centrality.h"
#include <TH1F.h>
#include <TH2F.h>
#include <cmath>

const double ketaBinWidth=0.1;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct PseudorapidityDensity {
  float etaMax = 1.5;
  float etaMin = -1.5;
  float vtxZMax = 10;
  float vtxZMin = -10;
  int etaBins = TMath::Nint((etaMax-etaMin)/ketaBinWidth);
  int vtxZBins = TMath::Nint(vtxZMax-vtxZMin);

  OutputObj<TH1F> hStat{TH1F("hStat", "TotalEvents", 1, 0.5, 1.5)};
  OutputObj<TH1F> hdNdeta{TH1F("hdNdeta", "dNdeta", 50, -2.5, 2.5)};
  OutputObj<TH2F> vtxZEta{TH2F("vtxZEta", ";#eta;vtxZ", 50, -2.5, 2.5, 60, -30, 30)};
  OutputObj<TH2F> phiEta{TH2F("phiEta", ";#eta;#varphi", 50, -2.5, 2.5, 200, 0. , 2*TMath::Pi())};


 Filter etaFilter = (aod::track::eta < etaMax) && (aod::track::eta > etaMin);
 Filter trackTypeFilter = (aod::track::trackType == static_cast<uint8_t>(aod::track::TrackTypeEnum::Run2Tracklet));
 Filter posZFilter = (aod::collision::posZ < vtxZMax) && (aod::collision::posZ > vtxZMin);

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collisions, soa::Filtered<aod::Tracks> const& tracklets)
  {
    if (!collisions.sel7())
      return;
    hStat->Fill(collisions.size());
    float vtxZ = collisions.posZ();
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

