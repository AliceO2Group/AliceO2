// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "dndeta.h"
#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::pwgmm::multiplicity;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Use MC information"}};
  workflowOptions.push_back(optionDoMC);
}
// always should be after customize() function
#include "Framework/runDataProcessing.h"

template <>
void PseudorapidityDensity<aod::track::TrackTypeEnum::Run2Tracklet>::init(o2::framework::InitContext&)
{
  auto hstat = registry.get<TH1>(HIST("EventSelection"));
  auto x = hstat->GetXaxis();
  x->SetBinLabel(1, "All");
  x->SetBinLabel(2, "Selected");
  x->SetBinLabel(3, "Rejected");
}

template <>
void PseudorapidityDensity<aod::track::TrackTypeEnum::Run2Tracklet>::process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks)
{
  registry.fill(HIST("EventSelection"), 1.);
  if (collision.sel7()) {
    registry.fill(HIST("EventSelection"), 2.);
    auto z = collision.posZ();
    registry.fill(HIST("EventsNtrkZvtx"), tracks.size(), z);
    for (auto& track : tracks) {
      registry.fill(HIST("TracksEtaZvtx"), track.eta(), z);
      registry.fill(HIST("TracksPhiEta"), track.phi(), track.eta());
    }
  } else {
    registry.fill(HIST("EventSelection"), 3.);
  }
}

template <>
void PseudorapidityDensityMc<aod::track::TrackTypeEnum::Run2Tracklet>::init(o2::framework::InitContext&)
{
  auto h = registry.get<TH1>(HIST("EventEfficiency"));
  auto x = h->GetXaxis();
  x->SetBinLabel(1, "Generated");
  x->SetBinLabel(2, "Reconstructed");
  x->SetBinLabel(3, "Selected");
}

template <>
void PseudorapidityDensityMc<aod::track::TrackTypeEnum::Run2Tracklet>::processGen(soa::Filtered<aod::McCollisions>::iterator const& collision, soa::Filtered<Particles> const& primaries)
{
  registry.fill(HIST("EventEfficiency"), 1.);
  for (auto& particle : primaries) {
    if ((particle.eta() < etaMax) && (particle.eta() > etaMin)) {
      registry.fill(HIST("TracksEtaZvtxGen"), particle.eta(), collision.posZ());
      registry.fill(HIST("TracksPhiEtaGen"), particle.phi(), particle.eta());
    }
  }
}

template <>
void PseudorapidityDensityMc<aod::track::TrackTypeEnum::Run2Tracklet>::processMatching(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::McCollisionLabels>>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks, aod::McCollisions const&)
{
  registry.fill(HIST("EventEfficiency"), 2.);
  if (collision.sel7()) {
    registry.fill(HIST("EventEfficiency"), 3.);
    auto z = collision.mcCollision().posZ();
    registry.fill(HIST("EventsNtrkZvtxGen"), tracks.size(), z);
  }
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  const bool doMC = cfgc.options().get<bool>("doMC");
  WorkflowSpec ws{adaptAnalysisTask<PseudorapidityDensity<aod::track::TrackTypeEnum::Run2Tracklet>>(cfgc, TaskName{"pseudorapidity-density"})};
  if (doMC) {
    ws.push_back(adaptAnalysisTask<SelectPhysicalPrimaries<false>>(cfgc, TaskName{"select-physical-primaries"}));
    ws.push_back(adaptAnalysisTask<PseudorapidityDensityMc<aod::track::TrackTypeEnum::Run2Tracklet>>(cfgc, TaskName{"pseudorapidity-density-mc"}));
  }
  return ws;
}
