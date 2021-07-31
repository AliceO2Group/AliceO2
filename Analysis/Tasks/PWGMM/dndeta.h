// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef DNDETA_H
#define DNDETA_H
#include "Framework/Configurable.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "AnalysisCore/MC.h"
#include "TDatabasePDG.h"
namespace o2::aod
{
namespace mcparticle
{
DECLARE_SOA_COLUMN(Primary, primary, bool);
DECLARE_SOA_COLUMN(Charge, charge, float);
} // namespace mcparticle
DECLARE_SOA_TABLE(SelPrimaries, "AOD", "SPRIM", aod::mcparticle::Primary, aod::mcparticle::Charge);
} // namespace o2::aod

namespace o2::pwgmm::multiplicity
{
template <bool RUN3>
struct SelectPhysicalPrimaries {
  o2::framework::Produces<o2::aod::SelPrimaries> prims;
  //  TDatabasePDG* pdg; //until it is made into a Service
  //  void init(o2::framework::InitContext&)
  //  {
  //    pdg = new TDatabasePDG();
  //    pdg->ReadPDGTable();
  //  }
  void process(o2::aod::McParticles& particles)
  {
    auto pdg = new TDatabasePDG();
    pdg->ReadPDGTable();
    for (auto& particle : particles) {
      auto p = pdg->GetParticle(particle.pdgCode());
      float charge = 0;
      if (p == nullptr) {
        LOGF(WARN, "[%d] Unknown particle with PDG code %d", particle.globalIndex(), particle.pdgCode());
      } else {
        charge = p->Charge();
      }
      if constexpr (RUN3) {
        prims(MC::isPhysicalPrimaryRun3(particle), charge);
      } else {
        prims(MC::isPhysicalPrimary(particle), charge);
      }
    }
    delete pdg;
  }
};

template <uint8_t TRACKTYPE>
struct PseudorapidityDensity {
  o2::framework::Configurable<float> etaMax{"etaMax", 2.0, "max eta value"};
  o2::framework::Configurable<float> etaMin{"etaMin", -2.0, "min eta value"};
  o2::framework::Configurable<float> vtxZMax{"vtxZMax", 15, "max z vertex"};
  o2::framework::Configurable<float> vtxZMin{"vtxZMin", -15, "min z vertex"};

  o2::framework::HistogramRegistry registry{
    "registry",
    {
      {"EventsNtrkZvtx", "; N_{trk}; Z_{vtx}; events", {o2::framework::HistType::kTH2F, {{301, -0.5, 300.5}, {201, -20.1, 20.1}}}}, //
      {"TracksEtaZvtx", "; #eta; Z_{vtx}; tracks", {o2::framework::HistType::kTH2F, {{21, -2.1, 2.1}, {201, -20.1, 20.1}}}},        //
      {"TracksPhiEta", "; #varphi; #eta; tracks", {o2::framework::HistType::kTH2F, {{600, 0, 2 * M_PI}, {21, -2.1, 2.1}}}},         //
      {"EventSelection", ";status;events", {o2::framework::HistType::kTH1F, {{3, 0.5, 3.5}}}},                                      //
    }                                                                                                                               //
  };

  void init(o2::framework::InitContext&);

  o2::framework::expressions::Filter etaFilter = (aod::track::eta < etaMax) && (aod::track::eta > etaMin);
  o2::framework::expressions::Filter trackTypeFilter = (aod::track::trackType == TRACKTYPE);
  o2::framework::expressions::Filter posZFilter = (aod::collision::posZ < vtxZMax) && (aod::collision::posZ > vtxZMin);

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks);
};
using namespace o2::framework;
template <uint8_t TRACKTYPE>
struct PseudorapidityDensityMc {
  o2::framework::Configurable<float> etaMax{"etaMax", 2.0, "max eta value"};
  o2::framework::Configurable<float> etaMin{"etaMin", -2.0, "min eta value"};
  o2::framework::Configurable<float> vtxZMax{"vtxZMax", 15, "max z vertex"};
  o2::framework::Configurable<float> vtxZMin{"vtxZMin", -15, "min z vertex"};

  o2::framework::HistogramRegistry registry{
    "registry",
    {
      {"EventsNtrkZvtxGen", "; N_{trk}; Z_{vtx}; events", {o2::framework::HistType::kTH2F, {{301, -0.5, 300.5}, {201, -20.1, 20.1}}}}, //
      {"TracksEtaZvtxGen", "; #eta; Z_{vtx}; tracks", {o2::framework::HistType::kTH2F, {{21, -2.1, 2.1}, {201, -20.1, 20.1}}}},        //
      {"TracksPhiEtaGen", "; #varphi; #eta; tracks", {o2::framework::HistType::kTH2F, {{600, 0, 2 * M_PI}, {21, -2.1, 2.1}}}},         //
      {"EventEfficiency", "; status; events", {o2::framework::HistType::kTH1F, {{3, 0.5, 3.5}}}}                                       //
    }                                                                                                                                  //
  };

  void init(o2::framework::InitContext&);

  o2::framework::expressions::Filter etaFilter = (aod::track::eta < etaMax) && (aod::track::eta > etaMin);
  o2::framework::expressions::Filter trackTypeFilter = (aod::track::trackType == TRACKTYPE);
  o2::framework::expressions::Filter posZFilter = (aod::collision::posZ < vtxZMax) && (aod::collision::posZ > vtxZMin);
  o2::framework::expressions::Filter posZFilterMC = (aod::mccollision::posZ < vtxZMax) && (aod::mccollision::posZ > vtxZMin);

  using Particles = soa::Join<aod::McParticles, aod::SelPrimaries>;
  o2::framework::expressions::Filter chargedPrimariesFilter = (aod::mcparticle::primary == true) && (aod::mcparticle::charge != 0);

  void processGen(soa::Filtered<aod::McCollisions>::iterator const& collision, soa::Filtered<Particles> const& primaries);

  PROCESS_SWITCH(PseudorapidityDensityMc<TRACKTYPE>, processGen, "Process generator-level info", true);

  void processMatching(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::McCollisionLabels>>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks, aod::McCollisions const&);

  PROCESS_SWITCH(PseudorapidityDensityMc<TRACKTYPE>, processMatching, "Process generator-level info matched to reco", true);
};
} // namespace o2::pwgmm::multiplicity

#endif // DNDETA_H
