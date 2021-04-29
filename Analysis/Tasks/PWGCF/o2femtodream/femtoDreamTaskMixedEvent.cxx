// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file femtoDreamTaskMixedEvent.cxx
/// \brief Analysis task for particle pairing in mixed events
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "o2femtodream/FemtoDerived.h"
#include "o2femtodream/FemtoDreamContainer.h"

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/ASoAHelpers.h"

#include "TDatabasePDG.h"

using namespace o2;
using namespace o2::analysis::femtoDream;
using namespace o2::framework;

struct femtoDreamTaskPremixing {

  Configurable<std::vector<float>> CfgVtxBins{"CfgVtxBins", std::vector<float>{-10.0f, -8.f, -6.f, -4.f, -2.f, 0.f, 2.f, 4.f, 6.f, 8.f, 10.f}, "Mixing bins - z-vertex"};
  Configurable<std::vector<float>> CfgMultBins{"CfgMultBins", std::vector<float>{0.0f, 20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 99999.f}, "Mixing bins - multiplicity"};

  std::vector<float> CastCfgVtxBins, CastCfgMultBins;

  Produces<aod::Hashes> hashes;
  Produces<aod::MixingEvents> mixingEvents;

  // Calculate hash for an element based on 2 properties and their bins.
  template <typename T1, typename T2>
  int getMixingBin(const T1& vtxBins, const T1& multBins, const T2& vtx, const T2& mult)
  {
    // underflow
    if (vtx < vtxBins.at(0)) {
      return -1;
    }
    if (mult < multBins.at(0)) {
      return -1;
    }

    for (int i = 1; i < vtxBins.size(); i++) {
      if (vtx < vtxBins.at(i)) {
        for (int j = 1; j < multBins.size(); j++) {
          if (mult < multBins.at(j)) {
            return i + j * (vtxBins.size() + 1);
          }
        }
      }
    }

    // overflow
    return -1;
  }

  void init(InitContext&)
  {
    CastCfgVtxBins = (std::vector<float>)CfgVtxBins;
    CastCfgMultBins = (std::vector<float>)CfgMultBins;
  }

  void process(aod::FemtoDreamCollision& collision, aod::FemtoDreamParticles& particles)
  {
    hashes(getMixingBin(CastCfgVtxBins, CastCfgMultBins, collision.posZ(), collision.multV0M()));

    /// \todo should differ for identical and non-identical pairs / triplets
    /// \todo make configurable
    if (particles.size() > 0) {
      mixingEvents(1);
    } else {
      mixingEvents(0);
    }
  }
};

struct femtoDreamTaskMixedEvent {

  O2_DEFINE_CONFIGURABLE(CfgPDGCodePartOne, int, 2212, "PDG Code of particle one");
  O2_DEFINE_CONFIGURABLE(CfgPDGCodePartTwo, int, 2212, "PDG Code of particle two");

  //  Filter evtFilter = (aod::mixingEvent::use > 0);

  /// Histograms
  FemtoDreamContainer* mixedEventCont;
  HistogramRegistry resultRegistry{"Correlations", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext&)
  {
    mixedEventCont = new FemtoDreamContainer();//(&resultRegistry);
    mixedEventCont->setMasses(TDatabasePDG::Instance()->GetParticle(CfgPDGCodePartOne)->Mass(),
                              TDatabasePDG::Instance()->GetParticle(CfgPDGCodePartTwo)->Mass());
  }

  void process(soa::Join<aod::FemtoDreamCollisions, aod::Hashes, aod::MixingEvents>& collisions, aod::FemtoDreamParticles& particles)
  {

    collisions.bindExternalIndices(&particles);
    auto particlesTuple = std::make_tuple(particles);
    AnalysisDataProcessorBuilder::GroupSlicer slicer(collisions, particlesTuple);

    for (auto& [collision1, collision2] : selfCombinations("fBin", 5, -1, collisions, collisions)) {

      // can't this be run as a filter? Apparently not
      if (!collision1.use() || !collision2.use()) {
        continue;
      }

      auto it1 = slicer.begin();
      auto it2 = slicer.begin();
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision1.index()) {
          it1 = slice;
          break;
        }
      }
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision2.index()) {
          it2 = slice;
          break;
        }
      }

      auto particles1 = std::get<aod::FemtoDreamParticles>(it1.associatedTables());
      particles1.bindExternalIndices(&collisions);
      auto particles2 = std::get<aod::FemtoDreamParticles>(it2.associatedTables());
      particles2.bindExternalIndices(&collisions);

      // doesn't work for some reason
      //  for (auto& [t1, t2] : combinations(CombinationsFullIndexPolicy(tracks1, tracks2))) {
      for (auto& part1 : particles1) {
        for (auto& part2 : particles2) {
          mixedEventCont->setPair(part1, part2);
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<femtoDreamTaskPremixing>(cfgc),
                        adaptAnalysisTask<femtoDreamTaskMixedEvent>(cfgc)};
  return workflow;
}
