// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TH1.h>
#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/Expressions.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/AnalysisTask.h"
#include "Framework/Condition.h"
#include "SimulationDataFormat/O2DatabasePDG.h"

#include <catch_amalgamated.hpp>

using namespace o2::framework;
using namespace o2::soa;
using namespace o2;

struct P {
  void process1(aod::Collisions const&)
  {
  }

  PROCESS_SWITCH(P, process1, "", true);
};

namespace o2::aod
{
namespace ct
{
DECLARE_SOA_CONFIGURABLE_EXPRESSION_COLUMN(Test, test, float, "test");
}
DECLARE_SOA_CONFIGURABLE_EXTENDED_TABLE(TracksMore, TracksIU, "TRKMORE", ct::Test);
} // namespace o2::aod

TEST_CASE("IdentificationConcepts")
{
  // ASoA
  int i;
  REQUIRE(not_void<decltype(i)>);

  REQUIRE(is_persistent_column<o2::aod::track::CollisionId>);

  REQUIRE(is_self_index_column<o2::aod::mcparticle::DaughtersIdSlice>);

  REQUIRE(!is_index_column<o2::aod::mcparticle::DaughtersIdSlice>);
  REQUIRE(is_index_column<o2::aod::track::CollisionId>);
  REQUIRE(is_index_column<o2::aod::indices::CollisionIds>);

  REQUIRE(o2::aod::is_aod_hash<o2::aod::Hash<"AOD"_h>>);
  REQUIRE(o2::aod::is_origin_hash<o2::aod::Hash<"AOD"_h>>);

  REQUIRE(has_parent_t<o2::aod::Track>);

  REQUIRE(is_metadata<o2::aod::TracksIUExtensionMetadata>);

  REQUIRE(is_metadata_trait<o2::aod::MetadataTrait<o2::aod::Hash<"TRACK/0"_h>>>);

  REQUIRE(has_metadata<o2::aod::MetadataTrait<o2::aod::Hash<"TRACK/0"_h>>>);

  REQUIRE(has_extension<o2::aod::MetadataTrait<o2::aod::Hash<"EXTRACK/0"_h>>::metadata>);

  REQUIRE(is_spawnable_column<o2::aod::track::Pt>);

  REQUIRE(is_indexing_column<Index<>>);

  REQUIRE(is_dynamic_column<o2::aod::track::Energy<o2::aod::track::Signed1Pt, o2::aod::track::Tgl>>);

  REQUIRE(is_marker_column<o2::soa::Marker<1>>);

  REQUIRE(is_column<o2::aod::track::Pt>);
  REQUIRE(is_column<Index<>>);
  REQUIRE(is_column<o2::aod::track::Energy<o2::aod::track::Signed1Pt, o2::aod::track::Tgl>>);
  REQUIRE(is_column<o2::soa::Marker<1>>);

  REQUIRE(is_table<o2::aod::Collisions>);

  REQUIRE(is_iterator<o2::aod::Collision>);

  REQUIRE(with_originals<o2::aod::Collisions>);

  REQUIRE(with_sources<o2::aod::MetadataTrait<o2::aod::Hash<"MA_RN3_SP/0"_h>>::metadata>);

  REQUIRE(with_base_table<o2::aod::Tracks>);

  REQUIRE(is_index_table<o2::aod::Run3MatchedSparse>);

  Preslice<o2::aod::Tracks> ps = o2::aod::track::collisionId;
  REQUIRE(is_preslice<decltype(ps)>);

  struct : PresliceGroup {
    Preslice<o2::aod::Tracks> pc = o2::aod::track::collisionId;
    Preslice<o2::aod::McParticles> pmcc = o2::aod::mcparticle::mcCollisionId;
  } preslices;
  REQUIRE(is_preslice_group<decltype(preslices)>);
  REQUIRE(is_preslice<decltype(preslices.pc)>);
  REQUIRE(is_preslice<decltype(preslices.pmcc)>);

  REQUIRE(has_filtered_policy<soa::Filtered<o2::aod::Tracks>::iterator>);

  REQUIRE(is_filtered_iterator<soa::Filtered<o2::aod::Tracks>::iterator>);

  REQUIRE(is_filtered_table<soa::Filtered<o2::aod::Tracks>>);

  REQUIRE(is_filtered<soa::Filtered<o2::aod::Tracks>::iterator>);
  REQUIRE(is_filtered<soa::Filtered<o2::aod::Tracks>>);

  REQUIRE(is_not_filtered_table<o2::aod::Collisions>);

  REQUIRE(is_join<o2::aod::Tracks>);

  auto tl = []() -> SmallGroups<o2::aod::Collisions> { return {std::vector<std::shared_ptr<arrow::Table>>{}, SelectionVector{}, 0}; };
  REQUIRE(is_smallgroups<decltype(tl())>);

  // AnalysisHelpers
  REQUIRE(is_producable<o2::aod::Collisions>);

  Produces<o2::aod::Collisions> prod;
  REQUIRE(is_produces<decltype(prod)>);

  struct : ProducesGroup {
    Produces<o2::aod::Collisions> p;
  } prodg;
  REQUIRE(is_produces_group<decltype(prodg)>);

  REQUIRE(is_spawnable<o2::aod::Tracks>);

  Spawns<o2::aod::Tracks> spw;
  REQUIRE(is_spawns<decltype(spw)>);

  Builds<o2::aod::Run3MatchedSparse> bld;
  REQUIRE(is_builds<decltype(bld)>);

  Defines<o2::aod::TracksMore> def;
  DefinesDelayed<o2::aod::TracksMore> ddef;
  REQUIRE(is_defines<decltype(def)>);
  REQUIRE(is_defines<decltype(ddef)>);

  OutputObj<TH1F> oo{"test"};
  REQUIRE(is_outputobj<decltype(oo)>);

  Service<o2::O2DatabasePDG> srv;
  REQUIRE(is_service<decltype(srv)>);

  Partition<o2::aod::Tracks> part = o2::aod::track::collisionId >= 0;
  REQUIRE(is_partition<decltype(part)>);

  // AnalysisTask
  Enumeration<0, 1> en;
  REQUIRE(is_enumeration<decltype(en)>);

  // Condition
  Condition<int> c{""};
  REQUIRE(is_condition<decltype(c)>);

  struct : ConditionGroup {
    Condition<float> c{""};
  } cg;
  REQUIRE(is_condition_group<decltype(cg)>);

  // Configurable
  Configurable<int> cc{"", 1, ""};
  REQUIRE(is_configurable<decltype(cc)>);

  ConfigurableAxis ca{"", {0, 1, 2, 3}, ""};
  REQUIRE(is_configurable_axis<decltype(ca)>);

  REQUIRE(is_process_configurable<decltype(P::doprocess1)>);
  REQUIRE(is_process_configurable<decltype((P::doprocess1))>);

  struct : ConfigurableGroup {
    Configurable<int> c{"", 1, ""};
  } ccg;
  REQUIRE(is_configurable_group<decltype(ccg)>);

  // Expressions
  expressions::Filter f = o2::aod::track::pt > 1.0f;
  REQUIRE(expressions::is_filter<decltype(f)>);

  // Combinations
  using C = SameKindPair<aod::Collisions, aod::Tracks, ColumnBinningPolicy<aod::collision::PosZ>>;
  REQUIRE(is_combinations_generator<C>);
}
