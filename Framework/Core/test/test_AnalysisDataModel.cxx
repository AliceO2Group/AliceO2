// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework ASoA
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ASoA.h"
#include "Framework/TableBuilder.h"
#include "Framework/AnalysisDataModel.h"
#include "gandiva/tree_expr_builder.h"
#include "arrow/status.h"
#include "gandiva/filter.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;
using namespace o2::aod;

BOOST_AUTO_TEST_CASE(TestJoinedTables)
{
  TableBuilder trackBuilder;
  auto trackWriter = trackBuilder.cursor<Tracks>();
  trackWriter(0, 0, 0, 0, 0, 0, 0, 0, 0);
  auto tracks = trackBuilder.finalize();

  TableBuilder trackParCovBuilder;
  auto trackParCovWriter = trackParCovBuilder.cursor<TracksCov>();
  trackParCovWriter(0, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
  auto covs = trackParCovBuilder.finalize();

  using Test = Join<Tracks, TracksCov>;

  Test tests{0, tracks, covs};
  BOOST_REQUIRE(tests.asArrowTable()->num_columns() != 0);
  BOOST_REQUIRE_EQUAL(tests.asArrowTable()->num_columns(),
                      tracks->num_columns() + covs->num_columns());
  auto tests2 = join(Tracks{tracks}, TracksCov{covs});
  static_assert(std::is_same_v<Test::table_t, decltype(tests2)>,
                "Joined tables should have the same type, regardless how we construct them");
}
