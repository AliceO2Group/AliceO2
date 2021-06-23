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

DECLARE_SOA_STORE();

namespace col
{
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);
DECLARE_SOA_COLUMN(D, d, float);
} // namespace col

DECLARE_SOA_TABLE(XY, "AOD", "XY", col::X, col::Y);
DECLARE_SOA_TABLE(ZD, "AOD", "ZD", col::Z, col::D);

BOOST_AUTO_TEST_CASE(TestJoinedTables)
{
  TableBuilder XYBuilder;
  //FIXME: using full tracks, instead of stored because of unbound dynamic
  //       column (normalized phi)
  auto xyWriter = XYBuilder.cursor<XY>();
  xyWriter(0, 0, 0);
  auto tXY = XYBuilder.finalize();

  TableBuilder ZDBuilder;
  auto zdWriter = ZDBuilder.cursor<ZD>();
  zdWriter(0, 7, 1);
  auto tZD = ZDBuilder.finalize();

  using Test = o2::soa::Join<XY, ZD>;

  Test tests{0, tXY, tZD};
  BOOST_REQUIRE(tests.asArrowTable()->num_columns() != 0);
  BOOST_REQUIRE_EQUAL(tests.asArrowTable()->num_columns(),
                      tXY->num_columns() + tZD->num_columns());
  auto tests2 = join(XY{tXY}, ZD{tZD});
  static_assert(std::is_same_v<Test::table_t, decltype(tests2)>,
                "Joined tables should have the same type, regardless how we construct them");
}
