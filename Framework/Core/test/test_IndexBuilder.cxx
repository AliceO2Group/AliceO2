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

#define BOOST_TEST_MODULE Test Framework IndexBuilder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace arrow;
using namespace o2::soa;

DECLARE_SOA_STORE();
namespace coords
{
DECLARE_SOA_COLUMN_FULL(X, x, float, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, float, "y");
DECLARE_SOA_COLUMN_FULL(Z, z, float, "z");
} // namespace coords
DECLARE_SOA_TABLE(Points, "TST", "POINTS", Index<>, coords::X, coords::Y, coords::Z);

namespace extra_1
{
DECLARE_SOA_INDEX_COLUMN(Point, point);
DECLARE_SOA_COLUMN_FULL(D, d, float, "d");
} // namespace extra_1
DECLARE_SOA_TABLE(Distances, "TST", "DISTANCES", Index<>, extra_1::PointId, extra_1::D);

namespace extra_2
{
DECLARE_SOA_INDEX_COLUMN(Point, point);
DECLARE_SOA_COLUMN_FULL(IsTrue, istrue, bool, "istrue");
} // namespace extra_2
DECLARE_SOA_TABLE(Flags, "TST", "Flags", Index<>, extra_2::PointId, extra_2::IsTrue);

namespace extra_3
{
DECLARE_SOA_INDEX_COLUMN(Point, point);
DECLARE_SOA_COLUMN_FULL(Category, category, int32_t, "category");
} // namespace extra_3
DECLARE_SOA_TABLE(Categorys, "TST", "Categories", Index<>, extra_3::PointId, extra_3::Category);

namespace indices
{
DECLARE_SOA_INDEX_COLUMN(Point, point);
DECLARE_SOA_INDEX_COLUMN(Distance, distance);
DECLARE_SOA_INDEX_COLUMN(Flag, flag);
DECLARE_SOA_INDEX_COLUMN(Category, category);
} // namespace indices

DECLARE_SOA_TABLE(IDXs, "TST", "Index", Index<>, indices::PointId, indices::DistanceId, indices::FlagId, indices::CategoryId);
DECLARE_SOA_TABLE(IDX2s, "TST", "Index2", Index<>, indices::DistanceId, indices::PointId, indices::FlagId, indices::CategoryId);

BOOST_AUTO_TEST_CASE(TestIndexBuilder)
{
  TableBuilder b1;
  auto w1 = b1.cursor<Points>();
  TableBuilder b2;
  auto w2 = b2.cursor<Distances>();
  TableBuilder b3;
  auto w3 = b3.cursor<Flags>();
  TableBuilder b4;
  auto w4 = b4.cursor<Categorys>();

  for (auto i = 0; i < 10; ++i) {
    w1(0, i * 2., i * 3., i * 4.);
  }

  std::array<int, 7> d{0, 1, 2, 4, 7, 8, 9};
  std::array<int, 5> f{0, 1, 2, 5, 8};
  std::array<int, 7> c{0, 1, 2, 3, 5, 7, 8};

  for (auto i : d) {
    w2(0, i, i * 10.);
  }

  for (auto i : f) {
    w3(0, i, static_cast<bool>(i % 2));
  }

  for (auto i : c) {
    w4(0, i, i + 2);
  }

  auto t1 = b1.finalize();
  Points st1{t1};
  auto t2 = b2.finalize();
  Distances st2{t2};
  auto t3 = b3.finalize();
  Flags st3{t3};
  auto t4 = b4.finalize();
  Categorys st4{t4};

  auto t5 = IndexExclusive::indexBuilder("test1", typename IDXs::persistent_columns_t{}, st1, std::tie(st1, st2, st3, st4));
  BOOST_REQUIRE_EQUAL(t5->num_rows(), 4);
  IDXs idxt{t5};
  idxt.bindExternalIndices(&st1, &st2, &st3, &st4);
  for (auto& row : idxt) {
    BOOST_REQUIRE(row.distance().pointId() == row.pointId());
    BOOST_REQUIRE(row.flag().pointId() == row.pointId());
    BOOST_REQUIRE(row.category().pointId() == row.pointId());
  }

  auto t6 = IndexSparse::indexBuilder("test2", typename IDX2s::persistent_columns_t{}, st1, std::tie(st2, st1, st3, st4));
  BOOST_REQUIRE_EQUAL(t6->num_rows(), st2.size());
  IDX2s idxs{t6};
  std::array<int, 7> fs{0, 1, 2, -1, -1, 4, -1};
  std::array<int, 7> cs{0, 1, 2, -1, 5, 6, -1};
  idxs.bindExternalIndices(&st1, &st2, &st3, &st4);
  auto i = 0;
  for (auto const& row : idxs) {
    BOOST_REQUIRE(row.has_distance());
    BOOST_REQUIRE(row.has_point());
    if (row.has_flag()) {
      BOOST_REQUIRE(row.flag().pointId() == row.pointId());
    }
    if (row.has_category()) {
      BOOST_REQUIRE(row.category().pointId() == row.pointId());
    }
    BOOST_REQUIRE(row.flagId() == fs[i]);
    BOOST_REQUIRE(row.categoryId() == cs[i]);
    ++i;
  }
}
