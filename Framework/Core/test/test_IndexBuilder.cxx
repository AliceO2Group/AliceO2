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

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include <catch_amalgamated.hpp>

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

TEST_CASE("TestIndexBuilder")
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

  auto t5 = IndexBuilder<Exclusive>::indexBuilder<Points>("test1a", {t1, t2, t3, t4}, typename IDXs::persistent_columns_t{}, o2::framework::pack<Points, Distances, Flags, Categorys>{});
  REQUIRE(t5->num_rows() == 4);
  IDXs idxt{t5};
  idxt.bindExternalIndices(&st1, &st2, &st3, &st4);
  for (auto& row : idxt) {
    REQUIRE(row.distance().pointId() == row.pointId());
    REQUIRE(row.flag().pointId() == row.pointId());
    REQUIRE(row.category().pointId() == row.pointId());
  }

  auto t6 = IndexBuilder<Sparse>::indexBuilder<Points>("test3", {t2, t1, t3, t4}, typename IDX2s::persistent_columns_t{}, o2::framework::pack<Distances, Points, Flags, Categorys>{});
  REQUIRE(t6->num_rows() == st2.size());
  IDXs idxs{t6};
  std::array<int, 7> fs{0, 1, 2, -1, -1, 4, -1};
  std::array<int, 7> cs{0, 1, 2, -1, 5, 6, -1};
  idxs.bindExternalIndices(&st1, &st2, &st3, &st4);
  auto i = 0;
  for (auto const& row : idxs) {
    REQUIRE(row.has_distance());
    REQUIRE(row.has_point());
    if (row.has_flag()) {
      REQUIRE(row.flag().pointId() == row.pointId());
    }
    if (row.has_category()) {
      REQUIRE(row.category().pointId() == row.pointId());
    }
    REQUIRE(row.flagId() == fs[i]);
    REQUIRE(row.categoryId() == cs[i]);
    ++i;
  }
}

namespace extra_4
{
DECLARE_SOA_COLUMN_FULL(Bin, bin, int, "bin");
DECLARE_SOA_COLUMN_FULL(Color, color, int, "color");
} // namespace extra_4

DECLARE_SOA_TABLE(BinnedPoints, "TST", "BinnedPoints", Index<>, extra_4::Bin, indices::PointId);
DECLARE_SOA_TABLE(ColoredPoints, "TST", "ColoredPoints", Index<>, extra_4::Color, indices::PointId);

namespace indices
{
DECLARE_SOA_SLICE_INDEX_COLUMN(BinnedPoint, binsSlice);
DECLARE_SOA_ARRAY_INDEX_COLUMN(ColoredPoint, colorsList);
} // namespace indices

DECLARE_SOA_TABLE(IDX3s, "TST", "Index3", Index<>, indices::PointId, indices::BinnedPointIdSlice, indices::ColoredPointIds);

TEST_CASE("AdvancedIndexTables")
{
  TableBuilder b1;
  auto w1 = b1.cursor<Points>();
  for (auto i = 0; i < 10; ++i) {
    w1(0, i * 2., i * 3., i * 4.);
  }
  auto t1 = b1.finalize();
  Points st1{t1};

  TableBuilder b2;
  auto w2 = b2.cursor<BinnedPoints>();
  std::array<int, 3> skipPoints = {2, 6, 9};
  std::array<int, 10> sizes = {5, 3, 0, 12, 4, 1, 0, 8, 2, 0};
  auto count = 0;
  for (auto i = 0; i < 10; ++i) {
    if (i == skipPoints[count]) {
      ++count;
      continue;
    }
    for (auto j = 0; j < sizes[i]; ++j) {
      w2(0, j + 1, i);
    }
  }
  auto t2 = b2.finalize();
  BinnedPoints st2{t2};

  TableBuilder b3;
  auto w3 = b3.cursor<ColoredPoints>();
  std::array<int, 20> pointIds1 = {19, 2, 10, 5, 7, 17, 1, 3, 9, 12, 17, 6, 4, 13, 8, 5, 16, 15, 18, 0};
  std::array<int, 20> pointIds2 = {3, 19, 2, 6, 4, 13, 11, 5, 7, 11, 1, 9, 12, 17, 8, 14, 16, 2, 18, 0};
  std::array<int, 20> pointIds3 = {19, 2, 9, 15, 1, 3, 9, 12, 17, 18, 0, 10, 5, 7, 11, 6, 4, 13, 9, 14};
  for (int i = 0; i < 20; ++i) {
    w3(0, i, pointIds1[i]);
  }
  for (int i = 0; i < 20; ++i) {
    w3(0, i + 20, pointIds2[i]);
  }
  for (int i = 0; i < 20; ++i) {
    w3(0, i + 40, pointIds3[i]);
  }
  auto tc = b3.finalize();
  ColoredPoints st3{tc};

  std::array<int, 10> colorsizes = {3, 3, 4, 3, 3, 4, 3, 3, 2, 5};
  std::array<std::vector<int>, 10> colorvalues = {{{19, 39, 50},
                                                   {6, 30, 44},
                                                   {1, 22, 37, 41},
                                                   {7, 20, 45},
                                                   {12, 24, 56},
                                                   {3, 15, 27, 52},
                                                   {11, 23, 55},
                                                   {4, 28, 53},
                                                   {14, 34},
                                                   {8, 31, 42, 46, 58}}};

  auto t3 = IndexBuilder<Sparse>::indexBuilder<Points>("test4", {t1, t2, tc}, typename IDX3s::persistent_columns_t{}, o2::framework::pack<Points, BinnedPoints, ColoredPoints>{});
  REQUIRE(t3->num_rows() == st1.size());
  IDX3s idxs{t3};
  idxs.bindExternalIndices(&st1, &st2, &st3);
  count = 0;
  for (auto const& row : idxs) {
    REQUIRE(row.has_point());
    if (row.has_binsSlice()) {
      auto slice = row.binsSlice();
      REQUIRE(slice.size() == sizes[count]);
      for (auto const& bin : slice) {
        REQUIRE(bin.pointId() == row.pointId());
      }
    }
    auto colors = row.colorsList();
    REQUIRE(colors.size() == colorsizes[count]);
    for (auto j = 0; j < colors.size(); ++j) {
      REQUIRE(colors[j].color() == colorvalues[count][j]);
    }
    ++count;
  }
}
