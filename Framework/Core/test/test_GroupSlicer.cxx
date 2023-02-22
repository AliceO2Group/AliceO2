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

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include <arrow/util/config.h>

#include <catch_amalgamated.hpp>

using namespace o2;
using namespace o2::framework;

namespace o2::aod
{
namespace test
{
DECLARE_SOA_COLUMN(ID, id, int);
DECLARE_SOA_COLUMN(Foo, foo, float);
DECLARE_SOA_COLUMN(Bar, bar, float);
DECLARE_SOA_COLUMN(EventProperty, eventProperty, float);
} // namespace test
DECLARE_SOA_TABLE(Events, "AOD", "EVTS",
                  o2::soa::Index<>,
                  test::ID,
                  test::EventProperty,
                  test::Foo,
                  test::Bar);

using Event = Events::iterator;

namespace test
{
DECLARE_SOA_INDEX_COLUMN(Event, event);
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);
} // namespace test

namespace unsorted
{
DECLARE_SOA_INDEX_COLUMN(Event, event);
}

DECLARE_SOA_TABLE(TrksX, "AOD", "TRKSX",
                  test::EventId,
                  test::X);
DECLARE_SOA_TABLE(TrksY, "AOD", "TRKSY",
                  test::EventId,
                  test::Y);
DECLARE_SOA_TABLE(TrksZ, "AOD", "TRKSZ",
                  test::EventId,
                  test::Z);
DECLARE_SOA_TABLE(TrksU, "AOD", "TRKSU",
                  test::X,
                  test::Y,
                  test::Z);

DECLARE_SOA_TABLE(TrksXU, "AOD", "TRKSX",
                  unsorted::EventId,
                  test::X);
DECLARE_SOA_TABLE(TrksYU, "AOD", "TRKSY",
                  unsorted::EventId,
                  test::Y);
DECLARE_SOA_TABLE(TrksZU, "AOD", "TRKSZ",
                  unsorted::EventId,
                  test::Z);

namespace test
{
DECLARE_SOA_COLUMN(Arr, arr, float[3]);
DECLARE_SOA_COLUMN(Boo, boo, bool);
DECLARE_SOA_COLUMN(Lst, lst, std::vector<double>);
} // namespace test

DECLARE_SOA_TABLE(EventExtra, "AOD", "EVTSXTRA", test::Arr, test::Boo, test::Lst);

} // namespace o2::aod
TEST_CASE("GroupSlicerOneAssociated")
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, i, 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  TableBuilder builderT;
  auto trksWriter = builderT.cursor<aod::TrksX>();
  for (auto i = 0; i < 20; ++i) {
    for (auto j = 0.f; j < 5; j += 0.5f) {
      trksWriter(0, i, 0.5f * j);
    }
  }
  auto trkTable = builderT.finalize();
  aod::Events e{evtTable};
  aod::TrksX t{trkTable};
  REQUIRE(e.size() == 20);
  REQUIRE(t.size() == 10 * 20);

  auto tt = std::make_tuple(t);
  o2::framework::GroupSlicer g(e, tt);

  unsigned int count = 0;
  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == count);
    auto trks = std::get<aod::TrksX>(as);
    REQUIRE(trks.size() == 10);
    for (auto& trk : trks) {
      REQUIRE(trk.eventId() == count);
    }
    ++count;
  }
}

TEST_CASE("GroupSlicerSeveralAssociated")
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, i, 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  TableBuilder builderTX;
  auto trksWriterX = builderTX.cursor<aod::TrksX>();
  TableBuilder builderTY;
  auto trksWriterY = builderTY.cursor<aod::TrksY>();
  TableBuilder builderTZ;
  auto trksWriterZ = builderTZ.cursor<aod::TrksZ>();

  TableBuilder builderTXYZ;
  auto trksWriterXYZ = builderTXYZ.cursor<aod::TrksU>();

  for (auto i = 0; i < 20; ++i) {
    for (auto j = 0.f; j < 5; j += 0.5f) {
      trksWriterX(0, i, 0.5f * j);
    }
    for (auto j = 0.f; j < 10; j += 0.5f) {
      trksWriterY(0, i, 0.5f * j);
    }
    for (auto j = 0.f; j < 15; j += 0.5f) {
      trksWriterZ(0, i, 0.5f * j);
    }

    for (auto j = 0.f; j < 5; j += 0.5f) {
      trksWriterXYZ(0, 0.5f * j, 2.f * j, 2.5f * j);
    }
  }
  auto trkTableX = builderTX.finalize();
  auto trkTableY = builderTY.finalize();
  auto trkTableZ = builderTZ.finalize();

  auto trkTableXYZ = builderTXYZ.finalize();

  aod::Events e{evtTable};
  aod::TrksX tx{trkTableX};
  aod::TrksY ty{trkTableY};
  aod::TrksZ tz{trkTableZ};

  aod::TrksU tu{trkTableXYZ};

  REQUIRE(e.size() == 20);
  REQUIRE(tx.size() == 10 * 20);
  REQUIRE(ty.size() == 20 * 20);
  REQUIRE(tz.size() == 30 * 20);

  REQUIRE(tu.size() == 10 * 20);

  auto tt = std::make_tuple(tx, ty, tz, tu);
  o2::framework::GroupSlicer g(e, tt);

  unsigned int count = 0;
  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == count);
    auto trksx = std::get<aod::TrksX>(as);
    auto trksy = std::get<aod::TrksY>(as);
    auto trksz = std::get<aod::TrksZ>(as);

    auto trksu = std::get<aod::TrksU>(as);

    REQUIRE(trksx.size() == 10);
    REQUIRE(trksy.size() == 20);
    REQUIRE(trksz.size() == 30);

    REQUIRE(trksu.size() == 10 * 20);

    for (auto& trk : trksx) {
      REQUIRE(trk.eventId() == count);
    }
    for (auto& trk : trksy) {
      REQUIRE(trk.eventId() == count);
    }
    for (auto& trk : trksz) {
      REQUIRE(trk.eventId() == count);
    }

    ++count;
  }
}

TEST_CASE("GroupSlicerMismatchedGroups")
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, i, 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  TableBuilder builderT;
  auto trksWriter = builderT.cursor<aod::TrksX>();
  for (auto i = 0; i < 20; ++i) {
    if (i == 3 || i == 10 || i == 12 || i == 16 || i == 19) {
      continue;
    }
    for (auto j = 0.f; j < 5; j += 0.5f) {
      trksWriter(0, i, 0.5f * j);
    }
  }
  auto trkTable = builderT.finalize();
  aod::Events e{evtTable};
  aod::TrksX t{trkTable};
  REQUIRE(e.size() == 20);
  REQUIRE(t.size() == 10 * (20 - 5));

  auto tt = std::make_tuple(t);
  o2::framework::GroupSlicer g(e, tt);

  unsigned int count = 0;
  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == count);
    auto trks = std::get<aod::TrksX>(as);
    if (count == 3 || count == 10 || count == 12 || count == 16 || count == 19) {
      REQUIRE(trks.size() == 0);
    } else {
      REQUIRE(trks.size() == 10);
    }
    for (auto& trk : trks) {
      REQUIRE(trk.eventId() == count);
    }
    ++count;
  }
}

TEST_CASE("GroupSlicerMismatchedUnassignedGroups")
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, i, 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  int skip = 0;

  TableBuilder builderT;
  auto trksWriter = builderT.cursor<aod::TrksX>();
  for (auto i = 0; i < 20; ++i) {
    if (i == 3 || i == 10 || i == 12 || i == 16 || i == 19) {
      for (auto iz = 0; iz < 5; ++iz) {
        trksWriter(0, -1 - skip, 0.123f * iz + 1.654f);
      }
      ++skip;
      continue;
    }
    for (auto j = 0.f; j < 5; j += 0.5f) {
      trksWriter(0, i, 0.5f * j);
    }
  }
  for (auto i = 0; i < 5; ++i) {
    trksWriter(0, -6, 0.123f * i + 1.654f);
  }
  auto trkTable = builderT.finalize();

  aod::Events e{evtTable};
  aod::TrksX t{trkTable};
  REQUIRE(e.size() == 20);
  REQUIRE(t.size() == (30 + 10 * (20 - 5)));

  auto tt = std::make_tuple(t);
  o2::framework::GroupSlicer g(e, tt);

  unsigned int count = 0;
  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == count);
    auto trks = std::get<aod::TrksX>(as);
    if (count == 3 || count == 10 || count == 12 || count == 16 || count == 19) {
      REQUIRE(trks.size() == 0);
    } else {
      REQUIRE(trks.size() == 10);
    }
    for (auto& trk : trks) {
      REQUIRE(trk.eventId() == count);
    }
    ++count;
  }
}

TEST_CASE("GroupSlicerMismatchedFilteredGroups")
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, i, 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  TableBuilder builderT;
  auto trksWriter = builderT.cursor<aod::TrksX>();
  for (auto i = 0; i < 20; ++i) {
    if (i == 3 || i == 10 || i == 12 || i == 16) {
      continue;
    }
    for (auto j = 0.f; j < 5; j += 0.5f) {
      trksWriter(0, i, 0.5f * j);
    }
  }
  auto trkTable = builderT.finalize();
  using FilteredEvents = soa::Filtered<aod::Events>;
  soa::SelectionVector rows{2, 4, 10, 9, 15};
  FilteredEvents e{{evtTable}, {2, 4, 10, 9, 15}};
  aod::TrksX t{trkTable};
  REQUIRE(e.size() == 5);
  REQUIRE(t.size() == 10 * (20 - 4));

  auto tt = std::make_tuple(t);
  o2::framework::GroupSlicer g(e, tt);

  unsigned int count = 0;

  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == rows[count]);
    auto trks = std::get<aod::TrksX>(as);
    if (rows[count] == 3 || rows[count] == 10 || rows[count] == 12 || rows[count] == 16) {
      REQUIRE(trks.size() == 0);
    } else {
      REQUIRE(trks.size() == 10);
    }
    for (auto& trk : trks) {
      REQUIRE(trk.eventId() == rows[count]);
    }
    ++count;
  }
}

TEST_CASE("GroupSlicerMismatchedUnsortedFilteredGroups")
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, i, 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  TableBuilder builderT;
  auto trksWriter = builderT.cursor<aod::TrksXU>();
  std::vector<int> randomized{10, 2, 1, 0, 15, 3, 6, 4, 14, 5, 7, 9, 8, 19, 11, 13, 17, 12, 18, 19};
  std::vector<int64_t> sel;
  sel.resize(10 * (20 - 4));
  std::iota(sel.begin(), sel.end(), 0);
  for (auto i : randomized) {
    if (i == 3 || i == 10 || i == 12 || i == 16) {
      continue;
    }
    for (auto j = 0.f; j < 5; j += 0.5f) {
      trksWriter(0, i, 0.5f * j);
    }
  }
  auto trkTable = builderT.finalize();

  TableBuilder builderTE;
  auto trksWriterE = builderTE.cursor<aod::TrksXU>();
  auto trkTableE = builderTE.finalize();

  using FilteredEvents = soa::Filtered<aod::Events>;
  soa::SelectionVector rows{2, 4, 10, 9, 15};
  FilteredEvents e{{evtTable}, {2, 4, 10, 9, 15}};
  soa::SmallGroups<aod::TrksXU> t{{trkTable}, std::move(sel)};

  REQUIRE(e.size() == 5);
  REQUIRE(t.size() == 10 * (20 - 4));

  auto tt = std::make_tuple(t);

  o2::framework::GroupSlicer g(e, tt);

  unsigned int count = 0;

  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == rows[count]);
    auto trks = std::get<soa::SmallGroups<aod::TrksXU>>(as);
    if (rows[count] == 3 || rows[count] == 10 || rows[count] == 12 || rows[count] == 16) {
      REQUIRE(trks.size() == 0);
    } else {
      REQUIRE(trks.size() == 10);
    }
    for (auto& trk : trks) {
      REQUIRE(trk.eventId() == rows[count]);
    }
    ++count;
  }

  std::vector<int64_t> sele;
  soa::SmallGroups<aod::TrksXU> te{{trkTableE}, std::move(sele)};
  auto tte = std::make_tuple(te);
  o2::framework::GroupSlicer ge(e, tte);

  count = 0;
  for (auto& slice : ge) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == rows[count]);
    auto trks = std::get<soa::SmallGroups<aod::TrksXU>>(as);
    REQUIRE(trks.size() == 0);
    ++count;
  }

  soa::SmallGroupsUnfiltered<aod::TrksXU> tu{{trkTable}, std::vector<int64_t>{}};
  auto ttu = std::make_tuple(tu);
  o2::framework::GroupSlicer gu(e, ttu);

  count = 0;
  for (auto& slice : gu) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    REQUIRE(gg.globalIndex() == rows[count]);
    auto trks = std::get<soa::SmallGroupsUnfiltered<aod::TrksXU>>(as);
    if (rows[count] == 3 || rows[count] == 10 || rows[count] == 12 || rows[count] == 16) {
      REQUIRE(trks.size() == 0);
    } else {
      REQUIRE(trks.size() == 10);
    }
    ++count;
  }
}

TEST_CASE("EmptySliceables")
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, i, 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  TableBuilder builderT;
  auto trksWriter = builderT.cursor<aod::TrksX>();
  auto trkTable = builderT.finalize();

  aod::Events e{evtTable};
  aod::TrksX t{trkTable};
  REQUIRE(e.size() == 20);
  REQUIRE(t.size() == 0);

  auto tt = std::make_tuple(t);
  o2::framework::GroupSlicer g(e, tt);

  unsigned int count = 0;
  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    auto trks = std::get<aod::TrksX>(as);
    REQUIRE(gg.globalIndex() == count);
    REQUIRE(trks.size() == 0);
    ++count;
  }
}

TEST_CASE("ArrowDirectSlicing")
{
  int counts[] = {5, 5, 5, 4, 1};
  int offsets[] = {0, 5, 10, 15, 19, 20};
  int ids[] = {0, 1, 2, 3, 4};
  int sizes[] = {4, 1, 12, 5, 2};

  using BigE = soa::Join<aod::Events, aod::EventExtra>;

  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  auto step = 0;
  for (auto i = 0; i < 20; ++i) {
    if (i >= offsets[step + 1]) {
      ++step;
    }
    evtsWriter(0, ids[step], 0.5f * i, 2.f * i, 3.f * i);
  }
  auto evtTable = builderE.finalize();

  TableBuilder builderEE;
  auto evtsEWriter = builderEE.cursor<aod::EventExtra>();
  step = 0;

  for (auto i = 0; i < 20; ++i) {
    if (i >= offsets[step + 1]) {
      ++step;
    }
    float arr[3] = {0.1f * i, 0.2f * i, 0.3f * i};
    std::vector<double> d;
    for (auto z = 0; z < sizes[step]; ++z) {
      d.push_back((double)z * 0.5);
    }
    evtsEWriter(0, arr, i % 2 == 0, d);
  }
  auto evtETable = builderEE.finalize();

  aod::Events e{evtTable};
  aod::EventExtra ee{evtETable};
  BigE b_e{{evtTable, evtETable}};

  std::vector<std::shared_ptr<arrow::ChunkedArray>> slices_array;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> slices_bool;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> slices_vec;
  auto offset = 0;
  for (auto i = 0u; i < 5; ++i) {
    slices_array.emplace_back(evtETable->column(0)->Slice(offset, counts[i]));
    slices_bool.emplace_back(evtETable->column(1)->Slice(offset, counts[i]));
    slices_vec.emplace_back(evtETable->column(2)->Slice(offset, counts[i]));
    offset += counts[i];
    REQUIRE(slices_array[i]->length() == counts[i]);
    REQUIRE(slices_bool[i]->length() == counts[i]);
    REQUIRE(slices_vec[i]->length() == counts[i]);
  }

  std::vector<arrow::Datum> slices;
  std::vector<uint64_t> offsts;
  auto status = sliceByColumn("fID", "BigE", b_e.asArrowTable(), 20, &slices, &offsts);
  for (auto i = 0u; i < 5; ++i) {
#if ARROW_VERSION_MAJOR < 10
    auto tbl = arrow::util::get<std::shared_ptr<arrow::Table>>(slices[i].value);
#else
    auto tbl = slices[i].table();
#endif
    auto ca = tbl->GetColumnByName("fArr");
    auto cb = tbl->GetColumnByName("fBoo");
    auto cv = tbl->GetColumnByName("fLst");
    REQUIRE(ca->length() == counts[i]);
    REQUIRE(cb->length() == counts[i]);
    REQUIRE(cv->length() == counts[i]);
    REQUIRE(ca->Equals(slices_array[i]));
    REQUIRE(cb->Equals(slices_bool[i]));
    REQUIRE(cv->Equals(slices_vec[i]));
  }

  int j = 0u;
  for (auto i = 0u; i < 5; ++i) {
#if ARROW_VERSION_MAJOR < 10
    auto tbl = BigE::table_t{arrow::util::get<std::shared_ptr<arrow::Table>>(slices[i].value), static_cast<uint64_t>(offsts[i])};
#else
    auto tbl = BigE::table_t{slices[i].table(), static_cast<uint64_t>(offsts[i])};
#endif
    REQUIRE(tbl.size() == counts[i]);
    for (auto& row : tbl) {
      REQUIRE(row.id() == ids[i]);
      REQUIRE(row.boo() == (j % 2 == 0));
      auto rid = row.globalIndex();
      auto arr = row.arr();
      REQUIRE(arr[0] == 0.1f * (float)rid);
      REQUIRE(arr[1] == 0.2f * (float)rid);
      REQUIRE(arr[2] == 0.3f * (float)rid);

      auto d = row.lst();
      REQUIRE(d.size() == sizes[i]);
      for (auto z = 0u; z < d.size(); ++z) {
        REQUIRE(d[z] == 0.5 * (double)z);
      }
      ++j;
    }
  }
}
