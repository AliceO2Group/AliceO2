// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework GroupSlicer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ASoA.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"

#include <boost/test/unit_test.hpp>

using namespace o2;
using namespace o2::framework;

namespace o2::aod
{
namespace test
{
DECLARE_SOA_COLUMN(Foo, foo, float);
DECLARE_SOA_COLUMN(Bar, bar, float);
DECLARE_SOA_COLUMN(EventProperty, eventProperty, float);
} // namespace test
DECLARE_SOA_TABLE(Events, "AOD", "EVTS",
                  o2::soa::Index<>,
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
} // namespace o2::aod
BOOST_AUTO_TEST_CASE(GroupSlicerOneAssociated)
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, 0.5f * i, 2.f * i, 3.f * i);
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
  BOOST_CHECK_EQUAL(e.size(), 20);
  BOOST_CHECK_EQUAL(t.size(), 10 * 20);

  auto tt = std::make_tuple(t);
  o2::framework::AnalysisDataProcessorBuilder::GroupSlicer g(e, tt);

  unsigned int count = 0;
  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    BOOST_CHECK_EQUAL(gg.globalIndex(), count);
    auto trks = std::get<aod::TrksX>(as);
    BOOST_CHECK_EQUAL(trks.size(), 10);
    for (auto& trk : trks) {
      BOOST_CHECK_EQUAL(trk.eventId(), count);
    }
    ++count;
  }
}

BOOST_AUTO_TEST_CASE(GroupSlicerSeveralAssociated)
{
  TableBuilder builderE;
  auto evtsWriter = builderE.cursor<aod::Events>();
  for (auto i = 0; i < 20; ++i) {
    evtsWriter(0, 0.5f * i, 2.f * i, 3.f * i);
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

  BOOST_CHECK_EQUAL(e.size(), 20);
  BOOST_CHECK_EQUAL(tx.size(), 10 * 20);
  BOOST_CHECK_EQUAL(ty.size(), 20 * 20);
  BOOST_CHECK_EQUAL(tz.size(), 30 * 20);

  BOOST_CHECK_EQUAL(tu.size(), 10 * 20);

  auto tt = std::make_tuple(tx, ty, tz, tu);
  o2::framework::AnalysisDataProcessorBuilder::GroupSlicer g(e, tt);

  unsigned int count = 0;
  for (auto& slice : g) {
    auto as = slice.associatedTables();
    auto gg = slice.groupingElement();
    BOOST_CHECK_EQUAL(gg.globalIndex(), count);
    auto trksx = std::get<aod::TrksX>(as);
    auto trksy = std::get<aod::TrksY>(as);
    auto trksz = std::get<aod::TrksZ>(as);

    auto trksu = std::get<aod::TrksU>(as);

    BOOST_CHECK_EQUAL(trksx.size(), 10);
    BOOST_CHECK_EQUAL(trksy.size(), 20);
    BOOST_CHECK_EQUAL(trksz.size(), 30);

    BOOST_CHECK_EQUAL(trksu.size(), 10 * 20);

    for (auto& trk : trksx) {
      BOOST_CHECK_EQUAL(trk.eventId(), count);
    }
    for (auto& trk : trksy) {
      BOOST_CHECK_EQUAL(trk.eventId(), count);
    }
    for (auto& trk : trksz) {
      BOOST_CHECK_EQUAL(trk.eventId(), count);
    }

    ++count;
  }
}
