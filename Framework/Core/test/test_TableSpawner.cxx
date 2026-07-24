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

#include <catch_amalgamated.hpp>
#include "Framework/AnalysisHelpers.h"

#include <Framework/AnalysisDataModel.h>
#include <Framework/TableBuilder.h>

using namespace o2::framework;
using namespace o2::soa;
using namespace o2::aod;

namespace o2::aod
{
namespace test
{
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);
DECLARE_SOA_EXPRESSION_COLUMN(Rsq, rsq, float, test::x* test::x + test::y * test::y + test::z * test::z);
DECLARE_SOA_EXPRESSION_COLUMN(Sin, sin, float, test::x / nsqrt(test::x * test::x + test::y * test::y));

DECLARE_SOA_CONFIGURABLE_EXPRESSION_COLUMN(Cfg, cfg, float, "configurable");
} // namespace test

DECLARE_SOA_TABLE(Points, "AOD", "PTSNG", test::X, test::Y, test::Z);
DECLARE_SOA_EXTENDED_TABLE(ExPoints, Points, "EXPTSNG", 0, test::Rsq, test::Sin);

DECLARE_SOA_CONFIGURABLE_EXTENDED_TABLE(ExcPoints, Points, "CFGPTS", test::Cfg);
} // namespace o2::aod

TEST_CASE("TestTableSpawner")
{
  TableBuilder b1;
  auto w1 = b1.cursor<Points>();

  for (auto i = 1; i < 10; ++i) {
    w1(0, i * 2., i * 3., i * 4.);
  }

  auto t1 = b1.finalize();
  Points st1{t1};

  auto expoints_a = o2::soa::Extend<o2::aod::Points, test::Rsq, test::Sin>(st1);
  Spawns<ExPoints> s;
  auto extension = ExPointsExtension{o2::framework::spawner<o2::aod::Hash<"EXPTSNG/0"_h>>(t1, o2::aod::Hash<"ExPoints"_h>::str, s.projectors.data(), s.projector, s.schema)};
  auto expoints = ExPoints{{t1, extension.asArrowTable()}};

  REQUIRE(expoints_a.size() == 9);
  REQUIRE(extension.size() == 9);
  REQUIRE(expoints.size() == 9);

  auto rex = extension.begin();
  auto rexp = expoints.begin();
  auto rexp_a = expoints_a.begin();

  for (auto i = 1; i < 10; ++i) {
    float rsq = i * i * 4 + i * i * 9 + i * i * 16;
    float sin = i * 2 / std::sqrt(i * i * 4 + i * i * 9);
    REQUIRE(rexp_a.rsq() == rsq);
    REQUIRE(rex.rsq() == rsq);
    REQUIRE(rexp.rsq() == rsq);
    REQUIRE(rexp_a.sin() == sin);
    REQUIRE(rex.sin() == sin);
    REQUIRE(rexp.sin() == sin);
    ++rex;
    ++rexp;
    ++rexp_a;
  }

  Defines<ExcPoints> excpts;
  excpts.projectors[0] = test::x * test::x + test::y * test::y + test::z * test::z;

  auto extension_2 = ExcPointsCfgExtension{o2::framework::spawner<o2::aod::Hash<"EXCFGPTS/0"_h>>({t1}, o2::aod::Hash<"ExcPoints"_h>::str, excpts.projectors.data(), excpts.projector, excpts.schema)};
  auto excpoints = ExcPoints{{t1, extension_2.asArrowTable()}};

  rex = extension.begin();
  auto rex_2 = extension_2.begin();
  auto rexcp = excpoints.begin();

  for (auto i = 1; i < 10; ++i) {
    float rsq = i * i * 4 + i * i * 9 + i * i * 16;
    REQUIRE(rex.rsq() == rsq);
    REQUIRE(rex_2.cfg() == rsq);
    REQUIRE(rexcp.cfg() == rsq);
    ++rex;
    ++rex_2;
    ++rexcp;
  }
}
