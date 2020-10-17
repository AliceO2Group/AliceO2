// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework HistogramRegistry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/HistogramRegistry.h"
#include <boost/test/unit_test.hpp>

using namespace o2;
using namespace o2::framework;

namespace test
{
DECLARE_SOA_COLUMN_FULL(X, x, float, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, float, "y");
} // namespace test

HistogramRegistry foo()
{
  return {"r", true, {{"histo", "histo", {HistType::kTH1F, {{100, 0, 1}}}}}};
}

BOOST_AUTO_TEST_CASE(HistogramRegistryLookup)
{
  /// Construct a registry object with direct declaration
  HistogramRegistry registry{
    "registry", true, {
                        {"eta", "#Eta", {HistType::kTH1F, {{100, -2.0, 2.0}}}},                              //
                        {"phi", "#Phi", {HistType::kTH1D, {{102, 0, 2 * M_PI}}}},                            //
                        {"pt", "p_{T}", {HistType::kTH1D, {{1002, -0.01, 50.1}}}},                           //
                        {"ptToPt", "#ptToPt", {HistType::kTH2F, {{100, -0.01, 10.01}, {100, -0.01, 10.01}}}} //
                      }                                                                                      //
  };

  /// Get histograms by name
  BOOST_REQUIRE_EQUAL(registry.get<TH1>("eta")->GetNbinsX(), 100);
  BOOST_REQUIRE_EQUAL(registry.get<TH1>("phi")->GetNbinsX(), 102);
  BOOST_REQUIRE_EQUAL(registry.get<TH1>("pt")->GetNbinsX(), 1002);
  BOOST_REQUIRE_EQUAL(registry.get<TH2>("ptToPt")->GetNbinsX(), 100);
  BOOST_REQUIRE_EQUAL(registry.get<TH2>("ptToPt")->GetNbinsY(), 100);

  /// Get a pointer to the histogram
  auto histo = registry.get<TH1>("pt").get();
  BOOST_REQUIRE_EQUAL(histo->GetNbinsX(), 1002);

  /// Get registry object from a function
  auto r = foo();
  auto histo2 = r.get<TH1>("histo").get();
  BOOST_REQUIRE_EQUAL(histo2->GetNbinsX(), 100);
}

BOOST_AUTO_TEST_CASE(HistogramRegistryExpressionFill)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<float, float>({"x", "y"});
  rowWriterA(0, 0.0f, -2.0f);
  rowWriterA(0, 1.0f, -4.0f);
  rowWriterA(0, 2.0f, -1.0f);
  rowWriterA(0, 3.0f, -5.0f);
  rowWriterA(0, 4.0f, 0.0f);
  rowWriterA(0, 5.0f, -9.0f);
  rowWriterA(0, 6.0f, -7.0f);
  rowWriterA(0, 7.0f, -4.0f);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 8);
  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  TestA tests{tableA};
  BOOST_REQUIRE_EQUAL(8, tests.size());

  /// Construct a registry object with direct declaration
  HistogramRegistry registry{
    "registry", true, {
                        {"x", "test x", {HistType::kTH1F, {{100, 0.0f, 10.0f}}}},                            //
                        {"xy", "test xy", {HistType::kTH2F, {{100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}}} //
                      }                                                                                      //
  };

  /// Fill histogram with expression and table
  registry.fill<test::X>("x", tests, test::x > 3.0f);
  BOOST_CHECK_EQUAL(registry.get<TH1>("x")->GetEntries(), 4);

  /// Fill histogram with expression and table
  registry.fill<test::X, test::Y>("xy", tests, test::x > 3.0f && test::y > -5.0f);
  BOOST_CHECK_EQUAL(registry.get<TH2>("xy")->GetEntries(), 2);
}
