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

#include "Framework/HistogramRegistry.h"
#include <catch_amalgamated.hpp>
#include <iostream>

using namespace o2;
using namespace o2::framework;

namespace test
{
DECLARE_SOA_COLUMN_FULL(X, x, float, "x");
DECLARE_SOA_COLUMN_FULL(Y, y, float, "y");
} // namespace test

HistogramRegistry foo()
{
  return {"r", {{"histo", "histo", {HistType::kTH1F, {{100, 0, 1}}}}}};
}

TEST_CASE("HistogramRegistryLookup")
{
  /// Construct a registry object with direct declaration
  HistogramRegistry registry{
    "registry", {
                  {"eta", "#Eta", {HistType::kTH1F, {{100, -2.0, 2.0}}}},                              //
                  {"phi", "#Phi", {HistType::kTH1D, {{102, 0, 2 * M_PI}}}},                            //
                  {"pt", "p_{T}", {HistType::kTH1D, {{1002, -0.01, 50.1}}}},                           //
                  {"ptToPt", "#ptToPt", {HistType::kTH2F, {{100, -0.01, 10.01}, {100, -0.01, 10.01}}}} //
                }                                                                                      //
  };

  /// Get histograms by name
  REQUIRE(registry.get<TH1>(HIST("eta"))->GetNbinsX() == 100);
  REQUIRE(registry.get<TH1>(HIST("phi"))->GetNbinsX() == 102);
  REQUIRE(registry.get<TH1>(HIST("pt"))->GetNbinsX() == 1002);
  REQUIRE(registry.get<TH2>(HIST("ptToPt"))->GetNbinsX() == 100);
  REQUIRE(registry.get<TH2>(HIST("ptToPt"))->GetNbinsY() == 100);

  /// Get a pointer to the histogram
  auto histo = registry.get<TH1>(HIST("pt")).get();
  REQUIRE(histo->GetNbinsX() == 1002);

  /// Get registry object from a function
  auto r = foo();
  auto histo2 = r.get<TH1>(HIST("histo")).get();
  REQUIRE(histo2->GetNbinsX() == 100);

  registry.print();

  // check that registry behaves correctly when two different names have equal hash:
  /*
  auto str1 = "Maria has nine red beds.";
  auto str2 = "Steven has fifteen white tables.";
  REQUIRE(compile_time_hash(str1) ==  compile_time_hash(str2));;
  try {
    registry.add(str1, "", kTH1F, {{20, 0.0f, 10.01f}});
    registry.add(str2, "", kTH1F, {{20, 0.0f, 10.01f}});
  } catch (...) {
    std::cout << "Hash collision was detected correctly!" << std::endl;
  }
  */
}

TEST_CASE("HistogramRegistryExpressionFill")
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
  REQUIRE(tableA->num_rows() == 8);
  using TestA = o2::soa::Table<o2::soa::Index<>, test::X, test::Y>;
  TestA tests{tableA};
  REQUIRE(8 == tests.size());

  /// Construct a registry object with direct declaration
  HistogramRegistry registry{
    "registry", {
                  {"x", "test x", {HistType::kTH1F, {{100, 0.0f, 10.0f}}}},                            //
                  {"xy", "test xy", {HistType::kTH2F, {{100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}}} //
                }                                                                                      //
  };

  /// Fill histogram with expression and table
  registry.fill<test::X>(HIST("x"), tests, test::x > 3.0f);
  REQUIRE(registry.get<TH1>(HIST("x"))->GetEntries() == 4);

  /// Fill histogram with expression and table
  registry.fill<test::X, test::Y>(HIST("xy"), tests, test::x > 3.0f && test::y > -5.0f);
  REQUIRE(registry.get<TH2>(HIST("xy"))->GetEntries() == 2);
}

TEST_CASE("HistogramRegistryStepTHn")
{
  HistogramRegistry registry{"registry"};

  registry.add("stepTHnF", "a", {kStepTHnF, {{100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}, 2});
  registry.add("stepTHnD", "b", {kStepTHnD, {{100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}, 3});
  registry.addClone("stepTHnD", "stepTHnD2");

  auto histo = registry.get<StepTHn>(HIST("stepTHnD"));
  REQUIRE(histo->getNSteps() == 3);

  // fill first step at position (0,3)
  registry.fill(HIST("stepTHnF"), 0, 0., 3.);
  // fill second step (0,4)
  registry.fill(HIST("stepTHnF"), 1, 0., 4.);

  registry.fill(HIST("stepTHnD2"), 1, 0., 4.);

  registry.print();
}
