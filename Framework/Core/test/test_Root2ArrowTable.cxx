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

#include "Framework/TableBuilder.h"
#include "Framework/RootTableBuilderHelpers.h"
#include "Framework/ASoA.h"
#include "../src/ArrowDebugHelpers.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include <TTree.h>
#include <TRandom.h>
#include <arrow/table.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>

using namespace o2::framework;

TEST_CASE("RootTree2Table")
{
  using namespace o2::framework;
  /// Create a simple TTree
  TTree t1("t1", "a simple Tree with simple variables");
  Float_t xyz[3];
  Int_t ij[2];
  Float_t px, py, pz;
  Double_t random;
  Int_t ev;
  t1.Branch("px", &px, "px/F");
  t1.Branch("py", &py, "py/F");
  t1.Branch("pz", &pz, "pz/F");
  t1.Branch("random", &random, "random/D");
  t1.Branch("ev", &ev, "ev/I");
  t1.Branch("xyz", xyz, "xyz[3]/F");
  t1.Branch("ij", ij, "ij[2]/I");
  // fill the tree
  for (Int_t i = 0; i < 1000; i++) {
    xyz[0] = 1;
    xyz[1] = 2;
    xyz[2] = 3;
    gRandom->Rannor(px, py);
    pz = px * px + py * py;
    xyz[2] = i + 1;
    ij[0] = i;
    ij[1] = i + 1;
    random = gRandom->Rndm();
    ev = i + 1;
    t1.Fill();
  }

  // Create an arrow table from this.
  TableBuilder builder;
  TTreeReader reader(&t1);
  auto&& xyzReader = HolderMaker<float[3]>::make(reader, "xyz");
  auto&& ijkReader = HolderMaker<int[2]>::make(reader, "ij");
  auto&& pxReader = HolderMaker<float>::make(reader, "px");
  auto&& pyReader = HolderMaker<float>::make(reader, "py");
  auto&& pzReader = HolderMaker<float>::make(reader, "pz");
  auto&& randomReader = HolderMaker<double>::make(reader, "random");
  auto&& evReader = HolderMaker<int>::make(reader, "ev");

  RootTableBuilderHelpers::convertTTree(builder, reader, std::move(xyzReader), std::move(ijkReader), std::move(pxReader), std::move(pyReader), std::move(pzReader), std::move(randomReader), std::move(evReader));
  auto table = builder.finalize();
  REQUIRE(table->num_rows() == 1000);
  REQUIRE(table->num_columns() == 7);
  REQUIRE(table->schema()->field(0)->type()->id() == arrow::fixed_size_list(arrow::float32(), 3)->id());
  REQUIRE(table->schema()->field(1)->type()->id() == arrow::fixed_size_list(arrow::int32(), 2)->id());
  REQUIRE(table->schema()->field(2)->type()->id() == arrow::float32()->id());
  REQUIRE(table->schema()->field(3)->type()->id() == arrow::float32()->id());
  REQUIRE(table->schema()->field(4)->type()->id() == arrow::float32()->id());
  REQUIRE(table->schema()->field(5)->type()->id() == arrow::float64()->id());
  REQUIRE(table->schema()->field(6)->type()->id() == arrow::int32()->id());

  {
    auto chunkToUse = table->column(0)->chunk(0);
    chunkToUse = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
    auto array = std::static_pointer_cast<arrow::FloatArray>(chunkToUse);
    // array of 3 floats, time 1000.
    REQUIRE(array->length() == 3000);
    const float* c = reinterpret_cast<float const*>(array->values()->data());

    CHECK(c[0] == 1);
    CHECK(c[1] == 2);
    CHECK(c[2] == 1);
  }
  {
    auto chunkToUse = table->column(1)->chunk(0);
    chunkToUse = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
    auto array = std::static_pointer_cast<arrow::Int32Array>(chunkToUse);
    REQUIRE(array->length() == 2000);

    const int* ptr = reinterpret_cast<int const*>(array->values()->data());
    for (size_t i = 0; i < 1000; i++) {
      CHECK(ptr[2 * i + 0] == i);
      CHECK(ptr[2 * i + 1] == i + 1);
    }
  }
}

namespace o2::aod
{
DECLARE_SOA_VERSIONING();
namespace test
{
DECLARE_SOA_COLUMN_FULL(Px, px, float, "px");
DECLARE_SOA_COLUMN_FULL(Py, py, float, "py");
DECLARE_SOA_COLUMN_FULL(Pz, pz, float, "pz");
DECLARE_SOA_COLUMN_FULL(Xyz, xyz, float[3], "xyz");
DECLARE_SOA_COLUMN_FULL(Ij, ij, int[2], "ij");
DECLARE_SOA_COLUMN_FULL(Random, random, double, "random");
DECLARE_SOA_COLUMN_FULL(Ev, ev, int, "ev");
} // namespace test

DECLARE_SOA_TABLE(Test, "AOD", "ETAPHI",
                  test::Px, test::Py, test::Pz, test::Xyz, test::Ij,
                  test::Random, test::Ev);
} // namespace o2::aod

TEST_CASE("RootTree2TableViaASoA")
{
  using namespace o2::framework;
  /// Create a simple TTree
  TTree t2("t2", "a simple Tree with simple variables");
  Float_t xyz[3];
  Int_t ij[2];
  Float_t px, py, pz;
  Double_t random;
  Int_t ev;
  t2.Branch("px", &px, "px/F");
  t2.Branch("py", &py, "py/F");
  t2.Branch("pz", &pz, "pz/F");
  t2.Branch("random", &random, "random/D");
  t2.Branch("ev", &ev, "ev/I");
  t2.Branch("xyz", xyz, "xyz[3]/F");
  t2.Branch("ij", ij, "ij[2]/I");
  // fill the tree
  for (Int_t i = 0; i < 1000; i++) {
    gRandom->Rannor(xyz[0], xyz[1]);
    gRandom->Rannor(px, py);
    pz = px * px + py * py;
    xyz[2] = i + 1;
    ij[0] = i;
    ij[1] = i + 1;
    random = gRandom->Rndm();
    ev = i + 1;
    t2.Fill();
  }

  // Create an arrow table from this.
  TableBuilder builder;
  TTreeReader reader(&t2);
  REQUIRE(t2.GetEntries() == 1000);

  RootTableBuilderHelpers::convertASoA<o2::aod::Test>(builder, reader);
  auto table = builder.finalize();
  REQUIRE(table->num_rows() == 1000);
  REQUIRE(table->num_columns() == 7);
  REQUIRE(table->column(0)->type()->id() == arrow::float32()->id());
  REQUIRE(table->column(1)->type()->id() == arrow::float32()->id());
  REQUIRE(table->column(2)->type()->id() == arrow::float32()->id());
  REQUIRE(table->column(3)->type()->id() == arrow::fixed_size_list(arrow::float32(), 3)->id());
  REQUIRE(table->column(4)->type()->id() == arrow::fixed_size_list(arrow::int32(), 2)->id());
  REQUIRE(table->column(5)->type()->id() == arrow::float64()->id());
  REQUIRE(table->column(6)->type()->id() == arrow::int32()->id());

  o2::aod::Test testTable{table};
  for (auto& row : testTable) {
    REQUIRE(row.ij()[0] == row.ij()[1] - 1);
    REQUIRE(row.ij()[1] == row.ev());
  }
}
