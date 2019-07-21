// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework TableBuilder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Framework/TableBuilder.h"
#include "Framework/RootTableBuilderHelpers.h"
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

BOOST_AUTO_TEST_CASE(RootTree2Table)
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
  //fill the tree
  for (Int_t i = 0; i < 1000; i++) {
    gRandom->Rannor(xyz[0], xyz[1]);
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
  TTreeReaderArray<float> xyzReader(reader, "xyz");
  TTreeReaderArray<int> ijkReader(reader, "ij");
  TTreeReaderValue<float> pxReader(reader, "px");
  TTreeReaderValue<float> pyReader(reader, "py");
  TTreeReaderValue<float> pzReader(reader, "pz");
  TTreeReaderValue<double> randomReader(reader, "random");
  TTreeReaderValue<int> evReader(reader, "ev");

  RootTableBuilderHelpers::convertTTree(builder, reader, xyzReader, ijkReader, pxReader, pyReader, pzReader, randomReader, evReader);
  auto table = builder.finalize();
  BOOST_REQUIRE_EQUAL(table->num_rows(), 1000);
  BOOST_REQUIRE_EQUAL(table->num_columns(), 7);
  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::list(arrow::float32())->id());
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::list(arrow::int32())->id());
  BOOST_REQUIRE_EQUAL(table->column(2)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(3)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(4)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(5)->type()->id(), arrow::float64()->id());
  BOOST_REQUIRE_EQUAL(table->column(6)->type()->id(), arrow::int32()->id());

  {
    auto array = std::static_pointer_cast<arrow::ListArray>(table->column(0)->data()->chunk(0));
    BOOST_CHECK_EQUAL(array->value_length(0), 3);
  }
  {
    auto array = std::static_pointer_cast<arrow::ListArray>(table->column(1)->data()->chunk(0));
    auto values = std::static_pointer_cast<arrow::Int32Array>(array->values());
    BOOST_CHECK_EQUAL(array->value_length(0), 2);
    const int* ptr = values->raw_values() + values->offset();
    for (size_t i = 0; i < 1000; i++) {
      auto offset = array->value_offset(i);
      auto length = array->value_length(i);
      BOOST_CHECK_EQUAL(offset, i * 2);
      BOOST_CHECK_EQUAL(length, 2);
      BOOST_CHECK_EQUAL(*(ptr + offset), i);
      BOOST_CHECK_EQUAL(*(ptr + offset + 1), i + 1);
    }
  }
}
