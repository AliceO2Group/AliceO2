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

template class std::shared_ptr<arrow::Schema>;
template class std::shared_ptr<arrow::Column>;
template class std::vector<std::shared_ptr<arrow::Column>>;
template class std::shared_ptr<arrow::Array>;
template class std::vector<std::shared_ptr<arrow::Field>>;
template class std::shared_ptr<arrow::ChunkedArray>;
template class std::shared_ptr<arrow::Table>;
template class std::shared_ptr<arrow::Field>;

BOOST_AUTO_TEST_CASE(RootTree2Table)
{
  using namespace o2::framework;
  /// Create a simple TTree
  TTree t1("t1", "a simple Tree with simple variables");
  Float_t px, py, pz;
  Double_t random;
  Int_t ev;
  t1.Branch("px", &px, "px/F");
  t1.Branch("py", &py, "py/F");
  t1.Branch("pz", &pz, "pz/F");
  t1.Branch("random", &random, "random/D");
  t1.Branch("ev", &ev, "ev/I");
  //fill the tree
  for (Int_t i = 0; i < 1000; i++) {
    gRandom->Rannor(px, py);
    pz = px * px + py * py;
    random = gRandom->Rndm();
    ev = i;
    t1.Fill();
  }

  // Create an arrow table from this.
  TableBuilder builder;
  TTreeReader reader(&t1);
  TTreeReaderValue<float> pxReader(reader, "px");
  TTreeReaderValue<float> pyReader(reader, "py");
  TTreeReaderValue<float> pzReader(reader, "pz");
  TTreeReaderValue<double> randomReader(reader, "random");
  TTreeReaderValue<int> evReader(reader, "ev");

  RootTableBuilderHelpers::convertTTree(builder, reader, pxReader, pyReader, pzReader, randomReader, evReader);
  auto table = builder.finalize();
  BOOST_REQUIRE_EQUAL(table->num_rows(), 1000);
  BOOST_REQUIRE_EQUAL(table->num_columns(), 5);
  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(2)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(3)->type()->id(), arrow::float64()->id());
  BOOST_REQUIRE_EQUAL(table->column(4)->type()->id(), arrow::int32()->id());
}
