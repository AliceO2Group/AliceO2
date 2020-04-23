// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework TableTreeConverter
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Framework/CommonDataProcessors.h"
#include "Framework/TableTreeHelpers.h"

#include <TTree.h>
#include <TRandom.h>
#include <arrow/table.h>

BOOST_AUTO_TEST_CASE(TreeToTableConversion)
{
  using namespace o2::framework;
  /// Create a simple TTree
  Int_t ndp = 1000;

  TFile f1("tree2table.root", "RECREATE");
  TTree t1("t1", "a simple Tree with simple variables");
  Bool_t ok;
  Float_t px, py, pz;
  Double_t random;
  Int_t ev;
  t1.Branch("ok", &ok, "ok/O");
  t1.Branch("px", &px, "px/F");
  t1.Branch("py", &py, "py/F");
  t1.Branch("pz", &pz, "pz/F");
  t1.Branch("random", &random, "random/D");
  t1.Branch("ev", &ev, "ev/I");

  //fill the tree
  int ntruein = 0;
  for (int i = 0; i < ndp; i++) {
    ok = (i % 2) == 0;
    if (ok) {
      ntruein++;
    }
    gRandom->Rannor(px, py);
    pz = px * px + py * py;
    random = gRandom->Rndm();
    ev = i + 1;
    t1.Fill();
  }
  t1.Write();
  auto br = (TBranch*)t1.GetBranch("ok");

  // Create an arrow table from this.
  TreeToTable tr2ta(&t1);
  auto stat = tr2ta.addAllColumns();
  if (!stat) {
    LOG(ERROR) << "Table was not created!";
    return;
  }
  auto table = tr2ta.process();

  // count number of rows with ok==true
  int ntrueout = 0;
  auto chunks = getBackendColumnData(table->column(0));
  BOOST_REQUIRE_NE(chunks.get(), nullptr);
  auto oks =
    std::dynamic_pointer_cast<arrow::BooleanArray>(chunks->chunk(0));
  BOOST_REQUIRE_NE(oks.get(), nullptr);
  for (int ii = 0; ii < table->num_rows(); ii++) {
    ntrueout += oks->Value(ii) ? 1 : 0;
  }

  // test result
  BOOST_REQUIRE_EQUAL(table->Validate().ok(), true);
  BOOST_REQUIRE_EQUAL(table->num_rows(), ndp);
  BOOST_REQUIRE_EQUAL(table->num_columns(), 6);
  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::boolean()->id());
  BOOST_REQUIRE_EQUAL(ntruein, ntrueout);
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(2)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(3)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table->column(4)->type()->id(), arrow::float64()->id());
  BOOST_REQUIRE_EQUAL(table->column(5)->type()->id(), arrow::int32()->id());

  f1.Close();

  TFile* f2 = new TFile("table2tree.root", "RECREATE");
  TableToTree ta2tr(table, f2, "mytree");
  stat = ta2tr.addAllBranches();
  auto t2 = ta2tr.process();
  br = (TBranch*)t2->GetBranch("ok");

  BOOST_REQUIRE_EQUAL(t2->GetEntries(), ndp);
  BOOST_REQUIRE_EQUAL(br->GetEntries(), ndp);

  f2->Close();
}
