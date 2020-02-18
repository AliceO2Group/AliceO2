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

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TreeToTableConversion)
{
  using namespace o2::framework;
  /// Create a simple TTree
  Int_t ndp = 10000000;

  TFile f1("tree2table.root", "RECREATE");
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
  for (Int_t i = 0; i < ndp; i++) {
    gRandom->Rannor(px, py);
    pz = px * px + py * py;
    random = gRandom->Rndm();
    ev = i + 1;
    t1.Fill();
  }

  // Create an arrow table from this.
  TreeToTable tr2ta(&t1);
  auto stat = tr2ta.AddAllColumns();
  if (!stat) {
    LOG(INFO) << "Table was not created!";
    return;
  }
  auto table = tr2ta.Process();

  // test result
  BOOST_REQUIRE_EQUAL(table.get()->Validate().ok(), true);
  BOOST_REQUIRE_EQUAL(table.get()->num_rows(), ndp);
  BOOST_REQUIRE_EQUAL(table.get()->num_columns(), 5);
  BOOST_REQUIRE_EQUAL(table.get()->column(0)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table.get()->column(1)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table.get()->column(2)->type()->id(), arrow::float32()->id());
  BOOST_REQUIRE_EQUAL(table.get()->column(3)->type()->id(), arrow::float64()->id());
  BOOST_REQUIRE_EQUAL(table.get()->column(4)->type()->id(), arrow::int32()->id());

  f1.Close();
}
