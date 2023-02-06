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

#define BOOST_TEST_MODULE Test Framework TableTreeConverter
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Framework/CommonDataProcessors.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"

#include <TTree.h>
#include <TRandom.h>
#include <arrow/table.h>
#include <array>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TreeToTableConversion)
{
  /// Create a simple TTree
  Int_t ndp = 17;

  TFile f1("tree2table.root", "RECREATE");
  TTree t1("t1", "a simple Tree with simple variables");
  Bool_t ok, ts[5] = {false};
  Float_t px, py, pz;
  Double_t random;
  Int_t ev;
  uint8_t b;
  const Int_t nelem = 9;
  Double_t ij[nelem] = {0};
  float xyzw[96];
  memset(xyzw, 1, 96 * 4);
  TString leaflist = Form("ij[%i]/D", nelem);

  Int_t ncols = 10;
  t1.Branch("ok", &ok, "ok/O");
  t1.Branch("px", &px, "px/F");
  t1.Branch("py", &py, "py/F");
  t1.Branch("pz", &pz, "pz/F");
  t1.Branch("random", &random, "random/D");
  t1.Branch("ev", &ev, "ev/I");
  t1.Branch("ij", ij, leaflist.Data());
  t1.Branch("tests", ts, "tests[5]/O");
  t1.Branch("xyzw", xyzw, "xyzw[96]/F");
  t1.Branch("small", &b, "small/b");

  //fill the tree
  int ntruein[2] = {0};
  for (int i = 0; i < ndp; i++) {
    ok = (i % 2) == 0;
    if (ok) {
      ntruein[0]++;
    }
    gRandom->Rannor(px, py);
    pz = px * px + py * py;
    random = gRandom->Rndm();
    ev = i + 1;
    b = i % 3;
    for (Int_t jj = 0; jj < nelem; jj++) {
      ij[jj] = i + 100 * jj;
    }
    for (Int_t jj = 0; jj < 5; jj++) {
      ts[jj] = (((i + jj) % 2) == 0);
      if (ts[jj]) {
        ntruein[1]++;
      }
    }

    t1.Fill();
  }
  t1.Write();

  // Create an arrow table from this.
  TreeToTable tr2ta;
  tr2ta.addAllColumns(&t1);
  tr2ta.fill(&t1);
  auto table = tr2ta.finalize();
  f1.Close();

  // test result
  BOOST_REQUIRE_EQUAL(table->Validate().ok(), true);
  BOOST_REQUIRE_EQUAL(table->num_rows(), ndp);
  BOOST_REQUIRE_EQUAL(table->num_columns(), ncols);

  BOOST_REQUIRE_EQUAL(table->column(0)->type()->id(), arrow::Type::BOOL);
  BOOST_REQUIRE_EQUAL(table->column(1)->type()->id(), arrow::Type::FLOAT);
  BOOST_REQUIRE_EQUAL(table->column(2)->type()->id(), arrow::Type::FLOAT);
  BOOST_REQUIRE_EQUAL(table->column(3)->type()->id(), arrow::Type::FLOAT);
  BOOST_REQUIRE_EQUAL(table->column(4)->type()->id(), arrow::Type::DOUBLE);
  BOOST_REQUIRE_EQUAL(table->column(5)->type()->id(), arrow::Type::INT32);
  BOOST_REQUIRE_EQUAL(table->column(6)->type()->id(), arrow::Type::FIXED_SIZE_LIST);
  BOOST_REQUIRE_EQUAL(table->column(7)->type()->id(), arrow::Type::FIXED_SIZE_LIST);
  BOOST_REQUIRE_EQUAL(table->column(8)->type()->id(), arrow::Type::FIXED_SIZE_LIST);
  BOOST_REQUIRE_EQUAL(table->column(9)->type()->id(), arrow::Type::UINT8);

  BOOST_REQUIRE(table->column(0)->type()->Equals(arrow::boolean()));
  BOOST_REQUIRE(table->column(1)->type()->Equals(arrow::float32()));
  BOOST_REQUIRE(table->column(2)->type()->Equals(arrow::float32()));
  BOOST_REQUIRE(table->column(3)->type()->Equals(arrow::float32()));
  BOOST_REQUIRE(table->column(4)->type()->Equals(arrow::float64()));
  BOOST_REQUIRE(table->column(5)->type()->Equals(arrow::int32()));
  BOOST_REQUIRE(table->column(6)->type()->Equals(arrow::fixed_size_list(arrow::float64(), nelem)));
  BOOST_REQUIRE(table->column(7)->type()->Equals(arrow::fixed_size_list(arrow::boolean(), 5)));
  BOOST_REQUIRE(table->column(8)->type()->Equals(arrow::fixed_size_list(arrow::float32(), 96)));
  BOOST_REQUIRE(table->column(9)->type()->Equals(arrow::uint8()));

  // count number of rows with ok==true
  int ntrueout = 0;
  auto chunks = table->column(0);
  BOOST_REQUIRE_NE(chunks.get(), nullptr);

  auto oks = std::dynamic_pointer_cast<arrow::BooleanArray>(chunks->chunk(0));
  BOOST_REQUIRE_NE(oks.get(), nullptr);

  for (int ii = 0; ii < table->num_rows(); ii++) {
    ntrueout += oks->Value(ii) ? 1 : 0;
  }
  BOOST_REQUIRE_EQUAL(ntruein[0], ntrueout);

  // count number of ts with ts==true
  chunks = table->column(7);
  BOOST_REQUIRE_NE(chunks.get(), nullptr);

  auto chunkToUse = std::static_pointer_cast<arrow::FixedSizeListArray>(chunks->chunk(0))->values();
  BOOST_REQUIRE_NE(chunkToUse.get(), nullptr);

  auto tests = std::dynamic_pointer_cast<arrow::BooleanArray>(chunkToUse);
  ntrueout = 0;
  for (int ii = 0; ii < table->num_rows() * 5; ii++) {
    ntrueout += tests->Value(ii) ? 1 : 0;
  }
  BOOST_REQUIRE_EQUAL(ntruein[1], ntrueout);

  // save table as tree
  TFile* f2 = TFile::Open("table2tree.root", "RECREATE");
  TableToTree ta2tr(table, f2, "mytree");
  ta2tr.addAllBranches();

  auto t2 = ta2tr.process();
  auto br = (TBranch*)t2->GetBranch("ok");
  BOOST_REQUIRE_EQUAL(t2->GetEntries(), ndp);
  BOOST_REQUIRE_EQUAL(br->GetEntries(), ndp);
  br = (TBranch*)t2->GetBranch("tests");
  BOOST_REQUIRE_EQUAL(br->GetEntries(), ndp);

  f2->Close();
}

namespace o2::aod
{
DECLARE_SOA_STORE();
namespace cols
{
DECLARE_SOA_COLUMN(Ivec, ivec, std::vector<int>);
DECLARE_SOA_COLUMN(Fvec, fvec, std::vector<float>);
DECLARE_SOA_COLUMN(Dvec, dvec, std::vector<double>);
DECLARE_SOA_COLUMN(UIvec, uivec, std::vector<uint8_t>);
} // namespace cols

DECLARE_SOA_TABLE(Vectors, "AOD", "VECS", o2::soa::Index<>, cols::Ivec, cols::Fvec, cols::Dvec, cols::UIvec);
} // namespace o2::aod

BOOST_AUTO_TEST_CASE(VariableLists)
{
  TableBuilder b;
  auto writer = b.cursor<o2::aod::Vectors>();
  std::vector<int> iv;
  std::vector<float> fv;
  std::vector<double> dv;
  std::vector<uint8_t> ui;

  std::array<int, 3> empty = {3, 7, 10};
  auto count = 0;
  for (auto i = 1; i < 1000; ++i) {
    iv.clear();
    fv.clear();
    dv.clear();
    ui.clear();
    if (count < empty.size() && i != empty[count]) {
      for (auto j = 0; j < i % 10 + 1; ++j) {
        iv.push_back(j + 2);
        fv.push_back((j + 2) * 0.2134f);
        dv.push_back((j + 4) * 0.192873819237);
        ui.push_back(j);
      }
    } else {
      count++;
    }
    writer(0, iv, fv, dv, ui);
  }
  auto table = b.finalize();

  auto* f = TFile::Open("variable_lists.root", "RECREATE");
  TableToTree ta2tr(table, f, "lists");
  ta2tr.addAllBranches();
  auto tree = ta2tr.process();
  f->Close();

  auto* f2 = TFile::Open("variable_lists.root", "READ");
  auto* treeptr = static_cast<TTree*>(f2->Get("lists;1"));
  TreeToTable tr2ta;
  tr2ta.addAllColumns(treeptr);
  tr2ta.fill(treeptr);
  auto ta = tr2ta.finalize();
  o2::aod::Vectors v{ta};
  int i = 1;
  count = 0;
  for (auto& row : v) {
    auto ivr = row.ivec();
    auto fvr = row.fvec();
    auto dvr = row.dvec();
    auto uvr = row.uivec();
    if (count < empty.size() && i != empty[count]) {
      for (auto j = 0; j < i % 10 + 1; ++j) {
        BOOST_CHECK_EQUAL(ivr[j], j + 2);
        BOOST_CHECK_EQUAL(fvr[j], (j + 2) * 0.2134f);
        BOOST_CHECK_EQUAL(dvr[j], (j + 4) * 0.192873819237);
        BOOST_CHECK_EQUAL(uvr[j], j);
      }
    } else {
      BOOST_CHECK_EQUAL(ivr.size(), 0);
      BOOST_CHECK_EQUAL(fvr.size(), 0);
      BOOST_CHECK_EQUAL(dvr.size(), 0);
      BOOST_CHECK_EQUAL(uvr.size(), 0);
      count++;
    }
    ++i;
  }
}
