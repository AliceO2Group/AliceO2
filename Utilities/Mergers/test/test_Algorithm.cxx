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

/// \file test_Algorithm.cxx
/// \brief A unit test of mergers
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <boost/test/tools/interface.hpp>
#include <gsl/span>
#include <memory>
#define BOOST_TEST_MODULE Test Utilities MergerAlgorithm
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Mergers/MergerAlgorithm.h"
#include "Mergers/CustomMergeableTObject.h"
#include "Mergers/CustomMergeableObject.h"
#include "Mergers/ObjectStore.h"

#include <TObjArray.h>
#include <TObjString.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <TTree.h>
#include <THnSparse.h>
#include <TF1.h>
#include <TGraph.h>
#include <TProfile.h>

// using namespace o2::framework;
using namespace o2::mergers;

constexpr size_t bins = 10;
constexpr size_t min = 0;
constexpr size_t max = 10;

BOOST_AUTO_TEST_CASE(MergerEmptyObjects)
{
  {
    TH1I* target = nullptr;
    TH1I* other = new TH1I("obj1", "obj1", bins, min, max);
    BOOST_CHECK_THROW(algorithm::merge(target, other), std::runtime_error);
    delete other;
  }
  {
    TH1I* target = new TH1I("obj1", "obj1", bins, min, max);
    TH1I* other = nullptr;
    BOOST_CHECK_THROW(algorithm::merge(target, other), std::runtime_error);
    delete target;
  }
}

BOOST_AUTO_TEST_CASE(MergerNotSupportedTObject)
{
  TObjString* target = new TObjString("foo");
  TObjString* other = new TObjString("bar");
  algorithm::merge(target, other);
  BOOST_CHECK(target->GetString() == "foo");
  delete target;
  delete other;
}

BOOST_AUTO_TEST_CASE(MergerTheSameObject)
{
  TH1I* object = new TH1I("obj1", "obj1", bins, min, max);
  BOOST_CHECK_THROW(algorithm::merge(object, object), std::runtime_error);
  delete object;
}

BOOST_AUTO_TEST_CASE(MergerSingularObjects)
{
  {
    TH1I* target = new TH1I("obj1", "obj1", bins, min, max);
    target->Fill(5);
    TH1I* other = new TH1I("obj2", "obj2", bins, min, max);
    other->Fill(2);
    other->Fill(2);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(2)), 2);
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(5)), 1);

    delete other;
    delete target;
  }
  {
    TH2I* target = new TH2I("obj1", "obj1", bins, min, max, bins, min, max);
    target->Fill(5, 5);
    TH2I* other = new TH2I("obj2", "obj2", bins, min, max, bins, min, max);
    other->Fill(2, 2);
    other->Fill(2, 2);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(2, 2)), 2);
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(5, 5)), 1);

    delete other;
    delete target;
  }
  {
    TH3I* target = new TH3I("obj1", "obj1", bins, min, max, bins, min, max, bins, min, max);
    target->Fill(5, 5, 5);
    TH3I* other = new TH3I("obj2", "obj2", bins, min, max, bins, min, max, bins, min, max);
    other->Fill(2, 2, 2);
    other->Fill(2, 2, 2);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(2, 2, 2)), 2);
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(5, 5, 5)), 1);

    delete other;
    delete target;
  }
  {
    const size_t dim = 5;
    const Int_t binsDims[dim] = {bins, bins, bins, bins, bins};
    const Double_t mins[dim] = {min, min, min, min, min};
    const Double_t maxs[dim] = {max, max, max, max, max};

    THnI* target = new THnI("obj1", "obj1", dim, binsDims, mins, maxs);
    target->FillBin(5, 1);
    THnI* other = new THnI("obj2", "obj2", dim, binsDims, mins, maxs);
    other->FillBin(2, 1);
    other->FillBin(2, 1);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetBinContent(2), 2);
    BOOST_CHECK_EQUAL(target->GetBinContent(5), 1);

    delete other;
    delete target;
  }
  {
    const size_t dim = 5;
    const Int_t binsDims[dim] = {bins, bins, bins, bins, bins};
    const Double_t mins[dim] = {min, min, min, min, min};
    const Double_t maxs[dim] = {max, max, max, max, max};

    Double_t entry[dim] = {5, 5, 5, 5, 5};

    THnSparseI* target = new THnSparseI("obj1", "obj1", dim, binsDims, mins, maxs);
    target->Fill(entry);
    THnSparseI* other = new THnSparseI("obj2", "obj2", dim, binsDims, mins, maxs);
    entry[0] = entry[1] = entry[2] = entry[3] = entry[4] = 2;
    other->Fill(entry);
    other->Fill(entry);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetBinContent(0, nullptr), 1);
    BOOST_CHECK_EQUAL(target->GetBinContent(1), 2);

    delete other;
    delete target;
  }
  {
    struct format1 {
      Int_t a;
      Double_t b;
    } branch1;
    ULong64_t branch2;

    auto createTree = [&](std::string name) -> TTree* {
      TTree* tree = new TTree();
      tree->SetName(name.c_str());
      tree->Branch("b1", &branch1, "a/I:b/L:c/F:d/D");
      tree->Branch("b2", &branch2);
      return tree;
    };

    TTree* target = createTree("obj1");
    TTree* other = createTree("obj2");

    branch1.a = 1;
    branch1.b = 2;
    branch2 = 3;

    target->Fill();
    other->Fill();
    other->Fill();

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetEntries(), 3);

    delete other;
    delete target;
  }
  {
    constexpr Int_t points = 5;
    Double_t x1[] = {0, 1, 2, 3, 4};
    Double_t y1[] = {0, 1, 2, 3, 4};
    Double_t x2[] = {5, 6, 7, 8, 9};
    Double_t y2[] = {5, 6, 7, 8, 9};

    TGraph* target = new TGraph(points, x1, y1);
    TGraph* other = new TGraph(points, x2, y2);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetN(), 2 * points);

    delete other;
    delete target;
  }
  {
    auto target = new TProfile("hprof1", "hprof1", bins, min, max, min, max);
    target->Fill(2, 2, 1);
    auto other = new TProfile("hprof2", "hprof2", bins, min, max, min, max);
    other->Fill(5, 5, 1);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->GetEntries(), 2);
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(0)), 0);
    BOOST_CHECK_EQUAL(target->GetBinContent(target->FindBin(1)), 0);
    BOOST_CHECK_CLOSE(target->GetBinContent(target->FindBin(2)), 2, 0.001);
    BOOST_CHECK_CLOSE(target->GetBinContent(target->FindBin(5)), 5, 0.001);

    delete other;
    delete target;
  }
  {
    auto* target = new CustomMergeableTObject("obj1", 123);
    auto* other = new CustomMergeableTObject("obj2", 321);

    BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
    BOOST_CHECK_EQUAL(target->getSecret(), 444);

    delete other;
    delete target;
  }
  {
    auto* target = new CustomMergeableObject(123);
    auto* other = new CustomMergeableObject(321);

    BOOST_CHECK_NO_THROW(target->merge(other));
    BOOST_CHECK_EQUAL(target->getSecret(), 444);

    delete other;
    delete target;
  }
}

BOOST_AUTO_TEST_CASE(MergerCollection)
{
  // Setting up the target. Histo 1D + Custom stored in TObjArray.
  TObjArray* target = new TObjArray();
  target->SetOwner(true);

  TH1I* targetTH1I = new TH1I("histo 1d", "histo 1d", bins, min, max);
  targetTH1I->Fill(5);
  target->Add(targetTH1I);

  CustomMergeableTObject* targetCustom = new CustomMergeableTObject("custom", 9000);
  target->Add(targetCustom);

  // Setting up the other. Histo 1D + Histo 2D + Custom stored in TList.
  TList* other = new TList();
  other->SetOwner(true);

  TH1I* otherTH1I = new TH1I("histo 1d", "histo 1d", bins, min, max);
  otherTH1I->Fill(2);
  otherTH1I->Fill(2);
  other->Add(otherTH1I);

  TH2I* otherTH2I = new TH2I("histo 2d", "histo 2d", bins, min, max, bins, min, max);
  otherTH2I->Fill(5, 5);
  other->Add(otherTH2I);

  CustomMergeableTObject* otherCustom = new CustomMergeableTObject("custom", 1);
  other->Add(otherCustom);

  // Merge
  BOOST_CHECK_NO_THROW(algorithm::merge(target, other));

  // Make sure that deleting the object present only in `other` doesn't delete it in the `target`
  delete other;

  // Checks
  BOOST_REQUIRE_EQUAL(target->GetEntries(), 3);

  TH1I* resultTH1I = dynamic_cast<TH1I*>(target->FindObject("histo 1d"));
  BOOST_REQUIRE(resultTH1I != nullptr);
  BOOST_CHECK_EQUAL(resultTH1I->GetBinContent(resultTH1I->FindBin(2)), 2);
  BOOST_CHECK_EQUAL(resultTH1I->GetBinContent(resultTH1I->FindBin(5)), 1);

  TH2I* resultTH2I = dynamic_cast<TH2I*>(target->FindObject("histo 2d"));
  BOOST_REQUIRE(resultTH2I != nullptr);
  BOOST_CHECK_EQUAL(resultTH2I->GetBinContent(resultTH2I->FindBin(5, 5)), 1);

  auto* resultCustom = dynamic_cast<CustomMergeableTObject*>(target->FindObject("custom"));
  BOOST_REQUIRE(resultCustom != nullptr);
  BOOST_CHECK_EQUAL(resultCustom->getSecret(), 9001);

  delete target;
}

BOOST_AUTO_TEST_CASE(Deleting)
{
  TObjArray* main = new TObjArray();
  main->SetOwner(false);

  TH1I* histo1D = new TH1I("histo 1d", "histo 1d", bins, min, max);
  histo1D->Fill(5);
  main->Add(histo1D);

  CustomMergeableTObject* custom = new CustomMergeableTObject("custom", 9000);
  main->Add(custom);

  // Setting up the other. Histo 1D + Histo 2D + Custom stored in TList.
  auto* collectionInside = new TList();
  collectionInside->SetOwner(true);

  TH1I* histo1DinsideCollection = new TH1I("histo 1d", "histo 1d", bins, min, max);
  histo1DinsideCollection->Fill(2);
  collectionInside->Add(histo1DinsideCollection);

  TH2I* histo2DinsideCollection = new TH2I("histo 2d", "histo 2d", bins, min, max, bins, min, max);
  histo2DinsideCollection->Fill(5, 5);
  collectionInside->Add(histo2DinsideCollection);

  main->Add(collectionInside);

  // I am afraid we can't check more than that.
  BOOST_CHECK_NO_THROW(algorithm::deleteTCollections(main));
}

BOOST_AUTO_TEST_CASE(AverageHisto)
{
  TH1F* target = new TH1F("histo 1", "histo 1", bins, min, max);
  target->SetBit(TH1::kIsAverage);
  target->Fill(5);

  TH1F* other = new TH1F("histo 2", "histo 2", bins, min, max);
  other->SetBit(TH1::kIsAverage);
  other->Fill(5);

  BOOST_CHECK_NO_THROW(algorithm::merge(target, other));
  BOOST_CHECK_CLOSE(target->GetBinContent(other->FindBin(5)), 1.0, 0.001);

  delete target;
  delete other;
}

BOOST_AUTO_TEST_SUITE(VectorOfHistos)

gsl::span<float> to_span(std::shared_ptr<TH1F>& histo)
{
  return {histo->GetArray(), static_cast<uint>(histo->GetSize())};
}

template <typename T, std::size_t N>
gsl::span<T, N> to_array(T (&&arr)[N])
{
  return arr;
}

BOOST_AUTO_TEST_CASE(SameLength, *boost::unit_test::tolerance(0.001))
{
  auto target1_1 = std::make_shared<TH1F>("histo 1-1", "histo 1-1", bins, min, max);
  target1_1->Fill(5);
  target1_1->Fill(5);

  auto target1_2 = std::make_shared<TH1F>("histo 1-2", "histo 1-2", bins, min, max);
  target1_2->Fill(5);
  target1_2->Fill(5);
  target1_2->Fill(5);
  target1_2->Fill(5);

  VectorOfTObjectPtrs target{target1_1, target1_2};

  auto other1_1 = std::make_shared<TH1F>("histo 1-1", "histo 1-1", bins, min, max);
  other1_1->Fill(5);

  auto other1_2 = std::make_shared<TH1F>("histo 1-2", "histo 1-2", bins, min, max);
  other1_2->Fill(5);
  other1_2->Fill(5);

  VectorOfTObjectPtrs other{other1_1, other1_2};

  BOOST_TEST(target.size() == 2);
  BOOST_TEST(other.size() == 2);

  BOOST_TEST(to_span(target1_1) == to_array({0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(target1_2) == to_array({0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_1) == to_array({0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_2) == to_array({0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());

  BOOST_CHECK_NO_THROW(algorithm::merge(target, other));

  BOOST_TEST(target.size() == 2);
  BOOST_TEST(other.size() == 2);

  BOOST_TEST(to_span(target1_1) == to_array({0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(target1_2) == to_array({0., 0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_1) == to_array({0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_2) == to_array({0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(TargetLonger, *boost::unit_test::tolerance(0.001))
{
  auto target1_1 = std::make_shared<TH1F>("histo 1-1", "histo 1-1", bins, min, max);
  target1_1->Fill(5);
  target1_1->Fill(5);

  auto target1_2 = std::make_shared<TH1F>("histo 1-2", "histo 1-2", bins, min, max);
  target1_2->Fill(5);
  target1_2->Fill(5);
  target1_2->Fill(5);
  target1_2->Fill(5);

  VectorOfTObjectPtrs target{target1_1, target1_2};

  auto other1_1 = std::make_shared<TH1F>("histo 1-1", "histo 1-1", bins, min, max);
  other1_1->Fill(5);

  VectorOfTObjectPtrs other{other1_1};

  BOOST_TEST(target.size() == 2);
  BOOST_TEST(other.size() == 1);

  BOOST_TEST(to_span(target1_1) == to_array({0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(target1_2) == to_array({0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_1) == to_array({0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());

  BOOST_CHECK_NO_THROW(algorithm::merge(target, other));

  BOOST_TEST(target.size() == 2);
  BOOST_TEST(other.size() == 1);

  BOOST_TEST(to_span(target1_1) == to_array({0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(target1_2) == to_array({0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_1) == to_array({0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(OtherLonger, *boost::unit_test::tolerance(0.001))
{
  auto target1_1 = std::make_shared<TH1F>("histo 1-1", "histo 1-1", bins, min, max);
  target1_1->Fill(5);
  target1_1->Fill(5);

  VectorOfTObjectPtrs target{target1_1};

  auto other1_1 = std::make_shared<TH1F>("histo 1-1", "histo 1-1", bins, min, max);
  other1_1->Fill(5);

  auto other1_2 = std::make_shared<TH1F>("histo 1-2", "histo 1-2", bins, min, max);
  other1_2->Fill(5);
  other1_2->Fill(5);

  VectorOfTObjectPtrs other{other1_1, other1_2};

  BOOST_TEST(target.size() == 1);
  BOOST_TEST(other.size() == 2);

  BOOST_TEST(std::string_view{target[0]->GetName()} == "histo 1-1");

  BOOST_TEST(to_span(target1_1) == to_array({0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_1) == to_array({0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_2) == to_array({0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());

  BOOST_CHECK_NO_THROW(algorithm::merge(target, other));

  BOOST_TEST(target.size() == 2);
  BOOST_TEST(other.size() == 2);

  BOOST_TEST(std::string_view{target[0]->GetName()} == "histo 1-1");
  BOOST_TEST(std::string_view{target[1]->GetName()} == "histo 1-2");

  BOOST_TEST(to_span(target1_1) == to_array({0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_1) == to_array({0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
  BOOST_TEST(to_span(other1_2) == to_array({0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.}), boost::test_tools::per_element());
}

BOOST_AUTO_TEST_SUITE_END()
