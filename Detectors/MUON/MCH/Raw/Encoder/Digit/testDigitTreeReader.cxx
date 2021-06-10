// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE MCH DigitTreeReader
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DigitTreeReader.h"
#include <TFile.h>
#include <TTree.h>
#include "CommonDataFormat/InteractionRecord.h"
#include <vector>
#include <memory>

using namespace o2::mch;
using namespace o2::mch::raw;

std::unique_ptr<TTree> createTestTree()
{
  auto tree = std::make_unique<TTree>();

  tree->SetName("o2sim");

  std::vector<Digit>* digits = new std::vector<Digit>();
  std::vector<ROFRecord>* rofs = new std::vector<ROFRecord>();

  tree->Branch("MCHDigit", digits);
  tree->Branch("MCHROFRecords", rofs);

  constexpr int ndigits{10};

  for (auto i = 0; i < ndigits; i++) {
    digits->emplace_back(100, i, i * 10, 0);
  }

  rofs->emplace_back(o2::InteractionRecord{1234, 42}, 0, 2);
  rofs->emplace_back(o2::InteractionRecord{1234, 43}, 4, 3);
  rofs->emplace_back(o2::InteractionRecord{1234, 44}, 7, 4);

  tree->Fill();

  digits->clear();
  rofs->clear();
  tree->Fill();
  tree->Fill();
  return tree;
}

BOOST_AUTO_TEST_CASE(NofEntriesIsNumberOfTFIs3)
{
  auto tree = createTestTree();
  BOOST_CHECK_EQUAL(tree->GetEntries(), 3);
  TFile f("toto.root", "RECREATE");
  tree->Write("o2sim");
}

BOOST_AUTO_TEST_CASE(NofDigitsOfFirstRofIs2)
{
  std::vector<Digit> digits;
  o2::mch::ROFRecord rof;
  auto tree = createTestTree();
  DigitTreeReader dtr(tree.get());
  dtr.nextDigits(rof, digits);
  BOOST_CHECK_EQUAL(digits.size(), 2);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustThrowIfDigitAndRofBranchesDoNotExist)
{
  TTree emptyTree("emptyTree", "no branch at all");
  BOOST_CHECK_THROW(DigitTreeReader dtr(&emptyTree), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustThrowIfInputTreeIsNullptr)
{
  BOOST_CHECK_THROW(DigitTreeReader dtr(nullptr), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustThrowIfRofBranchIsMissing)
{
  TTree missingRofs("missingRofs", "MCHDigit branch alone");
  std::vector<o2::mch::Digit> digits;
  missingRofs.Branch("MCHDigit", &digits);
  BOOST_CHECK_THROW(DigitTreeReader dtr(&missingRofs), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustThrowIfDigitBranchIsMissing)
{
  TTree missingDigits("missingDigits", "MCHROFRecords branch alone");
  std::vector<o2::mch::ROFRecord> rofs;
  missingDigits.Branch("MCHROFRecords", &rofs);
  BOOST_CHECK_THROW(DigitTreeReader dtr(&missingDigits), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustThrowIfDigitBranchIsOfTheWrongType)
{
  TTree invalidDigits("invalidDigits", "MCHDigit branch present but of wrong type");
  std::vector<o2::mch::ROFRecord> rofs;
  invalidDigits.Branch("MCHDigit", &rofs); // setting wrong type on purpose
  invalidDigits.Branch("MCHROFRecords", &rofs);
  invalidDigits.Fill();
  BOOST_CHECK_THROW(DigitTreeReader dtr(&invalidDigits), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustThrowIfRofBranchIsOfTheWrongType)
{
  TTree invalidRofs("invalidRofs", "MCHROFRecords branch present but of wrong type");
  std::vector<o2::mch::Digit> digits;
  invalidRofs.Branch("MCHDigit", &digits);
  invalidRofs.Branch("MCHROFRecords", &digits); // setting wrong type on purpose
  invalidRofs.Fill();
  BOOST_CHECK_THROW(DigitTreeReader dtr(&invalidRofs), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustThrowIfNoEntry)
{
  TTree noEntry("noEntry", "All branches correct but no entry");
  std::vector<o2::mch::Digit> digits;
  std::vector<o2::mch::ROFRecord> rofs;
  noEntry.Branch("MCHDigit", &digits);
  noEntry.Branch("MCHROFRecords", &rofs);
  BOOST_CHECK_THROW(DigitTreeReader dtr(&noEntry), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DigitTreeReaderMustNotThrowIfInputTreeHasAllBranchesAndAtLeastOneEntry)
{
  TTree correct("noEntry", "All branches correct but no entry");
  std::vector<o2::mch::Digit> digits;
  std::vector<o2::mch::ROFRecord> rofs;
  correct.Branch("MCHDigit", &digits);
  correct.Branch("MCHROFRecords", &rofs);
  correct.Fill();
  BOOST_CHECK_NO_THROW(DigitTreeReader dtr(&correct));
}
