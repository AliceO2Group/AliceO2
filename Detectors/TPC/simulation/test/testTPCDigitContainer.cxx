// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCDigitContainer.cxx
/// \brief This task tests the DigitContainer of the TPC digitization
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC DigitContainer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>
#include "TPCBase/Digit.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/DigitMCMetaData.h"
#include "TPCSimulation/SAMPAProcessing.h"

namespace o2
{
namespace TPC
{

/// \brief Test of the DigitContainer
/// A couple of values are filled into a DigitContainer and we check whether we get the same results after full
/// conversion to digits
BOOST_AUTO_TEST_CASE(DigitContainer_test1)
{
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();
  DigitContainer digitContainer;
  o2::dataformats::MCTruthContainer<MCCompLabel> mMCTruthArray;

  const std::vector<int> MCevent = { 1, 250, 3, 62, 1000 };
  const std::vector<int> MCtrack = { 22, 3, 4, 23, 523 };
  const std::vector<int> CRU = { 1, 23, 36, 53, 214 };
  const std::vector<int> Time = { 231, 2, 500, 230, 1 };
  const std::vector<int> Row = { 12, 5, 6, 2, 6 };
  const std::vector<int> Pad = { 1, 15, 14, 23, 5 };
  const std::vector<int> nEle = { 60, 100, 250, 1023, 2 };

  for (int i = 0; i < CRU.size(); ++i) {
    digitContainer.addDigit(MCevent[i], MCtrack[i], CRU[i], Time[i], Row[i], Pad[i], nEle[i]);
  }

  /// here the raw pointer is needed owed to the internal handling of the TClonesArrays in FairRoot
  /// Usually the mDigitsArray is what is registered to the FairRootManager
  auto* mDigitsArray = new std::vector<o2::TPC::Digit>;
  digitContainer.fillOutputContainer(mDigitsArray, mMCTruthArray, nullptr, 1000);

  BOOST_CHECK(CRU.size() == mDigitsArray->size());

  int digits = 0;
  for (auto& digit : *mDigitsArray) {
    gsl::span<const o2::MCCompLabel> mcArray = mMCTruthArray.getLabels(digits);
    for (int j = 0; j < static_cast<int>(mcArray.size()); ++j) {
      BOOST_CHECK(mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digits).index + j).getTrackID() ==
                  MCtrack[digits]);
      BOOST_CHECK(mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digits).index + j).getEventID() ==
                  MCevent[digits]);
    }
    BOOST_CHECK(digit.getCRU() == CRU[digits]);
    BOOST_CHECK(digit.getTimeStamp() == Time[digits]);
    BOOST_CHECK(digit.getRow() == Row[digits]);
    BOOST_CHECK(digit.getPad() == Pad[digits]);
    //      BOOST_CHECK(digit.getCharge() == static_cast<int>(sampa.getADCSaturation(nEle[digits])));
    ++digits;
  }

  delete mDigitsArray;
}

/// \brief Test of the DigitContainer
/// A couple of values are into the very same voxel (CRU, TimeBin, Row, Pad) and we check whether the charges are
/// properly summed up
/// and that the MC labels are right
BOOST_AUTO_TEST_CASE(DigitContainer_test2)
{
  const Mapper& mapper = Mapper::instance();
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();
  DigitContainer digitContainer;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mMCTruthArray;

  const std::vector<int> MCevent = { 1, 62, 1, 62, 62, 50, 62, 1, 1, 1 };
  const std::vector<int> MCtrack = { 22, 3, 22, 3, 3, 70, 3, 7, 7, 7 };
  const std::vector<int> CRU = { 23, 23, 23, 23, 23, 23, 23, 23, 23, 23 };
  const std::vector<int> Time = { 231, 231, 231, 231, 231, 231, 231, 231, 231, 231 };
  const std::vector<int> Row = { 12, 12, 12, 12, 12, 12, 12, 12, 12, 12 };
  const std::vector<int> Pad = { 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 };
  const std::vector<int> nEle = { 60, 1, 252, 10, 2, 3, 5, 25, 24, 23 };

  const std::vector<int> MCeventSorted = { 62, 1, 1, 50 };
  const std::vector<int> MCtrackSorted = { 3, 7, 22, 70 };

  int nEleSum = 0;
  for (int i = 0; i < CRU.size(); ++i) {
    digitContainer.addDigit(MCevent[i], MCtrack[i], CRU[i], Time[i], Row[i], Pad[i], nEle[i]);
    nEleSum += nEle[i];
  }

  /// here the raw pointer is needed owed to the internal handling of the TClonesArrays in FairRoot
  /// Usually the mDigitsArray is what is registered to the FairRootManager
  auto* mDigitsArray = new std::vector<o2::TPC::Digit>;
  auto* mDigitsDebugArray = new std::vector<o2::TPC::DigitMCMetaData>;
  digitContainer.fillOutputContainer(mDigitsArray, mMCTruthArray, mDigitsDebugArray, 1000);

  BOOST_CHECK(mDigitsArray->size() == 1);
  BOOST_CHECK(mDigitsArray->size() == mDigitsDebugArray->size());

  int digits = 0;
  for (auto& digit : *mDigitsArray) {
    auto& digitMetaData = mDigitsDebugArray->at(digits);
    gsl::span<const o2::MCCompLabel> mcArray = mMCTruthArray.getLabels(digits);
    for (int j = 0; j < static_cast<int>(mcArray.size()); ++j) {
      BOOST_CHECK(mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digits).index + j).getTrackID() ==
                  MCtrackSorted[j]);
      BOOST_CHECK(mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digits).index + j).getEventID() ==
                  MCeventSorted[j]);
    }
    BOOST_CHECK(digit.getCRU() == CRU[digits]);
    BOOST_CHECK(digit.getTimeStamp() == Time[digits]);
    BOOST_CHECK(digit.getRow() == Row[digits]);
    BOOST_CHECK(digit.getPad() == Pad[digits]);
    BOOST_CHECK(digit.getCharge() == static_cast<int>(sampa.getADCSaturation(nEleSum - digitMetaData.getCommonMode())));
    //      BOOST_CHECK_CLOSE(digitMetaData.getCommonMode(), nEleSum/static_cast<float>(mapper.getPadsInIROC()), 1E-4);
    ++digits;
  }

  delete mDigitsArray;
}
}
}
