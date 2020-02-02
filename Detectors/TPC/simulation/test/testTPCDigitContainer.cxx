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
#include "TPCBase/CDBInterface.h"

namespace o2
{
namespace tpc
{

/// \brief Test of the DigitContainer
/// A couple of values are filled into a DigitContainer and we check whether we get the same results after full
/// conversion to digits
BOOST_AUTO_TEST_CASE(DigitContainer_test1)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  o2::conf::ConfigurableParam::updateFromString("TPCEleParam.DigiMode=3"); // propagate the ADC values, otherwise the computation get complicated
  const Mapper& mapper = Mapper::instance();
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();
  DigitContainer digitContainer;
  dataformats::MCTruthContainer<MCCompLabel> mMCTruthArray;
  digitContainer.reset();

  const std::vector<int> MCevent = {1, 250, 3, 62, 1000};
  const std::vector<int> MCtrack = {22, 3, 4, 23, 523};
  const std::vector<int> cru = {0, 0, 0, 0, 0};
  const std::vector<int> Time = {231, 2, 500, 230, 1};
  const std::vector<int> Row = {12, 5, 6, 2, 6};
  const std::vector<int> Pad = {1, 15, 14, 23, 5};
  const std::vector<int> nEle = {60, 100, 250, 1023, 2};

  const std::vector<int> timeMapping = {4, 1, 3, 0, 2};

  for (int i = 0; i < cru.size(); ++i) {
    const GlobalPadNumber globalPad = mapper.getPadNumberInROC(PadROCPos(CRU(cru[i]).roc(), PadPos(Row[i], Pad[i])));
    digitContainer.addDigit(MCCompLabel(MCtrack[i], MCevent[i], 0, false), cru[i], Time[i], globalPad, nEle[i]);
  }

  std::vector<Digit> mDigitsArray;
  std::vector<o2::tpc::CommonMode> commonMode;
  digitContainer.fillOutputContainer(mDigitsArray, mMCTruthArray, commonMode, 0, 0, true, true);

  for (size_t i = 0; i < commonMode.size(); ++i) {
    auto digit = mDigitsArray[i];
    const CRU cru = digit.getCRU();
    const auto gemStack = cru.gemStack();
    const float nPads = mapper.getNumberOfPads(GEMstack(gemStack));
    BOOST_CHECK_CLOSE(commonMode[i].getCommonMode(), digit.getChargeFloat() / nPads, 1E-6);
  }

  BOOST_CHECK(cru.size() == mDigitsArray.size());

  int digits = 0;
  for (auto& digit : mDigitsArray) {
    const int trueDigit = timeMapping[digits];
    gsl::span<const o2::MCCompLabel> mcArray = mMCTruthArray.getLabels(digits);
    for (int j = 0; j < static_cast<int>(mcArray.size()); ++j) {
      BOOST_CHECK(mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digits).index + j).getTrackID() ==
                  MCtrack[trueDigit]);
      BOOST_CHECK(mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(digits).index + j).getEventID() ==
                  MCevent[trueDigit]);
    }
    BOOST_CHECK(digit.getCRU() == cru[trueDigit]);
    BOOST_CHECK(digit.getTimeStamp() == Time[trueDigit]);
    BOOST_CHECK(digit.getRow() == Row[trueDigit]);
    BOOST_CHECK(digit.getPad() == Pad[trueDigit]);
    ++digits;
  }
}

/// \brief Test of the DigitContainer
/// A couple of values are into the very same voxel (CRU, TimeBin, Row, Pad) for each CRU in one sector and we check that the MC labels are right
BOOST_AUTO_TEST_CASE(DigitContainer_test2)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  o2::conf::ConfigurableParam::updateFromString("TPCEleParam.DigiMode=3"); // propagate the ADC values, otherwise the computation get complicated
  const Mapper& mapper = Mapper::instance();
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();
  DigitContainer digitContainer;
  digitContainer.reset();
  dataformats::MCTruthContainer<MCCompLabel> mMCTruthArray;

  // MC labels to add to each voxel
  const std::vector<int> MCevent = {1, 62, 1, 62, 62, 50, 62, 1, 1, 1};
  const std::vector<int> MCtrack = {22, 3, 22, 3, 3, 70, 3, 7, 7, 7};

  // voxel definitions
  const std::vector<int> cru = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<int> Time = {231, 231, 231, 231, 231, 231, 231, 231, 231, 231};
  const std::vector<int> Row = {11, 11, 11, 11, 11, 11, 11, 11, 11, 11}; // the last region only has 12 rows, so the row number should not exceed 11
  const std::vector<int> Pad = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
  const std::vector<int> nEle = {60, 1, 252, 10, 2, 3, 5, 25, 24, 23};

  // the resulting MC labels should be sorted by the number of occurrance, the one with
  // the highest occurrance first
  const std::vector<int> MCeventSorted = {62, 1, 1, 50};
  const std::vector<int> MCtrackSorted = {3, 7, 22, 70};

  for (int i = 0; i < cru.size(); ++i) {
    const CRU c(cru[i]);
    const DigitPos digiPadPos(c, PadPos(Row[i], Pad[i]));
    const GlobalPadNumber globalPad = mapper.globalPadNumber(digiPadPos.getGlobalPadPos());

    // add labels to each voxel, sum up the charge MCevent.size() times
    for (int j = 0; j < MCevent.size(); ++j) {
      digitContainer.addDigit(MCCompLabel(MCtrack[j], MCevent[j], 0, false), cru[i], Time[i], globalPad, nEle[i]);
    }
  }

  std::vector<Digit> mDigitsArray;
  std::vector<o2::tpc::CommonMode> commonMode;
  digitContainer.fillOutputContainer(mDigitsArray, mMCTruthArray, commonMode, 0, 0, true, true);

  BOOST_CHECK(mDigitsArray.size() == cru.size());

  std::array<float, GEMSTACKSPERSECTOR> chargeSum;
  chargeSum.fill(0);
  int digits = 0;
  for (const auto& digit : mDigitsArray) {
    // check MC labels and proper sorting
    const auto& mcArray = mMCTruthArray.getLabels(digits);
    BOOST_CHECK(mcArray.size() == MCtrackSorted.size());
    for (int j = 0; j < static_cast<int>(mcArray.size()); ++j) {
      BOOST_CHECK(mcArray[j].getTrackID() == MCtrackSorted[j]);
      BOOST_CHECK(mcArray[j].getEventID() == MCeventSorted[j]);
    }

    // check digit position
    const int row = Row[digits] + mapper.getGlobalRowOffsetRegion(CRU(digit.getCRU()).region());
    BOOST_CHECK(digit.getCRU() == cru[digits]);
    BOOST_CHECK(digit.getTimeStamp() == Time[digits]);
    BOOST_CHECK(digit.getRow() == row);
    BOOST_CHECK(digit.getPad() == Pad[digits]);

    const CRU cru = digit.getCRU();
    const auto gemStack = cru.gemStack();
    chargeSum[gemStack] += digit.getChargeFloat();
    ++digits;
  }

  for (size_t i = 0; i < commonMode.size(); ++i) {
    const float nPads = mapper.getNumberOfPads(GEMstack(i));
    BOOST_CHECK_CLOSE(commonMode[i].getCommonMode(), chargeSum[i] / nPads, 1E-6);
  }
}
} // namespace tpc
} // namespace o2
