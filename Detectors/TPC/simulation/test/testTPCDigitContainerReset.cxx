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

/// \file testTPCDigitContainerReset.cxx
/// \brief Test that DigitContainer::reset() leaves no MC labels behind for the next sector

#define BOOST_TEST_MODULE Test TPC DigitContainer reset
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <vector>
#include <fmt/format.h>
#include "DataFormatsTPC/Digit.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBaseRecSim/CDBInterface.h"

namespace o2::tpc
{

// A sector whose last time bins lie beyond the timeframe limit must not pass their labels to the next sector
BOOST_AUTO_TEST_CASE(DigitContainer_noLabelCarryOver)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  // limit the flush to one orbit (time bin 447) and propagate the ADC values unchanged
  o2::conf::ConfigurableParam::updateFromString(fmt::format("DigiParams.maxOrbitsToDigitize=1;TPCEleParam.DigiMode={}", (int)DigitzationMode::PropagateADC));
  const Mapper& mapper = Mapper::instance();
  SAMPAProcessing::instance();

  const GlobalPadNumber globalPad = mapper.getPadNumberInROC(PadROCPos(CRU(0).roc(), PadPos(10, 10)));
  const MCCompLabel labelFirst(1, 1, 0, false);
  const MCCompLabel labelSecond(2, 2, 0, false);

  DigitContainer digitContainer;
  digitContainer.reset();

  // first sector: a signal beyond the timeframe limit, which the final flush does not reach
  digitContainer.addDigit(labelFirst, CRU(0), 600, globalPad, 100);
  std::vector<Digit> digits;
  dataformats::MCTruthContainer<MCCompLabel> labels;
  std::vector<CommonMode> commonMode;
  digitContainer.fillOutputContainer(digits, labels, commonMode, Sector(0), 0, true, true);
  BOOST_CHECK(digits.empty());

  // second sector, as after Digitizer::setSector(): same pad, early time bin
  digitContainer.reset();
  digits.clear();
  labels.clear();
  commonMode.clear();
  digitContainer.addDigit(labelSecond, CRU(10), 153, globalPad, 100);
  digitContainer.fillOutputContainer(digits, labels, commonMode, Sector(1), 0, true, true);

  BOOST_REQUIRE(digits.size() == 1);
  const auto digitLabels = labels.getLabels(0);
  BOOST_CHECK_EQUAL(digitLabels.size(), 1);
  for (const auto& label : digitLabels) {
    BOOST_CHECK(label == labelSecond);
  }
}

} // namespace o2::tpc
