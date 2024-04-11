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

#define BOOST_TEST_MODULE Test_EMCAL_Calib
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <array>
#include <random>
#include "EMCALCalib/CalibContainerErrors.h"
#include "EMCALCalib/Pedestal.h"

/// \brief Standard tests for Pedestal container

namespace o2
{
namespace emcal
{

using pedestalarray = std::vector<short>;

pedestalarray createRandomPedestals(bool isLEDMON)
{
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution gaussrand{40., 5.};
  pedestalarray pedestalcontainer(isLEDMON ? 480 : 17664);
  for (std::size_t ichan{0}; ichan < pedestalcontainer.size(); ++ichan) {
    pedestalcontainer[ichan] = std::round(gaussrand(gen));
  }

  return pedestalcontainer;
}

pedestalarray shiftPedestalValue(const pedestalarray& input, short shift = 1)
{
  pedestalarray shifted;
  for (std::size_t ichan{0}; ichan < input.size(); ++ichan) {
    shifted[ichan] = input[ichan] + shift;
  }
  return shifted;
}

BOOST_AUTO_TEST_CASE(testPedestal)
{
  auto pedestalsHG = createRandomPedestals(false),
       pedestalsLG = createRandomPedestals(false),
       pedestalsLEDMONHG = createRandomPedestals(true),
       pedestalsLEDMONLG = createRandomPedestals(true);

  o2::emcal::Pedestal pedestalObject;
  for (std::size_t ichan{0}; ichan < pedestalsHG.size(); ++ichan) {
    pedestalObject.addPedestalValue(ichan, pedestalsHG[ichan], false, false);
    pedestalObject.addPedestalValue(ichan, pedestalsLG[ichan], true, false);
  }
  for (std::size_t ichan{0}; ichan < pedestalsLEDMONHG.size(); ++ichan) {
    pedestalObject.addPedestalValue(ichan, pedestalsLEDMONHG[ichan], false, true);
    pedestalObject.addPedestalValue(ichan, pedestalsLEDMONLG[ichan], true, true);
  }

  // test adding entries beyond range
  for (std::size_t ichan{17665}; ichan < 18000; ++ichan) {
    BOOST_CHECK_EXCEPTION(pedestalObject.addPedestalValue(ichan, 2, false, true), o2::emcal::CalibContainerIndexException, [ichan](const o2::emcal::CalibContainerIndexException& e) { return e.getIndex() == ichan; });
    BOOST_CHECK_EXCEPTION(pedestalObject.addPedestalValue(ichan, 3, true, false), o2::emcal::CalibContainerIndexException, [ichan](const o2::emcal::CalibContainerIndexException& e) { return e.getIndex() == ichan; });
  }

  // test reading values in range
  for (std::size_t ichan{0}; ichan < pedestalsHG.size(); ++ichan) {
    BOOST_CHECK_EQUAL(pedestalObject.getPedestalValue(ichan, false, false), pedestalsHG[ichan]);
    BOOST_CHECK_EQUAL(pedestalObject.getPedestalValue(ichan, true, false), pedestalsLG[ichan]);
  }
  for (std::size_t ichan{0}; ichan < pedestalsLEDMONHG.size(); ++ichan) {
    BOOST_CHECK_EQUAL(pedestalObject.getPedestalValue(ichan, false, true), pedestalsLEDMONHG[ichan]);
    BOOST_CHECK_EQUAL(pedestalObject.getPedestalValue(ichan, true, true), pedestalsLEDMONLG[ichan]);
  }

  // test reading entries beyond range
  for (std::size_t ichan{17665}; ichan < 18000; ++ichan) {
    BOOST_CHECK_EXCEPTION(pedestalObject.getPedestalValue(ichan, false, false), o2::emcal::CalibContainerIndexException, [ichan](const o2::emcal::CalibContainerIndexException& e) { return e.getIndex() == ichan; });
    BOOST_CHECK_EXCEPTION(pedestalObject.getPedestalValue(ichan, true, false), o2::emcal::CalibContainerIndexException, [ichan](const o2::emcal::CalibContainerIndexException& e) { return e.getIndex() == ichan; });
    BOOST_CHECK_EXCEPTION(pedestalObject.getPedestalValue(ichan, false, true), o2::emcal::CalibContainerIndexException, [ichan](const o2::emcal::CalibContainerIndexException& e) { return e.getIndex() == ichan; });
    BOOST_CHECK_EXCEPTION(pedestalObject.getPedestalValue(ichan, true, true), o2::emcal::CalibContainerIndexException, [ichan](const o2::emcal::CalibContainerIndexException& e) { return e.getIndex() == ichan; });
  }

  // tests for operator==
  // shift pedestal by 1 for false test
  // Test all cases:
  // - same object
  // - same HG, different LG
  // - same LG, different HG
  // - both HG and LG different
  /*
  auto shiftedPedestalsHG = shiftPedestalValue(pedestalsHG, 1),
       shiftedPedestalsLG = shiftPedestalValue(pedestalsLG, 1);
  o2::emcal::Pedestal same, differLow, differHigh, differBoth;
  for (std::size_t ichan{0}; ichan < pedestalsHG.size(); ++ichan) {
    same.addPedestalValue(ichan, pedestalsHG[ichan], false);
    same.addPedestalValue(ichan, pedestalsLG[ichan], true);
    differLow.addPedestalValue(ichan, pedestalsHG[ichan], false);
    differLow.addPedestalValue(ichan, shiftedPedestalsLG[ichan], true);
    differHigh.addPedestalValue(ichan, shiftedPedestalsHG[ichan], false);
    differHigh.addPedestalValue(ichan, pedestalsLG[ichan], true);
    differBoth.addPedestalValue(ichan, shiftedPedestalsHG[ichan], false);
    differBoth.addPedestalValue(ichan, shiftedPedestalsLG[ichan], true);
  }
  BOOST_CHECK_EQUAL(pedestalObject, same);
  BOOST_CHECK_NE(pedestalObject, differLow);
  BOOST_CHECK_NE(pedestalObject, differHigh);
  BOOST_CHECK_NE(pedestalObject, differBoth);
  */
}

} // namespace emcal

} // namespace o2