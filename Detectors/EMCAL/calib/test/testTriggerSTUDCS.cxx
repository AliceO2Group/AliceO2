// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test EMCAL Calib
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "EMCALCalib/TriggerSTUDCS.h"

#include <algorithm>

namespace o2
{

namespace emcal
{

/// \brief Apply reference configuration
/// Reference configuration taken from pp 2016
/// \param testobject Object for test to be configured
///
void ConfigureReference(TriggerSTUDCS& testobject)
{
  testobject.setGammaHigh(0, 0);
  testobject.setGammaHigh(1, 0);
  testobject.setGammaHigh(2, 115);
  testobject.setGammaLow(0, 0);
  testobject.setGammaLow(1, 0);
  testobject.setGammaLow(2, 51);
  testobject.setJetHigh(0, 0);
  testobject.setJetHigh(1, 0);
  testobject.setJetHigh(2, 255);
  testobject.setJetLow(0, 0);
  testobject.setJetLow(1, 0);
  testobject.setJetLow(2, 204);
  testobject.setPatchSize(2);
  testobject.setFw(0x2A012);
  testobject.setMedianMode(0);
  testobject.setRegion(0xffffffff);
  for (int i = 0; i < 4; i++)
    testobject.setPHOSScale(i, 0);
}

BOOST_AUTO_TEST_CASE(TriggerSTUDCS_test)
{

  /// \brief testing all the getters and setters
  TriggerSTUDCS testobject;
  ConfigureReference(testobject);

  std::array<int, 3> Ghigh = {0, 0, 115};
  std::array<int, 3> Glow = {0, 0, 51};
  std::array<int, 3> Jhigh = {0, 0, 255};
  std::array<int, 3> Jlow = {0, 0, 204};
  int PatchSize = 2;
  int Fw = 0x2A012;
  int MedianMode = 0;
  int Region = 0xffffffff;
  int RawData = 1;

  BOOST_CHECK_EQUAL(testobject.getRegion(), Region);
  BOOST_CHECK_EQUAL(testobject.getFw(), Fw);
  BOOST_CHECK_EQUAL(testobject.getPatchSize(), PatchSize);
  BOOST_CHECK_EQUAL(testobject.getMedianMode(), MedianMode);
  BOOST_CHECK_EQUAL(testobject.getRawData(), RawData);

  for (int itrig = 0; itrig < 3; itrig++) {
    BOOST_CHECK_EQUAL(testobject.getGammaHigh(itrig), Ghigh[itrig]);
    BOOST_CHECK_EQUAL(testobject.getJetHigh(itrig), Jhigh[itrig]);
    BOOST_CHECK_EQUAL(testobject.getGammaLow(itrig), Glow[itrig]);
    BOOST_CHECK_EQUAL(testobject.getJetLow(itrig), Jlow[itrig]);
  }

  /// \brief Test for operator== on itself
  /// Tests whether operator== returns true in case the object is tested
  /// against itself.
  BOOST_CHECK_EQUAL(testobject == testobject, true);

  /// \brief Test for operator== on same object
  /// Tests whether operator== returns true in case both
  /// objects have the same content.
  TriggerSTUDCS test1, test2;
  ConfigureReference(test1);
  ConfigureReference(test2);
  BOOST_CHECK_EQUAL(test1 == test2, true);

  /// \brief Testing the copy constructor
  TriggerSTUDCS test3(test1);
  BOOST_CHECK_EQUAL(test3 == test1, true);

  /// \brief Testing the assignment operator
  TriggerSTUDCS test4 = test1;
  BOOST_CHECK_EQUAL(test4 == test1, true);

  /// \brief Test for operator== on different objects
  /// Tests whether the operator== returns false if at least one setting
  /// is different. For this operator== is tested with multiple objects
  /// based on a reference setting where only one parameter is changed at
  /// the time.
  TriggerSTUDCS ref;
  ConfigureReference(ref);
  ref.setRegion(0xffffff7f);
  BOOST_CHECK_EQUAL(ref == testobject, false);
  ref.setGammaHigh(2, 77);
  ref.setGammaLow(2, 51);
  ref.setJetHigh(2, 191);
  ref.setJetLow(2, 128);
  ref.setPatchSize(0);
  ref.setFw(0x1A012);
  ref.setMedianMode(1);
  ref.setPHOSScale(0, 1);
  ref.setPHOSScale(1, 2);
  ref.setPHOSScale(2, 1);
  ref.setPHOSScale(3, 0);
  BOOST_CHECK_EQUAL(ref == testobject, false);

  /// \brief Test for the stream operator
  /// Test if operator<< for a reference configuration produces
  /// the expected reference string. Test is implemented using a streaming
  /// operator.
  std::string reference = std::string("Gamma High: (0, 0, 115)\nGamma Low:  (0, 0, 51)\nJet High:   (0, 0, 255)\nJet Low:    (0, 0, 204)\n") + std::string("GetRawData: 1, Region: ffffffff (11111111111111111111111111111111), Median: 0, Firmware: 2a012, PHOS Scale: (0, 0, 0, 0)\n");

  TriggerSTUDCS test;
  ConfigureReference(test);
  std::stringstream testmaker;
  testmaker << test;
  BOOST_CHECK_EQUAL(testmaker.str() == reference, true);
}

} // namespace emcal

} // namespace o2
