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
#include "EMCALCalib/TriggerDCS.h"
#include "EMCALCalib/TriggerTRUDCS.h"
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
void ConfigureReferenceTRU(TriggerTRUDCS& testobject)
{
  testobject.setSELPF(7711);
  testobject.setL0SEL(1);
  testobject.setL0COSM(100);
  testobject.setGTHRL0(132);
  testobject.setMaskReg(1024, 0);
  testobject.setMaskReg(0, 1);
  testobject.setMaskReg(512, 2);
  testobject.setMaskReg(31985, 3);
  testobject.setMaskReg(0, 4);
  testobject.setMaskReg(0, 5);
  testobject.setRLBKSTU(0);
  testobject.setFw(0x21);
}

void ConfigureReferenceSTU(TriggerSTUDCS& testobject)
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

BOOST_AUTO_TEST_CASE(TriggerDCS_test)
{

  /// \brief testing all the getters and setters
  TriggerSTUDCS testSTUDCal;
  ConfigureReferenceSTU(testSTUDCal);

  TriggerSTUDCS testSTUEMCal;
  ConfigureReferenceSTU(testSTUEMCal);
  testSTUEMCal.setRegion(0xffffff7f);

  TriggerTRUDCS testTRU;
  ConfigureReferenceTRU(testTRU);

  TriggerDCS testobject;
  testobject.setSTUEMCal(testSTUEMCal);
  testobject.setSTUDCal(testSTUDCal);
  testobject.setTRU(testTRU);

  BOOST_CHECK_EQUAL(testobject.getSTUDCSEMCal(), testSTUEMCal);
  BOOST_CHECK_EQUAL(testobject.getSTUDCSDCal(), testSTUDCal);
  BOOST_CHECK_EQUAL(testobject.getTRUDCS(0), testTRU);

  /// \brief Test for operator== on itself
  /// Tests whether operator== returns true in case the object is tested
  /// against itself.
  BOOST_CHECK_EQUAL(testobject == testobject, true);

  /// \brief Test for operator== on different objects
  /// Tests whether the operator== returns false if at least one setting
  /// is different. For this operator== is tested with multiple objects
  /// based on a reference setting where only one parameter is changed at
  /// the time.
  TriggerDCS ref;

  TriggerTRUDCS testTRU1;
  ConfigureReferenceTRU(testTRU1);
  testTRU1.setSELPF(7000);
  ref.setSTUEMCal(testSTUEMCal);
  ref.setSTUDCal(testSTUDCal);
  ref.setTRU(testTRU1);
  BOOST_CHECK_EQUAL(ref == testobject, false);
}

} // namespace emcal

} // namespace o2
