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
#include "EMCALCalib/TriggerTRUDCS.h"

#include <algorithm>

namespace o2
{

namespace emcal
{

/// \brief Apply reference configuration
/// Reference configuration taken from pp 2016
/// \param testobject Object for test to be configured
///
void ConfigureReference(TriggerTRUDCS& testobject)
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

BOOST_AUTO_TEST_CASE(TriggerTRUDCS_test)
{

  /// \brief testing all the getters and setters
  TriggerTRUDCS testobject;
  ConfigureReference(testobject);

  uint64_t SELPF = 7711;
  uint64_t L0SEL = 1;
  uint64_t L0COSM = 100;
  uint64_t GTHRL0 = 132;
  std::array<uint32_t, 6> MaskReg = {1024, 0, 512, 31985, 0, 0};
  uint64_t RLBKSTU = 0;
  uint64_t Fw = 0x21;

  BOOST_CHECK_EQUAL(testobject.getSELPF(), SELPF);
  BOOST_CHECK_EQUAL(testobject.getL0SEL(), L0SEL);
  BOOST_CHECK_EQUAL(testobject.getL0COSM(), L0COSM);
  BOOST_CHECK_EQUAL(testobject.getGTHRL0(), GTHRL0);
  BOOST_CHECK_EQUAL(testobject.getRLBKSTU(), RLBKSTU);

  for (int ireg = 0; ireg < 6; ireg++)
    BOOST_CHECK_EQUAL(testobject.getMaskReg(ireg), MaskReg[ireg]);

  BOOST_CHECK_EQUAL(testobject.getFw(), Fw);

  /// \brief Test for operator== on itself
  /// Tests whether operator== returns true in case the object is tested
  /// against itself.
  BOOST_CHECK_EQUAL(testobject == testobject, true);

  /// \brief Test for operator== on same object
  /// Tests whether operator== returns true in case both
  /// objects have the same content.
  TriggerTRUDCS test1, test2;
  ConfigureReference(test1);
  ConfigureReference(test2);
  BOOST_CHECK_EQUAL(test1 == test2, true);

  /// \brief Testing the copy constructor
  TriggerTRUDCS test3(test1);
  BOOST_CHECK_EQUAL(test3 == test1, true);

  /// \brief Testing the assignment operator
  TriggerTRUDCS test4 = test1;
  BOOST_CHECK_EQUAL(test4 == test1, true);

  /// \brief Test for operator== on different objects
  /// Tests whether the operator== returns false if at least one setting
  /// is different. For this operator== is tested with multiple objects
  /// based on a reference setting where only one parameter is changed at
  /// the time.
  TriggerTRUDCS ref;
  ConfigureReference(ref);
  ref.setSELPF(7000);
  BOOST_CHECK_EQUAL(ref == testobject, false);
  ref.setL0SEL(2);
  ref.setL0COSM(1000);
  ref.setGTHRL0(184);
  ref.setRLBKSTU(1);
  ref.setFw(0x11);
  ref.setMaskReg(768, 0);
  ref.setMaskReg(15, 1);
  ref.setMaskReg(37632, 2);
  ref.setMaskReg(63, 3);
  ref.setMaskReg(0, 4);
  ref.setMaskReg(208, 5);
  BOOST_CHECK_EQUAL(ref == testobject, false);

  /// \brief Test for the stream operator
  /// Test if operator<< for a reference configuration produces
  /// the expected reference string. Test is implemented using a streaming
  /// operator.
  std::string reference = std::string("SELPF: 1e1f, L0SEL: 1, L0COSM: 100, GTHRL0: 132, RLBKSTU: 0, FW: 21\n") + std::string("Reg0: 00000000000000000000010000000000 (1024)\nReg1: 00000000000000000000000000000000 (0)\n") + std::string("Reg2: 00000000000000000000001000000000 (512)\nReg3: 00000000000000000111110011110001 (31985)\n") + std::string("Reg4: 00000000000000000000000000000000 (0)\nReg5: 00000000000000000000000000000000 (0)\n");

  TriggerTRUDCS test;
  ConfigureReference(test);
  std::stringstream testmaker;
  testmaker << test;
  BOOST_CHECK_EQUAL(testmaker.str() == reference, true);
}

} // namespace emcal

} // namespace o2
