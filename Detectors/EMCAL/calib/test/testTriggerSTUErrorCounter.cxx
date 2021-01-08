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
#include "EMCALCalib/TriggerSTUErrorCounter.h"

#include <algorithm>

namespace o2
{

namespace emcal
{

BOOST_AUTO_TEST_CASE(TriggerSTUErrorCounter_test)
{

  /// \brief testing all the getters and setters
  TriggerSTUErrorCounter testobject1, testobject2;

  std::pair<int, unsigned long> TimeAndError(123, 10);
  int Time = 123;
  unsigned long Error = 10;

  testobject1.setValue(TimeAndError);
  testobject2.setValue(Time, Error);

  BOOST_CHECK_EQUAL(testobject1.getTime(), TimeAndError.first);
  BOOST_CHECK_EQUAL(testobject1.getErrorCount(), TimeAndError.second);
  BOOST_CHECK_EQUAL(testobject2.getTime(), Time);
  BOOST_CHECK_EQUAL(testobject2.getErrorCount(), Error);

  /// \brief Test for operator== on itself
  /// Tests whether operator== returns true in case the object is tested
  /// against itself.
  BOOST_CHECK_EQUAL(testobject1 == testobject1, true);

  /// \brief Test for operator== on same object
  /// Tests whether operator== returns true in case both
  /// objects have the same content.
  BOOST_CHECK_EQUAL(testobject1 == testobject2, true);

  /// \brief Testing the copy constructor
  TriggerSTUErrorCounter testobject3(testobject1);
  BOOST_CHECK_EQUAL(testobject3 == testobject1, true);

  /// \brief Testing the assignment operator
  TriggerSTUErrorCounter testobject4 = testobject1;
  BOOST_CHECK_EQUAL(testobject4 == testobject1, true);

  /// \brief Test for operator== on different objects
  /// Tests whether the operator== returns false if at least one setting
  /// is different. For this operator== is tested with multiple objects
  /// based on a reference setting where only one parameter is changed at
  /// the time.
  TriggerSTUErrorCounter ref;
  ref.setValue(std::make_pair(2, 77));
  BOOST_CHECK_EQUAL(ref == testobject1, false);
  ref.setValue(2, 51);
  BOOST_CHECK_EQUAL(ref == testobject1, false);
}

} // namespace emcal

} // namespace o2
