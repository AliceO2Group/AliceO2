// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw DigitsMerger
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "DigitsMerger.h"

using namespace o2::mch;
using namespace o2::mch::raw;


BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(digitsmerger)

static const int sampaWindowSize = 100;


/// \brief Helper function for creating two digits that can be merged together
static void makeMergeableDigits(MergerDigit& d1, MergerDigit& d2)
{
  int solarId = 0;
  int dsAddr = 0;
  int chAddr = 0;
  int deId = 100;
  int padId = 0;
  int adc = 100;
  int orbit = 0;
  int sampaTime = 50;
  int bunchCrossing = 0;
  int nSamples = sampaWindowSize - sampaTime;
  Digit::Time time{sampaTime, bunchCrossing, orbit};

  d1 = MergerDigit{o2::mch::Digit(deId, padId, adc, time, nSamples), false/*, solarId, dsAddr, chAddr*/};

  int adc2 = 10;
  int sampaTime2 = 0;
  int bunchCrossing2 = sampaWindowSize * 4;
  int nSamples2 = 10;
  Digit::Time time2{sampaTime2, bunchCrossing2, orbit};

  d2 = MergerDigit{o2::mch::Digit(deId, padId, adc2, time2, nSamples2), false/*, solarId, dsAddr, chAddr*/};
}


/// \brief Helper function for adding one digit to the merger object
static void addDigit(Merger& merger, int feeId, MergerDigit& d)
{
  int solarId = 0;
  int dsAddr = 0;
  int chAddr = 0;
  merger.addDigit(feeId, solarId, dsAddr, chAddr, d.digit.getDetID(), d.digit.getPadID(), d.digit.getADC(), d.digit.getTime(), d.digit.nofSamples());
}


/// \brief Helper function to increment the orbit number in an existing digit
void incrementOrbit(o2::mch::Digit& d)
{
  o2::mch::Digit d2 = d;
  o2::mch::Digit::Time t = d.getTime();
  t.orbit += 1;
  d = o2::mch::Digit(d2.getDetID(), d2.getPadID(), d2.getADC(), t, d2.nofSamples());
}


/// \brief Helper function to increment the detector ID number in an existing digit
void incrementDetID(o2::mch::Digit& d)
{
  o2::mch::Digit d2 = d;
  d = o2::mch::Digit(d2.getDetID()+1, d2.getPadID(), d2.getADC(), d2.getTime(), d2.nofSamples());
}


/// \brief Helper function to increment the pad ID in an existing digit
void incrementPadID(o2::mch::Digit& d)
{
  o2::mch::Digit d2 = d;
  d = o2::mch::Digit(d2.getDetID(), d2.getPadID()+1, d2.getADC(), d2.getTime(), d2.nofSamples());
}


/// \brief Helper function to modify the sampa time in an existing digit
void incrementSampaTime(o2::mch::Digit& d, int delta)
{
  o2::mch::Digit d2 = d;
  o2::mch::Digit::Time t = d.getTime();
  t.sampaTime += delta;
  d = o2::mch::Digit(d2.getDetID(), d2.getPadID(), d2.getADC(), t, d2.nofSamples());
}


/// \brief Helper function to modify the bunch crossing value in an existing digit
void incrementBunchCrossing(o2::mch::Digit& d, int delta)
{
  o2::mch::Digit d2 = d;
  o2::mch::Digit::Time t = d.getTime();
  t.bunchCrossing += delta;
  d = o2::mch::Digit(d2.getDetID(), d2.getPadID(), d2.getADC(), t, d2.nofSamples());
}


/// \brief Helper function for running the digits merger on two input digits, and store the result in a vector object.
/// \brief The two digits are inserted in the same orbit.
void runMergerSameOrbit(MergerDigit& d1, MergerDigit& d2, std::vector<Digit>& digitsOut)
{
  const auto storeDigit = [&](const Digit& d) {
    digitsOut.emplace_back(d);
  };

  int feeId = 0;
  int orbit = d1.digit.getTime().orbit;

  // initialize the merger object
  Merger merger;
  merger.setDigitHandler(storeDigit);

  // start a new orbit, add the two digits, and stop the orbit
  merger.setOrbit(feeId, orbit, false);
  addDigit(merger, feeId, d1);
  addDigit(merger, feeId, d2);
  merger.setOrbit(feeId, orbit, true);

  // start/stop a new orbit to trigger the sending of the merged digit
  merger.setOrbit(feeId, orbit+1, false);
  merger.setOrbit(feeId, orbit+1, true);
}


/// \brief Helper function for running the digits merger on two input digits, and store the result in a vector object.
/// \brief The two digits are inserted in two consecutive orbits.
void runMergerConsecutiveOrbits(MergerDigit& d1, MergerDigit& d2, std::vector<Digit>& digitsOut)
{
  const auto storeDigit = [&](const Digit& d) {
    digitsOut.emplace_back(d);
  };

  int feeId = 0;
  int orbit = d1.digit.getTime().orbit;

  // initialize the merger object
  Merger merger;
  merger.setDigitHandler(storeDigit);

  // start a new orbit, add the two digits, and stop the orbit
  merger.setOrbit(feeId, orbit, false);
  addDigit(merger, feeId, d1);
  merger.setOrbit(feeId, orbit, true);

  // start/stop a new orbit to trigger the sending of the merged digit
  merger.setOrbit(feeId, orbit+1, false);
  addDigit(merger, feeId, d2);
  merger.setOrbit(feeId, orbit+1, true);
}


/// \brief helper macro for checking the equivalence of two digits
#define CHECK_DIGIT(d1, d2) \
    BOOST_CHECK_EQUAL(d1.getDetID(), d2.getDetID()); \
    BOOST_CHECK_EQUAL(d1.getPadID(), d2.getPadID()); \
    BOOST_CHECK_EQUAL(d1.getTime().sampaTime, d2.getTime().sampaTime); \
    BOOST_CHECK_EQUAL(d1.getTime().bunchCrossing, d2.getTime().bunchCrossing); \
    BOOST_CHECK_EQUAL(d1.getTime().orbit, d2.getTime().orbit); \
    BOOST_CHECK_EQUAL(d1.getADC(), d2.getADC()); \
    BOOST_CHECK_EQUAL(d1.nofSamples(), d2.nofSamples());


/// \brief helper macro for checking the contents of a merged digit
#define CHECK_MERGED_DIGIT(d, d1, d2) \
    BOOST_CHECK_EQUAL(d.getDetID(), d1.getDetID()); \
    BOOST_CHECK_EQUAL(d.getPadID(), d1.getPadID()); \
    \
    // check that the merged digit has the same time as the fist input digit \
    BOOST_CHECK_EQUAL(d.getTime().sampaTime, d1.getTime().sampaTime); \
    BOOST_CHECK_EQUAL(d.getTime().bunchCrossing, d1.getTime().bunchCrossing); \
    BOOST_CHECK_EQUAL(d.getTime().orbit, d1.getTime().orbit); \
    \
    // check that the total charge of the merged digit is equal to the sum of the individual charges \
    BOOST_CHECK_EQUAL(d.getADC(), d1.getADC() + d2.getADC()); \
    \
    // check that the number of samples of the merged digit is equal to the sum of the individual digits \
    BOOST_CHECK_EQUAL(d.nofSamples(), d1.nofSamples() + d2.nofSamples());


/// \brief Test of digits merging in the same orbit
BOOST_AUTO_TEST_CASE(MergeDigits)
{
  std::vector<Digit> digitsOut;
  MergerDigit d1, d2;
  makeMergeableDigits(d1, d2);

  runMergerSameOrbit(d1, d2, digitsOut);

  // check that the input digits have been merged into a single one with the expected parameters
  BOOST_CHECK_EQUAL(digitsOut.size(), 1);

  Digit& d = digitsOut[0];
  CHECK_MERGED_DIGIT(d, d1.digit, d2.digit);
}


/// \brief Test of digits merging in two consecutive orbits
BOOST_AUTO_TEST_CASE(MergeDigitsConsecutiveOrbits)
{
  std::vector<Digit> digitsOut;
  MergerDigit d1, d2;
  makeMergeableDigits(d1, d2);

  incrementOrbit(d2.digit);

  runMergerConsecutiveOrbits(d1, d2, digitsOut);

  // check that the input digits have been merged into a single one with the expected parameters
  BOOST_CHECK_EQUAL(digitsOut.size(), 1);

  Digit& d = digitsOut[0];
  CHECK_MERGED_DIGIT(d, d1.digit, d2.digit);
}


/// \brief Check that two digits with different detector IDs are not merged
BOOST_AUTO_TEST_CASE(MergeDigitsWrongDetID)
{
  std::vector<Digit> digitsOut;
  MergerDigit d1, d2;
  makeMergeableDigits(d1, d2);

  incrementDetID(d2.digit);

  runMergerSameOrbit(d1, d2, digitsOut);

  // check that the input digits have not been merged
  BOOST_CHECK_EQUAL(digitsOut.size(), 2);

  // check that the two output digits are identical to the input ones
  Digit& do1 = digitsOut[0];
  Digit& do2 = digitsOut[1];
  CHECK_DIGIT(do1, d1.digit);
  CHECK_DIGIT(do2, d2.digit);
}


/// \brief Check that two digits with different pad IDs are not merged
BOOST_AUTO_TEST_CASE(MergeDigitsWrongPadID)
{
  std::vector<Digit> digitsOut;
  MergerDigit d1, d2;
  makeMergeableDigits(d1, d2);

  incrementPadID(d2.digit);

  runMergerSameOrbit(d1, d2, digitsOut);

  // check that the input digits have not been merged
  BOOST_CHECK_EQUAL(digitsOut.size(), 2);

  // check that the two output digits are identical to the input ones
  Digit& do1 = digitsOut[0];
  Digit& do2 = digitsOut[1];
  CHECK_DIGIT(do1, d1.digit);
  CHECK_DIGIT(do2, d2.digit);
}


/// \brief Check that two digits with incompatible SAMPA times are not merged
BOOST_AUTO_TEST_CASE(MergeDigitsWrongSampaTime)
{
  std::vector<Digit> digitsOut;
  MergerDigit d1, d2;
  makeMergeableDigits(d1, d2);

  incrementSampaTime(d1.digit, -1);

  runMergerSameOrbit(d1, d2, digitsOut);

  // check that the input digits have not been merged
  BOOST_CHECK_EQUAL(digitsOut.size(), 2);

  // check that the two output digits are identical to the input ones
  Digit& do1 = digitsOut[0];
  Digit& do2 = digitsOut[1];
  CHECK_DIGIT(do1, d1.digit);
  CHECK_DIGIT(do2, d2.digit);
}


/// \brief Check that two digits with incompatible bunch crossings are not merged
BOOST_AUTO_TEST_CASE(MergeDigitsWrongBunchCrossing)
{
  std::vector<Digit> digitsOut;
  MergerDigit d1, d2;
  makeMergeableDigits(d1, d2);

  incrementBunchCrossing(d2.digit, 40);

  runMergerSameOrbit(d1, d2, digitsOut);

  // check that the input digits have not been merged
  BOOST_CHECK_EQUAL(digitsOut.size(), 2);

  // check that the two output digits are identical to the input ones
  Digit& do1 = digitsOut[0];
  Digit& do2 = digitsOut[1];
  CHECK_DIGIT(do1, d1.digit);
  CHECK_DIGIT(do2, d2.digit);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
