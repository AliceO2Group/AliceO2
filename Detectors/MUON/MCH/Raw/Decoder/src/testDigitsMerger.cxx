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


static Digit::Time makeTime(int sampaTime, int bunchCrossing, int orbit)
{
  Digit::Time time;
  time.sampaTime = sampaTime;
  time.bunchCrossing = bunchCrossing;
  time.orbit = orbit;
  return time;
}


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
  Digit::Time time = makeTime(sampaTime, bunchCrossing, orbit);

  d1 = MergerDigit{o2::mch::Digit(deId, padId, adc, time, nSamples), false, solarId, dsAddr, chAddr};

  int adc2 = 10;
  int sampaTime2 = 0;
  int bunchCrossing2 = sampaWindowSize * 4;
  int nSamples2 = 10;
  Digit::Time time2 = makeTime(sampaTime2, bunchCrossing2, orbit);

  d2 = MergerDigit{o2::mch::Digit(deId, padId, adc2, time2, nSamples2), false, solarId, dsAddr, chAddr};
}


static void addDigit(Merger& merger, int feeId, MergerDigit& d)
{
  merger.addDigit(feeId, d.solarId, d.dsAddr, d.chAddr, d.digit.getDetID(), d.digit.getPadID(), d.digit.getADC(), d.digit.getTime(), d.digit.nofSamples());
}


BOOST_AUTO_TEST_CASE(MergeDigits)
{
  MergerDigit d1, d2;
  makeMergeableDigits(d1, d2);

  int feeId = 0;
  int orbit = d1.digit.getTime().orbit;

  std::vector<Digit> digitsOut;
  const auto storeDigit = [&](const Digit& d) {
    digitsOut.emplace_back(d);
  };

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

  // check that the input digits have been merged into a single one with the expected parameters
  BOOST_CHECK_EQUAL(digitsOut.size(), 1);

  Digit& d = digitsOut[0];
  BOOST_CHECK_EQUAL(d.getDetID(), d1.digit.getDetID());
  BOOST_CHECK_EQUAL(d.getPadID(), d1.digit.getPadID());

  // check that the merged digit has the same time as the fist input digit
  BOOST_CHECK_EQUAL(d.getTime().sampaTime, d1.digit.getTime().sampaTime);
  BOOST_CHECK_EQUAL(d.getTime().bunchCrossing, d1.digit.getTime().bunchCrossing);
  BOOST_CHECK_EQUAL(d.getTime().orbit, d1.digit.getTime().orbit);

  // check that the total charge of the merged digit is equal to the sum of the individual charges
  BOOST_CHECK_EQUAL(d.getADC(), d1.digit.getADC() + d2.digit.getADC());

  // check that the number of samples of the merged digit is equal to the sum of the individual digits
  BOOST_CHECK_EQUAL(d.nofSamples(), d1.digit.nofSamples() + d2.digit.nofSamples());
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
