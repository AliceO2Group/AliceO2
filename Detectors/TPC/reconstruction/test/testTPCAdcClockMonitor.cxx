// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCAdcClockMonitor.cxx
/// \brief This task tests the AdcClockMonitor.cxx module of the TPC GBT frame reader
/// \author Sebastian Klewin

#define BOOST_TEST_MODULE Test TPC AdcClockMonitor
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCReconstruction/AdcClockMonitor.h"

#include <vector>
#include <iostream>
#include <iomanip>

namespace o2
{
namespace tpc
{

/// @brief Test 1 basic class IO tests
BOOST_AUTO_TEST_CASE(AdcClockMonitor_test1)
{
  AdcClockMonitor mon(0);

  BOOST_CHECK_EQUAL(mon.getState(), 1);
}

/// @brief Test 2 valid sequences
BOOST_AUTO_TEST_CASE(AdcClockMonitor_test2)
{
  // Instantiate clock monitpr
  AdcClockMonitor mon(0);

  // do test for all possible clock phases
  for (unsigned phase = 0; phase < 4; ++phase) {

    // prepare valid clock sequence
    std::vector<unsigned short> clockSequence(10, 0);
    unsigned clock = (0xFFFF0000 >> phase);
    for (int i = 0; i < 50; ++i) {
      clockSequence.emplace_back((clock >> 28) & 0xF);
      clock = (clock << 4) | (clock >> 28);
    }

    // feed monitor with clock sequence
    mon.reset();
    int seq;
    for (std::vector<unsigned short>::iterator it = clockSequence.begin(); it != clockSequence.end(); ++it) {
      seq = std::distance(clockSequence.begin(), it);
      // only the 18th sequence completes the full cycle
      // 10 times 000 in the beginning + 8 sequences with needed parts to recognize full pattern
      BOOST_CHECK_EQUAL(mon.addSequence(*it), (seq >= 18 ? 0 : 1));
    }
  }
}

/// @brief Test 3 valid sequences, 1 pattern replaced with something else
BOOST_AUTO_TEST_CASE(AdcClockMonitor_test3)
{
  // Instantiate clock monitpr
  AdcClockMonitor mon(0);

  // do test for all possible clock phases
  for (unsigned phase = 0; phase < 4; ++phase) {

    // prepare valid clock sequence
    std::vector<unsigned short> clockSequence(10, 0);
    unsigned clock = (0xFFFF0000 >> phase);
    for (int i = 0; i < 50; ++i) {
      clockSequence.emplace_back((clock >> 28) & 0xF);
      clock = (clock << 4) | (clock >> 28);
    }

    // feed monitor with clock sequence, replace beforehand one sequence of first pattern with something else
    unsigned short oldSeq;
    for (int pos = 10; pos < 18; ++pos) {
      mon.reset();
      oldSeq = clockSequence[pos];
      clockSequence[pos] = 0xA;
      int seq;
      for (std::vector<unsigned short>::iterator it = clockSequence.begin(); it != clockSequence.end(); ++it) {
        seq = std::distance(clockSequence.begin(), it);
        // only the 26th sequence completes the full cycle
        // 10 times 000 in the beginning
        // + 8 sequences with with wrong part
        // + 8 sequences with correct parts to recognize full pattern

        // one special case for phase = 0 and position of wrong part = 17 (last needed part for full pattern)
        // phase 0 means the following pattern            0000 FFFF 0000 FFFF ...
        //                                                             ^
        //                                                             |
        // part 17 wrong means this one wrong (not 0)     -------------|
        //
        // The finding algorithm neads two clean transitions from 0 to F to recognize
        // a clock with phase 0. If part 17 is wrong, we won't have this clean transition
        // at this position, therefore the the clock will be recognized one complete
        // pattern later. For every other wrong part, we still habe here the first transition
        // from 0 to F and for every other phase, we don't need two sequences for the
        // transition just the one with the transition
        int expGoodSequence = (phase == 0 && pos == 17) ? (18 + 8 + 8) : (18 + 8);

        BOOST_CHECK_EQUAL(mon.addSequence(*it), (seq >= expGoodSequence ? 0 : 1));
      }
      clockSequence[pos] = oldSeq;
    }
  }
}

} // namespace tpc
} // namespace o2
