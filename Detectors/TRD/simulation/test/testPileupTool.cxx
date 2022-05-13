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

#define BOOST_TEST_MODULE Test TRD Pileup Tool
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/SignalArray.h" // for SignalContainer
#include "TRDSimulation/PileupTool.h"

#include <algorithm>

namespace o2
{
namespace trd
{

BOOST_AUTO_TEST_CASE(TRDPileupTool_test)
{
  PileupTool tool;

  double triggerTime;
  std::vector<SignalArray> signalArrays(3);
  std::array<SignalContainer, constants::MAXCHAMBER> chamberSignals;
  std::deque<std::array<SignalContainer, constants::MAXCHAMBER>> pileupSignals;

  // set the signals
  signalArrays[0].firstTBtime = 0;
  signalArrays[1].firstTBtime = 2000;
  signalArrays[2].firstTBtime = 4900;
  std::fill(signalArrays[0].signals.begin(), signalArrays[0].signals.end(), 1);
  std::fill(signalArrays[1].signals.begin(), signalArrays[1].signals.end(), 1);
  std::fill(signalArrays[2].signals.begin(), signalArrays[2].signals.end(), 1);
  signalArrays[0].labels = {1}; // dummy label;
  signalArrays[1].labels = {1}; // dummy label;
  signalArrays[2].labels = {1}; // dummy label;

  triggerTime = 0;
  // simulate that chamber 0 recieves three signals at t1,...t2,...t3
  // with t1 triggering, and t2 incoming before t1+30tb, and t3 to far in the future
  // all with key = 1111
  chamberSignals[0][1111] = signalArrays[0]; // chamber 0, with key 1, first signal at t1
  pileupSignals.push_back(chamberSignals);   // pileup
  chamberSignals[0][1111] = signalArrays[1]; // chamber 0, with key 1, second signal at t2
  pileupSignals.push_back(chamberSignals);   // pileup
  chamberSignals[0][1111] = signalArrays[2]; // chamber 0, with key 1, third signal at t3
  pileupSignals.push_back(chamberSignals);   // pileup

  std::array<float, constants::TIMEBINS> expected1 = {
    01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 02, 02, 02, 02, 02, 02, 02, 02, 02, 02};

  auto result1 = tool.addSignals(pileupSignals, triggerTime);

  pileupSignals.clear();
  triggerTime = 2000;
  // simulate that chamber 0 recieves three signals at t1,...t2,...t3
  // with t2 triggering, and t1 from the past, and t3 incoming before t2+30tb
  // all with key = 1111
  chamberSignals[0][1111] = signalArrays[0]; // chamber 0, with key 1, first signal at t1
  pileupSignals.push_back(chamberSignals);   // pileup
  chamberSignals[0][1111] = signalArrays[1]; // chamber 0, with key 1, second signal at t2
  pileupSignals.push_back(chamberSignals);   // pileup
  chamberSignals[0][1111] = signalArrays[2]; // chamber 0, with key 1, third signal at t3
  pileupSignals.push_back(chamberSignals);   // pileup

  std::array<float, constants::TIMEBINS> expected2 = {
    02, 02, 02, 02, 02, 02, 02, 02, 02, 02, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 02};

  auto result2 = tool.addSignals(pileupSignals, triggerTime);

  pileupSignals.clear();
  triggerTime = 4900;
  // simulate that chamber 0 recieves three signals at t1,...t2,...t3
  // with t3 triggering, and t1 and t2 from the past
  // note that t1 is too old
  // all with key = 1111
  chamberSignals[0][1111] = signalArrays[0]; // chamber 0, with key 1, first signal at t1
  pileupSignals.push_back(chamberSignals);   // pileup
  chamberSignals[0][1111] = signalArrays[1]; // chamber 0, with key 1, second signal at t2
  pileupSignals.push_back(chamberSignals);   // pileup
  chamberSignals[0][1111] = signalArrays[2]; // chamber 0, with key 1, third signal at t3
  pileupSignals.push_back(chamberSignals);   // pileup

  std::array<float, constants::TIMEBINS> expected3 = {
    02, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01, 01};

  auto result3 = tool.addSignals(pileupSignals, triggerTime);

  BOOST_TEST(result1[1111].signals == expected1, boost::test_tools::per_element());
  BOOST_TEST(result2[1111].signals == expected2, boost::test_tools::per_element());
  BOOST_TEST(result3[1111].signals == expected3, boost::test_tools::per_element());
}

} // namespace trd
} // namespace o2
