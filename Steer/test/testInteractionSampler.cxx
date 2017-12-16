// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test InteractionSampler class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/MCInteractionRecord.h"
#include "Steer/InteractionSampler.h"

namespace o2
{
BOOST_AUTO_TEST_CASE(InteractionSampler)
{
  using Sampler = o2::steer::InteractionSampler;

  const int ntest = 100;
  std::vector<o2::MCInteractionRecord> records; // destination for records
  records.reserve(ntest);

  printf("Testing sampler with default settings\n");
  // default sampler with BC filling like in TPC TDR, 50kHz
  Sampler defSampler;
  defSampler.init();
  defSampler.generateCollisionTimes(records);
  double t = -1.;
  for (const auto& rec : records) {
    BOOST_CHECK(rec.timeNS >= t); // make sure time is non-decreasing
    t = rec.timeNS;
  }

  printf("\nTesting sampler with custom bunch filling and low mu\n");
  // configure sampler with custom bunch filling and mu per BC
  Sampler sampler1;
  // train of 100 bunches spaced by 25 ns (1slot) and staring at BC=0
  sampler1.setBCTrain(100, 1, 0);
  // train of 100 bunches spaced by 50 ns (2slots) and staring at BC=200
  sampler1.setBCTrain(200, 2, 200);
  // add isolated BC at slot 1600
  sampler1.setBC(1600);
  // add 5 trains of 20 bunches with 100ns(4slots) spacing, separated by 10 slots and
  // starting at bunch 700
  sampler1.setBCTrains(5, 10, 20, 4, 700);
  // set total interaction rate in Hz
  sampler1.setInteractionRate(40e3);
  sampler1.init();
  sampler1.generateCollisionTimes(records);
  t = -1.;
  for (const auto& rec : records) {
    BOOST_CHECK(rec.timeNS >= t); // make sure time is non-decreasing
    t = rec.timeNS;
  }

  // reconfigure w/o modifying BC filling but setting per bunch
  // mu (large -> lot of in-bunch pile-up)
  printf("\nResetting/testing sampler with same bunch filling but high mu\n");
  sampler1.setMuPerBC(0.5);
  sampler1.init(); // this will reset all counters from previous calls
  // instead of filling the vector records, we can sample one by one
  t = -1.;
  for (int i = 0; i < ntest; i++) {
    auto rec = sampler1.generateCollisionTime();
    rec.print();
    // make sure time is non-decreasing and the BC is interacting
    BOOST_CHECK(rec.timeNS >= t && sampler1.getBC(rec.bc));
    t = rec.timeNS;
  }
  sampler1.print();
  sampler1.printBunchFilling();
}
}
