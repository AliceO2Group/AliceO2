// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test BasicHits class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include <unordered_map>
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2;

BOOST_AUTO_TEST_CASE(MCCompLabel_test)
{
  MCCompLabel lbUndef;
  BOOST_CHECK(!lbUndef.isSet()); // test invalid label status

  int ev = 200, src = 10;
  std::unordered_map<MCCompLabel, int> labelMap;
  for (int tr=-100;tr<200;tr+=150) {
    MCCompLabel lb(std::abs(tr), ev, src, tr < 0);
    std::cout << "Input:   [" << src << '/' << ev << '/'
              << std::setw(6) << tr << ']' << std::endl;
    std::cout << "Encoded: " << lb << " (packed: " << ULong_t(lb) << ")" << std::endl;
    labelMap[lb] = tr;
    int trE, evE, srcE;
    bool fake;
    lb.get(trE, evE, srcE, fake);
    std::cout << "Decoded: [" << srcE << '/' << evE << '/'
              << std::setw(6) << (fake ? '-' : '+') << trE << ']' << std::endl;

    BOOST_CHECK((fake && (tr == -trE)) || (!fake && (tr == trE)) && ev == evE && src == srcE);
  }

  for (auto& [key, value] : labelMap) {
    BOOST_CHECK(key.getTrackIDSigned() == value);
    BOOST_CHECK(key.getTrackID() == std::abs(value) && ((value < 0) == key.isFake()));
  }

  MCCompLabel noise(true);
  BOOST_CHECK(noise.isNoise() && !noise.isEmpty() && noise.isFake() && !noise.isValid());
  MCCompLabel dummy;
  BOOST_CHECK(dummy.isEmpty() && !dummy.isNoise() && dummy.isFake() && !dummy.isValid());
}
