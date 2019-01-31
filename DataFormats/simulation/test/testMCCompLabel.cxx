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
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2;

BOOST_AUTO_TEST_CASE(MCCompLabel_test)
{
  MCCompLabel lbUndef;
  BOOST_CHECK(!lbUndef.isSet()); // test invalid label status

  int ev = 200, src = 10;
  for (int tr=-100;tr<200;tr+=150) {
    MCCompLabel lb(tr, ev, src);
    std::cout << "Input:   [" << src << '/' << ev << '/'
              << std::setw(6) << tr << ']' << std::endl;
    std::cout << "Encoded: " << lb << " (packed: " << ULong_t(lb) << ")" << std::endl;
    int trE, evE, srcE;
    lb.get(trE, evE, srcE);
    std::cout << "Decoded: [" << srcE << '/' << evE << '/'
              << std::setw(6) << trE << ']' << std::endl;

    BOOST_CHECK(tr == trE && ev == evE && src == srcE);
  }
}
