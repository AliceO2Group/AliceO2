// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCEventLabel class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "Framework/Logger.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include <TRandom.h>

using namespace o2;

BOOST_AUTO_TEST_CASE(MCEventLabel_test)
{
  MCEventLabel lbUndef;
  BOOST_CHECK(!lbUndef.isSet()); // test invalid label status

  for (int itr = 0; itr < 100; itr++) {
    int ev = gRandom->Integer(MCEventLabel::MaxEventID()), src = gRandom->Integer(MCEventLabel::MaxSourceID());
    float w = gRandom->Rndm();
    MCEventLabel lb(ev, src, w);
    LOG(INFO) << "Input:   [" << src << '/' << ev << '/' << w << ']';
    LOG(INFO) << "Encoded: " << lb << " (packed: " << uint32_t(lb) << ")";
    int evE, srcE;
    float wE;
    lb.get(evE, srcE, wE);
    LOG(INFO) << "Decoded: [" << srcE << '/' << evE << '/' << wE << ']';
    BOOST_CHECK(ev == evE && src == srcE && std::abs(w - wE) < MCEventLabel::WeightPrecision());
  }
}
