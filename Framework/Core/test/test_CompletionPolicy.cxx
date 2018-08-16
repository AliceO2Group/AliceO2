// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework CompletionPolicy
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/CompletionPolicy.h"

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestCompletionPolicy)
{
  std::ostringstream oss;
  oss << CompletionPolicy::CompletionOp::Consume
      << CompletionPolicy::CompletionOp::Process
      << CompletionPolicy::CompletionOp::Wait
      << CompletionPolicy::CompletionOp::Discard;

  BOOST_REQUIRE_EQUAL(oss.str(), "consumeprocesswaitdiscard");
}
