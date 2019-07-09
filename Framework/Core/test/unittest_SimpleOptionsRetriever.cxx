// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Framework SimpleOptionsRetriever
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/SimpleOptionsRetriever.h"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TestInsertion)
{
  o2::framework::SimpleOptionsRetriever options{};

  options.setInt("foo", 123);
  BOOST_CHECK_EQUAL(options.getInt("foo"), 123);
}
