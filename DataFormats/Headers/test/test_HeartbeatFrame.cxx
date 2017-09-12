// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test HeartbeatFrame
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"

using DataHeader = o2::header::DataHeader;
using HeartbeatHeader = o2::header::HeartbeatHeader;
using HeartbeatTrailer = o2::header::HeartbeatTrailer;

BOOST_AUTO_TEST_CASE(test_heartbeatframe)
{
  HeartbeatHeader header;
  HeartbeatTrailer trailer;

  // checking the consistency operators
  BOOST_CHECK(header);
  BOOST_CHECK(trailer);

  // checking the block type identifier
  BOOST_CHECK(header.blockType == 1);
  BOOST_CHECK(trailer.blockType == 5);

  // checking length
  BOOST_CHECK(header.headerLength == 1);
  BOOST_CHECK(trailer.trailerLength == 1);
}
