// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   headerstack.cxx
/// @author Matthias Richter
/// @since  2017-09-19
/// @brief  Unit test for O2 header stack utilities

#define BOOST_TEST_MODULE Test Algorithm HeaderStack
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <cstring> // memcmp
#include "Headers/DataHeader.h" // hexdump
#include "Headers/NameHeader.h"
#include "../include/Algorithm/HeaderStack.h"

using DataHeader = o2::header::DataHeader;
using HeaderStack = o2::header::Stack;

BOOST_AUTO_TEST_CASE(test_headerstack)
{
  // make a header stack consisting of two O2 headers and extract them
  // via function calls using dispatchHeaderStackCallback, and through object
  // references using parseHeaderStack
  o2::header::DataHeader dh;
  dh.dataDescription = o2::header::DataDescription("SOMEDATA");
  dh.dataOrigin = o2::header::DataOrigin("TST");
  dh.subSpecification = 0;
  dh.payloadSize = 0;

  using Name8Header = o2::header::NameHeader<8>;
  Name8Header nh("NAMEDHDR");

  o2::header::Stack stack(dh, nh);

  // check that the call without any other arguments is compiling
  o2::algorithm::dispatchHeaderStackCallback(stack.buffer.get(), stack.bufferSize);

  // lambda functor given as argument for dispatchHeaderStackCallback
  auto checkDataHeader = [&dh] (const auto & header) {
    o2::header::hexDump("Extracted DataHeader", &header, sizeof(header));
    BOOST_CHECK(header == dh);
  };

  // lambda functor given as argument for dispatchHeaderStackCallback
  auto checkNameHeader = [&nh] (const auto & header) {
    o2::header::hexDump("Extracted NameHeader", &header, sizeof(header));
    // have to compare on byte level, no operator==
    BOOST_CHECK(memcmp(&header, &nh, sizeof(header)) == 0);
  };

  // check extraction of headers via callbacks
  o2::algorithm::dispatchHeaderStackCallback(stack.buffer.get(), stack.bufferSize,
                                             o2::header::DataHeader(),
                                             checkDataHeader,
                                             Name8Header(),
                                             checkNameHeader
                                             );

  // check extraction of only one header via callback
  o2::algorithm::dispatchHeaderStackCallback(stack.buffer.get(), stack.bufferSize,
                                             Name8Header(),
                                             checkNameHeader
                                             );

  // check that the call without any other arguments is compiling
  o2::algorithm::parseHeaderStack(stack.buffer.get(), stack.bufferSize);

  // check extraction of headers via object references
  o2::header::DataHeader targetDataHeader;
  Name8Header targetNameHeader;
  o2::algorithm::parseHeaderStack(stack.buffer.get(), stack.bufferSize,
                                  targetDataHeader,
                                  targetNameHeader
                                  );

  BOOST_CHECK(targetDataHeader == dh);
  BOOST_CHECK(memcmp(&targetNameHeader, &nh, sizeof(targetNameHeader)) == 0);
}
