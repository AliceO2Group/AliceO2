// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   o2formatparser.cxx
/// @author Matthias Richter
/// @since  2017-10-18
/// @brief  Unit test for O2 format parser

#define BOOST_TEST_MODULE Test Algorithm HeaderStack
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>              // memcmp
#include "Headers/DataHeader.h" // hexdump, DataHeader
#include "../include/Algorithm/O2FormatParser.h"

template <typename... Targs>
void hexDump(Targs... Fargs)
{
  // a simple redirect to enable/disable the hexdump printout
  o2::header::hexDump(Fargs...);
}

BOOST_AUTO_TEST_CASE(test_o2formatparser)
{
  std::vector<const char*> thedata = {
    "I'm raw data",
    "reconstructed data"};
  unsigned dataidx = 0;
  std::vector<o2::header::DataHeader> dataheaders;
  dataheaders.emplace_back(o2::header::DataDescription("RAWDATA"),
                           o2::header::DataOrigin("DET"),
                           0,
                           strlen(thedata[dataidx++]));
  dataheaders.emplace_back(o2::header::DataDescription("RECODATA"),
                           o2::header::DataOrigin("DET"),
                           0,
                           strlen(thedata[dataidx++]));

  std::vector<std::pair<const char*, size_t>> messages;
  for (dataidx = 0; dataidx < thedata.size(); ++dataidx) {
    messages.emplace_back(reinterpret_cast<char*>(&dataheaders[dataidx]),
                          sizeof(o2::header::DataHeader));
    messages.emplace_back(thedata[dataidx],
                          dataheaders[dataidx].payloadSize);
  }

  // handler callback for parseO2Format method
  auto insertFct = [&](const auto& dataheader,
                       auto ptr,
                       auto size) {
    hexDump("header", &dataheader, sizeof(dataheader));
    hexDump("data", ptr, size);
    BOOST_CHECK(dataheader == dataheaders[dataidx]);
    BOOST_CHECK(strncmp(ptr, thedata[dataidx], size) == 0);
    ++dataidx;
  }; // end handler callback

  // handler callback to get the pointer for message
  auto getPointerFct = [](auto arg) { return arg.first; };
  // handler callback to get the size for message
  auto getSizeFct = [](auto arg) { return arg.second; };

  dataidx = 0;
  auto result = o2::algorithm::parseO2Format(messages,
                                             getPointerFct,
                                             getSizeFct,
                                             insertFct);

  BOOST_REQUIRE(result >= 0);
  BOOST_CHECK(result == 2);
}
