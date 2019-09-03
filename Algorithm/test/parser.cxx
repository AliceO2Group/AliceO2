// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   parser.cxx
/// @author Matthias Richter
/// @since  2017-09-20
/// @brief  Unit test for data parsing methods in Algorithm/Parser.h

#define BOOST_TEST_MODULE Test Algorithm Parser
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../include/Algorithm/Parser.h"
#include "StaticSequenceAllocator.h"

// header test class
struct Header {
  unsigned identifier = 0xdeadbeef;
  size_t payloadSize = 0;

  Header(size_t ps) : payloadSize(ps) {}
};

// trailer test class
struct Trailer {
  unsigned identifier = 0xaaffee00;
  unsigned char flags = 0xaa;

  Trailer(unsigned char f) : flags(f) {}
};

// trailer test class including payload size
struct SizedTrailer {
  unsigned identifier = 0xaaffee00;
  unsigned char flags = 0xaa;
  size_t payloadSize = 0;

  SizedTrailer(size_t s, unsigned char f) : payloadSize(s), flags(f) {}
};

BOOST_AUTO_TEST_CASE(test_forwardparser_header_and_trailer)
{
  using FrameT = o2::algorithm::Composite<Header, Trailer>;
  // note: the length of the data is set in the header word
  using TestFrame = o2::algorithm::StaticSequenceAllocator;
  TestFrame tf(FrameT(16, "lotsofsillydata", 0xaa),
               FrameT(5, "test", 0xcc),
               FrameT(10, "dummydata", 0x33));

  using ParserT = o2::algorithm::ForwardParser<typename FrameT::HeaderType,
                                               typename FrameT::TrailerType>;

  auto checkHeader = [](const typename FrameT::HeaderType& header) {
    return header.identifier == 0xdeadbeef;
  };
  auto checkTrailer = [](const typename FrameT::TrailerType& trailer) {
    return trailer.identifier == 0xaaffee00;
  };
  auto getFrameSize = [](const typename ParserT::HeaderType& header) {
    // frame size includes total offset from header and trailer
    return header.payloadSize + ParserT::totalOffset;
  };

  std::vector<typename ParserT::FrameInfo> frames;
  auto insert = [&frames](typename ParserT::FrameInfo& info) {
    frames.emplace_back(info);
    return true;
  };

  ParserT parser;
  auto result = parser.parse(tf.buffer.get(), tf.size(),
                             checkHeader,
                             checkTrailer,
                             getFrameSize,
                             insert);

  BOOST_REQUIRE(result == 3);
  BOOST_REQUIRE(frames.size() == 3);

  BOOST_CHECK(memcmp(frames[0].payload, "lotsofsillydata", frames[0].length) == 0);
  BOOST_CHECK(memcmp(frames[1].payload, "test", frames[1].length) == 0);
  BOOST_CHECK(memcmp(frames[2].payload, "dummydata", frames[2].length) == 0);
}

BOOST_AUTO_TEST_CASE(test_forwardparser_header_and_void_trailer)
{
  using FrameT = o2::algorithm::Composite<Header>;
  // note: the length of the data is set in the header word
  using TestFrame = o2::algorithm::StaticSequenceAllocator;
  TestFrame tf(FrameT(16, "lotsofsillydata"),
               FrameT(5, "test"),
               FrameT(10, "dummydata"));

  using ParserT = o2::algorithm::ForwardParser<typename FrameT::HeaderType,
                                               typename FrameT::TrailerType>;

  auto checkHeader = [](const typename FrameT::HeaderType& header) {
    return header.identifier == 0xdeadbeef;
  };

  auto getFrameSize = [](const typename ParserT::HeaderType& header) {
    // frame size includes total offset from header and trailer
    return header.payloadSize + ParserT::totalOffset;
  };

  std::vector<typename ParserT::FrameInfo> frames;
  auto insert = [&frames](typename ParserT::FrameInfo& info) {
    frames.emplace_back(info);
    return true;
  };

  ParserT parser;
  auto result = parser.parse(tf.buffer.get(), tf.size(),
                             checkHeader,
                             getFrameSize,
                             insert);

  BOOST_REQUIRE(result == 3);
  BOOST_REQUIRE(frames.size() == 3);

  BOOST_CHECK(memcmp(frames[0].payload, "lotsofsillydata", frames[0].length) == 0);
  BOOST_CHECK(memcmp(frames[1].payload, "test", frames[1].length) == 0);
  BOOST_CHECK(memcmp(frames[2].payload, "dummydata", frames[2].length) == 0);
}

BOOST_AUTO_TEST_CASE(test_forwardparser_no_frames)
{
  using FrameT = o2::algorithm::Composite<Header>;
  // note: the length of the data is set in the header word
  using TestFrame = o2::algorithm::StaticSequenceAllocator;
  TestFrame tf(FrameT(16, "lotsofsillydata"),
               FrameT(5, "test"),
               FrameT(10, "dummydata"));

  using ParserT = o2::algorithm::ForwardParser<typename FrameT::HeaderType,
                                               typename FrameT::TrailerType>;

  auto checkHeader = [](const typename FrameT::HeaderType& header) {
    // simply indicate invalid header to read no frames
    return false;
  };

  auto getFrameSize = [](const typename ParserT::HeaderType& header) {
    // frame size includes total offset from header and trailer
    return header.payloadSize + ParserT::totalOffset;
  };

  std::vector<typename ParserT::FrameInfo> frames;
  auto insert = [&frames](typename ParserT::FrameInfo& info) {
    frames.emplace_back(info);
    return true;
  };

  ParserT parser;
  auto result = parser.parse(tf.buffer.get(), tf.size(),
                             checkHeader,
                             getFrameSize,
                             insert);

  // check that there are really no frames found
  BOOST_REQUIRE(result == 0);
}

BOOST_AUTO_TEST_CASE(test_forwardparser_format_error)
{
  using FrameT = o2::algorithm::Composite<Header>;
  // note: the length of the data is set in the header word
  using TestFrame = o2::algorithm::StaticSequenceAllocator;
  TestFrame tf(FrameT(16, "lotsofsillydata"),
               FrameT(4, "test"), // <- note wrong size
               FrameT(10, "dummydata"));

  using ParserT = o2::algorithm::ForwardParser<typename FrameT::HeaderType,
                                               typename FrameT::TrailerType>;

  auto checkHeader = [](const typename FrameT::HeaderType& header) {
    return header.identifier == 0xdeadbeef;
  };

  auto getFrameSize = [](const typename ParserT::HeaderType& header) {
    // frame size includes total offset from header and trailer
    return header.payloadSize + ParserT::totalOffset;
  };

  std::vector<typename ParserT::FrameInfo> frames;
  auto insert = [&frames](typename ParserT::FrameInfo& info) {
    frames.emplace_back(info);
    return true;
  };

  ParserT parser;
  auto result = parser.parse(tf.buffer.get(), tf.size(),
                             checkHeader,
                             getFrameSize,
                             insert);

  BOOST_REQUIRE(result == -1);
}

BOOST_AUTO_TEST_CASE(test_reverseparser)
{
  using FrameT = o2::algorithm::Composite<Header, SizedTrailer>;
  // note: the length of the data is set in the trailer word
  using TestFrame = o2::algorithm::StaticSequenceAllocator;
  TestFrame tf(FrameT(0, "lotsofsillydata", {16, 0xaa}),
               FrameT(0, "test", {5, 0xcc}),
               FrameT(0, "dummydata", {10, 0x33}));

  using ParserT = o2::algorithm::ReverseParser<typename FrameT::HeaderType,
                                               typename FrameT::TrailerType>;

  auto checkHeader = [](const typename FrameT::HeaderType& header) {
    return header.identifier == 0xdeadbeef;
  };
  auto checkTrailer = [](const typename FrameT::TrailerType& trailer) {
    return trailer.identifier == 0xaaffee00;
  };
  auto getFrameSize = [](const typename ParserT::TrailerType& trailer) {
    return trailer.payloadSize + ParserT::totalOffset;
  };

  std::vector<typename ParserT::FrameInfo> frames;
  auto insert = [&frames](const typename ParserT::FrameInfo& info) {
    frames.emplace_back(info);
    return true;
  };

  ParserT parser;
  auto result = parser.parse(tf.buffer.get(), tf.size(),
                             checkHeader,
                             checkTrailer,
                             getFrameSize,
                             insert);

  BOOST_REQUIRE(result == 3);
  BOOST_REQUIRE(frames.size() == 3);

  BOOST_CHECK(memcmp(frames[2].payload, "lotsofsillydata", frames[2].length) == 0);
  BOOST_CHECK(memcmp(frames[1].payload, "test", frames[1].length) == 0);
  BOOST_CHECK(memcmp(frames[0].payload, "dummydata", frames[0].length) == 0);
}
