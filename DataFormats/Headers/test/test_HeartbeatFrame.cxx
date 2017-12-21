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

/**
 * Helper struct to define a composite element from a header, some payload
 * and a trailer
 */
template <typename HeaderT
          , typename TrailerT>
struct Composite {
  using HeaderType = HeaderT;
  using TrailerType = TrailerT;
  size_t length = 0;

  template<size_t N>
  constexpr Composite(const HeaderType& h, const char  (&d)[N], const TrailerType& t)
    : header(h)
    , data(d)
    , trailer(t)
  {
    length = sizeof(HeaderType) + N + sizeof(TrailerType);
  }

  constexpr size_t getDataLength() const noexcept {
    return length - sizeof(HeaderType) - sizeof(TrailerType);
  }

  const HeaderType& header;
  const char* data = nullptr;
  const TrailerType& trailer;
};

/// recursively calculate the length of the sequence
template <typename T, typename... TArgs>
constexpr size_t sequenceLength(const T& first, const TArgs... args) noexcept {
  return sequenceLength(first) + sequenceLength(args...);
}

/// terminating template secialization of sequence length calculation
template <typename T>
constexpr size_t sequenceLength(const T& first) noexcept {
  return first.length;
}

/// recursive insert of variable number of elements
template <typename T, typename... TArgs>
constexpr size_t sequenceInsert(byte* buffer, const T& first, const TArgs... args) noexcept {
  size_t length = sequenceLength(first);
  sequenceInsert(buffer, first);
  buffer += length;
  length += sequenceInsert(buffer, args...);
  return length;
}

/// terminating template specialization, i.e. for the last element
template <typename T>
constexpr size_t sequenceInsert(byte* buffer, const T& element) noexcept {
  size_t length = 0;
  memcpy(buffer + length, &element.header, sizeof(typename T::HeaderType));
  length += sizeof(typename T::HeaderType);
  memcpy(buffer + length, element.data, element.getDataLength());
  length += element.getDataLength();
  memcpy(buffer + length, &element.trailer, sizeof(typename T::TrailerType));
  length += sizeof(typename T::TrailerType);
  return length;
}

/**
 * Helper struct to create a buffer from multiple blocks
 */
struct TestFrame {
  using BufferType = std::unique_ptr<byte[]>;

  BufferType buffer;
  size_t length;

  TestFrame() = delete;

  template <typename CompositeType, typename... Targs>
  TestFrame(CompositeType block, Targs... args)
  {
    length = sequenceLength(block, args...);
    std::cout << "make TestFrame length " << length << std::endl;
    buffer  = std::make_unique<byte[]>(length);
    sequenceInsert(buffer.get(), block, args...);
  }
};

BOOST_AUTO_TEST_CASE(test_parser)
{
  using FrameT = Composite<HeartbeatHeader, HeartbeatTrailer>;
  // note: the length of the data is set in the trailer word
  TestFrame tf(FrameT({0x1100000000000000}, "heartbeatdata", {0x510000000000000e}),
               FrameT({0x1100000000000001}, "test", {0x5100000000000005}),
               FrameT({0x1100000000000003}, "dummydata", {0x510000000000000a})
               );
  o2::header::hexDump("Test frame", tf.buffer.get(), tf.length);

  using ParserT = o2::header::ReverseParser<typename FrameT::HeaderType,
                                            typename FrameT::TrailerType>;
  ParserT parser;
  parser.parse(tf.buffer.get(), tf.length,
               [](const typename ParserT::HeaderType* header) {return (*header);},
               [](const typename ParserT::TrailerType* trailer) {return (*trailer);},
               [](const typename ParserT::TrailerType& trailer) {
                 return trailer.dataLength + ParserT::envelopeLength;
               },
               [](typename ParserT::FrameEntry entry) {
                 o2::header::hexDump("Entry", entry.payload, entry.length);
                 return true;
               }
               );
}

BOOST_AUTO_TEST_CASE(test_heartbeat_sequence)
{
  using FrameT = Composite<HeartbeatHeader, HeartbeatTrailer>;
  // note: the length of the data is set in the trailer word
  TestFrame tf1(FrameT({0x1100000000000000}, "heartbeatdata", {0x510000000000000e}),
                FrameT({0x1100000000000001}, "test", {0x5100000000000005}),
                FrameT({0x1100000000000003}, "dummydata", {0x510000000000000a})
                );
  TestFrame tf2(FrameT({0x1100000000000000}, "frame2a", {0x5100000000000008}),
                FrameT({0x1100000000000002}, "frame2b", {0x5100000000000008}),
                FrameT({0x1100000000000003}, "frame2c", {0x5100000000000008})
                );

  o2::header::HeartbeatFrameSequence<o2::header::DataHeader> seqHandler;

  //check iterators of the empty handler
  BOOST_CHECK(seqHandler.begin() == seqHandler.end());

  o2::header::DataHeader dh;
  dh.dataDescription = o2::header::DataDescription("FIRSTSLOT");
  dh.dataOrigin = o2::header::DataOrigin("TST");
  dh.subSpecification = 0;
  dh.payloadSize = 0;

  seqHandler.addSlot(dh, tf1.buffer.get(), tf1.length);
  seqHandler.addSlot(dh, tf2.buffer.get(), tf2.length);

  std::cout << "slots: " << seqHandler.getNSlots() << " columns: " << seqHandler.getNColumns() << std::endl;

  for (auto columnIt = seqHandler.begin(), end = seqHandler.end();
       columnIt != end; ++columnIt) {
    std::cout << "---------------------------------------" << std::endl;
    for (auto row : columnIt) {
      o2::header::hexDump("Entry", row.buffer, row.size);
    }
  }
}
