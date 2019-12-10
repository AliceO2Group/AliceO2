// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Utils RawParser
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DPLUtils/RawParser.h"
#include <iostream>

namespace o2::framework
{

BOOST_AUTO_TEST_CASE(test_RawParser)
{
  using V5 = header::RAWDataHeaderV5;
  using V4 = header::RAWDataHeaderV4;
  constexpr size_t PageSize = 8192;
  constexpr size_t NofPages = 3;
  std::array<unsigned char, NofPages * PageSize> buffer;
  for (int pageNo = 0; pageNo < buffer.size() / PageSize; pageNo++) {
    auto* rdh = reinterpret_cast<V5*>(buffer.data() + pageNo * PageSize);
    rdh->version = 5;
    rdh->headerSize = sizeof(V5);
    rdh->offsetToNext = PageSize;
    rdh->pageCnt = NofPages;
    rdh->packetCounter = pageNo;
    rdh->stop = pageNo + 1 == NofPages;
    auto* data = reinterpret_cast<size_t*>(buffer.data() + pageNo * PageSize + rdh->headerSize);
    *data = pageNo;
  }

  size_t count = 0;
  auto processor = [&count](auto data, size_t size) {
    BOOST_CHECK(size == PageSize - sizeof(V5));
    BOOST_CHECK(*reinterpret_cast<size_t const*>(data) == count);
    std::cout << "Processing block of size " << size << std::endl;
    count++;
  };
  RawParser parser(buffer.data(), buffer.size());
  parser.parse(processor);
  BOOST_CHECK(count == NofPages);

  parser.reset();

  std::cout << parser << std::endl;
  count = 0;
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it, ++count) {
    BOOST_CHECK(it.size() == PageSize - sizeof(V5));
    BOOST_CHECK(*reinterpret_cast<size_t const*>(it.data()) == count);
    BOOST_CHECK(it.get_if<V5>() != nullptr);
    BOOST_CHECK(it.get_if<V4>() == nullptr);
    BOOST_CHECK(it.raw() + it.offset() == it.data());
    std::cout << it << ": block size " << it.size() << std::endl;
  }
}

} // namespace o2::framework
