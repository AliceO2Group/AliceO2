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
    rdh->version = 4;
    rdh->headerSize = sizeof(V5);
    rdh->offsetToNext = PageSize;
    auto* data = reinterpret_cast<size_t*>(buffer.data() + pageNo * PageSize + rdh->headerSize);
    *data = pageNo;
  }

  size_t count = 0;
  auto processor = [&count](auto data, size_t length) {
    BOOST_CHECK(length == PageSize - sizeof(V5));
    BOOST_CHECK(*reinterpret_cast<size_t const*>(data) == count);
    std::cout << "Processing block of length " << length << std::endl;
    count++;
  };
  RawParser parser(buffer.data(), buffer.size());
  parser.parse(processor);
  BOOST_CHECK(count == NofPages);

  parser.reset();

  count = 0;
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it, ++count) {
    BOOST_CHECK(it.length() == PageSize - sizeof(V5));
    BOOST_CHECK(*reinterpret_cast<size_t const*>(it.data()) == count);
    //BOOST_CHECK(it.as<V5>() != nullptr);
    //BOOST_CHECK(it.as<V4>() == nullptr);
    std::cout << "Iterating block of length " << it.length() << std::endl;
  }
}

} // namespace o2::framework
