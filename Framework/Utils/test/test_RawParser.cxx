// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Utils RawParser
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include "DPLUtils/RawParser.h"
#include <iostream>

namespace o2::framework
{

constexpr size_t PageSize = 8192;
using V7 = header::RAWDataHeaderV7;
using V6 = header::RAWDataHeaderV6;
using V5 = header::RAWDataHeaderV5;
using V4 = header::RAWDataHeaderV4;

template <typename RDH, typename Container>
void fillPages(Container& buffer)
{
  RDH defaultHeader;
  size_t const NofPages = buffer.size() / PageSize;
  memset(buffer.data(), 0, buffer.size());
  for (int pageNo = 0; pageNo < NofPages; pageNo++) {
    auto* rdh = reinterpret_cast<RDH*>(buffer.data() + pageNo * PageSize);
    rdh->version = defaultHeader.version;
    rdh->headerSize = sizeof(RDH);
    rdh->offsetToNext = PageSize;
    rdh->memorySize = PageSize;
    rdh->pageCnt = NofPages;
    rdh->packetCounter = pageNo;
    rdh->stop = pageNo + 1 == NofPages;
    auto* data = reinterpret_cast<size_t*>(buffer.data() + pageNo * PageSize + rdh->headerSize);
    *data = pageNo;
  }
}

typedef boost::mpl::list<V5, V6, V7> testTypes;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_RawParser, RDH, testTypes)
{
  constexpr size_t NofPages = 3;
  std::array<unsigned char, NofPages * PageSize> buffer;
  fillPages<RDH>(buffer);

  size_t count = 0;
  auto processor = [&count](auto data, size_t size) {
    BOOST_CHECK(size == PageSize - sizeof(RDH));
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
    BOOST_CHECK(it.size() == PageSize - sizeof(RDH));
    BOOST_CHECK(*reinterpret_cast<size_t const*>(it.data()) == count);
    // FIXME: there is a problem with invoking get_if<T>, but only if the code is templatized
    // and called from within boost unit test macros.
    //    expected primary-expression before ‘>’ token
    //    BOOST_CHECK(it.get_if<RDH>() != nullptr);
    //                             ^
    // That's why the type is deduced by argument until the problem is understood.
    BOOST_CHECK(it.get_if((RDH*)nullptr) != nullptr);
    if constexpr (std::is_same<RDH, V4>::value == false) {
      BOOST_CHECK(it.get_if((V4*)nullptr) == nullptr);
    } else {
      BOOST_CHECK(it.get_if((V5*)nullptr) == nullptr);
    }
    BOOST_CHECK(it.raw() + it.offset() == it.data());
    std::cout << it << ": block size " << it.size() << std::endl;
  }
}

} // namespace o2::framework
