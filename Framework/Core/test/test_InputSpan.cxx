// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework InputSpan
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/InputSpan.h"
#include "Framework/DataRef.h"
#include <vector>
#include <string>
#include <boost/test/unit_test.hpp>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestInputSpan)
{
  std::vector<std::vector<std::string>> inputs(3);
  int routeNo = 0;
  for (auto& list : inputs) {
    int nParts = routeNo != 1 ? 1 : 3;
    for (size_t part = 0; part < nParts; ++part) {
      list.emplace_back("header_" + std::to_string(routeNo) + "-" + std::to_string(part));
      list.emplace_back("payload_" + std::to_string(routeNo) + "-" + std::to_string(part));
    }
    routeNo++;
  }

  auto getter = [&inputs](size_t i, size_t part) {
    return DataRef{nullptr, inputs[i].at(part * 2).data(), inputs[i].at(part * 2 + 1).data()};
  };
  auto nPartsGetter = [&inputs](size_t i) {
    return inputs[i].size() / 2;
  };

  InputSpan span{getter, nPartsGetter, inputs.size()};
  BOOST_REQUIRE(span.size() == inputs.size());
  routeNo = 0;
  for (; routeNo < span.size(); ++routeNo) {
    auto ref = span.get(routeNo);
    BOOST_CHECK(inputs[routeNo].at(0) == ref.header);
    BOOST_CHECK(inputs[routeNo].at(1) == ref.payload);
    if (routeNo == 1) {
      BOOST_CHECK(span.getNofParts(routeNo) == 3);
      ref = span.get(routeNo, 1);
      BOOST_CHECK(inputs[routeNo].at(2) == ref.header);
      BOOST_CHECK(inputs[routeNo].at(3) == ref.payload);
    } else {
      BOOST_CHECK(span.getNofParts(routeNo) == 1);
    }
  }

  routeNo = 0;
  for (auto it = span.begin(), end = span.end(); it != end; ++it) {
    size_t partNo = 0;
    BOOST_CHECK(it.size() * 2 == inputs[routeNo].size());
    for (auto const& ref : it) {
      BOOST_CHECK(inputs[routeNo].at(partNo++) == ref.header);
      BOOST_CHECK(inputs[routeNo].at(partNo++) == ref.payload);
      std::cout << ref.header << " " << ref.payload << std::endl;
    }
    routeNo++;
  }
}
