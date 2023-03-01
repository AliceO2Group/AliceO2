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

#include "Framework/InputSpan.h"
#include "Framework/DataRef.h"
#include <vector>
#include <string>
#include <catch_amalgamated.hpp>

using namespace o2::framework;

TEST_CASE("TestInputSpan")
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
  REQUIRE(span.size() == inputs.size());
  routeNo = 0;
  for (; routeNo < span.size(); ++routeNo) {
    auto ref = span.get(routeNo);
    REQUIRE(inputs[routeNo].at(0) == ref.header);
    REQUIRE(inputs[routeNo].at(1) == ref.payload);
    if (routeNo == 1) {
      REQUIRE(span.getNofParts(routeNo) == 3);
      ref = span.get(routeNo, 1);
      REQUIRE(inputs[routeNo].at(2) == ref.header);
      REQUIRE(inputs[routeNo].at(3) == ref.payload);
    } else {
      REQUIRE(span.getNofParts(routeNo) == 1);
    }
  }

  routeNo = 0;
  for (auto it = span.begin(), end = span.end(); it != end; ++it) {
    size_t partNo = 0;
    REQUIRE(it.size() * 2 == inputs[routeNo].size());
    for (auto const& ref : it) {
      REQUIRE(inputs[routeNo].at(partNo++) == ref.header);
      REQUIRE(inputs[routeNo].at(partNo++) == ref.payload);
      INFO(ref.header << " " << ref.payload);
    }
    routeNo++;
  }
}
