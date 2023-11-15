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

#include "Framework/AlgorithmSpec.h"
#include "Framework/InputRecord.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DataAllocator.h"
#include <catch_amalgamated.hpp>
#include <vector>

using namespace o2::framework;

TEST_CASE("TestAlgorithmSpec")
{
  using namespace o2::framework;
  AlgorithmSpec::ProcessCallback foo = [](ProcessingContext&) {};
  AlgorithmSpec::InitCallback bar = [&foo](InitContext&) { return foo; };
  AlgorithmSpec spec1{bar};
  AlgorithmSpec spec2{bar, AlgorithmSpec::emptyErrorCallback()};
  AlgorithmSpec spec3{AlgorithmSpec::InitCallback{[&foo](InitContext&) {
    return foo;
  }}};
  AlgorithmSpec spec4{AlgorithmSpec::InitCallback{[&foo](InitContext&) {
    return [](ProcessingContext&) {};
  }}};
  AlgorithmSpec spec5{AlgorithmSpec::InitCallback{[&foo](InitContext&) {
                        return [](ProcessingContext&) {};
                      }},
                      AlgorithmSpec::emptyErrorCallback()};
  AlgorithmSpec spec6{{[&foo](InitContext&) {
                        return [](ProcessingContext&) {};
                      }},
                      AlgorithmSpec::emptyErrorCallback()};

  AlgorithmSpec spec7{adaptStateless([](InputRecord&, DataAllocator&) {})};
  int i = 0;
  AlgorithmSpec spec8{adaptStateless([&i](InputRecord&, DataAllocator&) {})};
}
