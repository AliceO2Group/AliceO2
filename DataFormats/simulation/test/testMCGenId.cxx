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

#define BOOST_TEST_MODULE Test MCGenStatus class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/MCGenProperties.h"
#include "TRandom.h"

using namespace o2::mcgenid;

BOOST_AUTO_TEST_CASE(MCGenId_test)
{
  // create 2 vectors each with some random integers
  constexpr size_t length{100};
  constexpr int low{-2};
  constexpr int high{16};

  // initialise the seed (could be anything)
  gRandom->SetSeed();

  for (size_t i = 0; i < length; i++) {
    // draw random integers
    auto sourceId = static_cast<int>(gRandom->Uniform(low, high));
    auto generatorId = static_cast<int>(gRandom->Uniform(low, high));
    auto cocktailId = static_cast<int>(gRandom->Uniform(low, high));

    // encode them
    auto encoded = getEncodedGenId(generatorId, sourceId, cocktailId);

    // decode them
    auto sourceIdAfter = getSourceId(encoded);
    auto generatorIdAfter = getGeneratorId(encoded);
    auto cocktailIdAfter = getCocktailId(encoded);

    std::cout << "SourceID: " << sourceId << " ==> " << sourceIdAfter << "\n"
              << "generatorId: " << generatorId << " ==> " << generatorIdAfter << "\n"
              << "cocktailId: " << cocktailId << " ==> " << cocktailIdAfter << "\n";

    // check if original and decoded numbers are the same
    BOOST_CHECK(sourceIdAfter == sourceId);
    BOOST_CHECK(generatorIdAfter == generatorId);
    BOOST_CHECK(cocktailId == cocktailIdAfter);
  }
}
