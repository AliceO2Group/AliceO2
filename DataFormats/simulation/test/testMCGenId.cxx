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

#define BOOST_TEST_MODULE Test MCGenId class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/MCGenProperties.h"

using namespace o2::mcgenid;

BOOST_AUTO_TEST_CASE(MCGenId_test)
{
  // possible generator IDs range from 0 to 127 (included)
  constexpr int highGenerator{128};
  // possible sub-generator IDs range from -1 to 30 (included)
  constexpr int highSubGenerator{31};
  // possible soufce IDs range from 0 to 15 (included)
  constexpr int highSource{16};

  // test all combinations
  for (int sourceId = 0; sourceId < highSource; sourceId++) {
    for (int generatorId = 0; generatorId < highGenerator; generatorId++) {
      for (int subGeneratorId = -1; subGeneratorId < highSubGenerator; subGeneratorId++) {
        auto encoded = getEncodedGenId(generatorId, sourceId, subGeneratorId);
        // decode them
        auto sourceIdAfter = getSourceId(encoded);
        auto generatorIdAfter = getGeneratorId(encoded);
        auto subGeneratorIdAfter = getSubGeneratorId(encoded);

        std::cout << "SourceID: " << sourceId << " ==> " << sourceIdAfter << "\n"
                  << "generatorId: " << generatorId << " ==> " << generatorIdAfter << "\n"
                  << "subGeneratorId: " << subGeneratorId << " ==> " << subGeneratorIdAfter << "\n";

        // check if original and decoded numbers are the same
        BOOST_CHECK(sourceIdAfter == sourceId);
        BOOST_CHECK(generatorIdAfter == generatorId);
        BOOST_CHECK(subGeneratorId == subGeneratorIdAfter);
      }
    }
  }
}
