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

#define BOOST_TEST_MODULE Test MCTruthContainer class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <unistd.h>

#include "buildMatBudLUT.C"

namespace o2
{
BOOST_AUTO_TEST_CASE(MatBudLUT)
{
#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

  // using process specific geometry names in order
  // to avoid race/conditions with other tests accessing geometry
  std::string geomPrefix("matBudGeom");
  std::string matBudFile("matbud");
  matBudFile += std::to_string(getpid()) + ".root";
  BOOST_CHECK(buildMatBudLUT(2, 20, matBudFile, geomPrefix + std::to_string(getpid()))); // generate LUT
  BOOST_CHECK(testMBLUT(matBudFile));                                                    // test LUT manipulations

#endif //!GPUCA_ALIGPUCODE
}
} // namespace o2
