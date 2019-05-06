// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCTruthContainer class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "buildMatBudLUT.C"

namespace o2
{
BOOST_AUTO_TEST_CASE(MatBudLUT)
{
#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

  BOOST_CHECK(buildMatBudLUT(2, 20)); // generate LUT
  BOOST_CHECK(testMBLUT());           // test LUT manipulations

#endif //!GPUCA_ALIGPUCODE
}
} // namespace o2
