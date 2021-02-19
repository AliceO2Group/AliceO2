// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test analysis TimeRangeMasking class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

// boost includes
#include <boost/range/combine.hpp>
#include <boost/test/unit_test.hpp>

// o2 includes
#include "DataFormatsAnalysis/TimeRangeFlagsCollection.h"

namespace o2
{
namespace analysis
{

/// \brief Test reading ParameterGEM from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(test_TimeRangeMasking)
{
  TimeRangeFlagsCollection<uint64_t> masking;
}

} // namespace analysis
} // namespace o2

