// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TimeStamp class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonDataFormat/TimeStamp.h"

namespace o2
{

// basic TimeStamp tests
BOOST_AUTO_TEST_CASE(TimeStamp)
{
  o2::dataformats::TimeStampWithError<float, float> aTimeStamp(10., 0.1);
  BOOST_CHECK_CLOSE(aTimeStamp.getTimeStampError(), 0.1, 1E-4);
  BOOST_CHECK_CLOSE(aTimeStamp.getTimeStamp(), 10, 1E-4);
}

} // namespace o2
