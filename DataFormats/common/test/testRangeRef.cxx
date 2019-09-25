// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TimeRangeRef class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonDataFormat/RangeReference.h"
#include <FairLogger.h>

namespace o2
{

// basic TimeStamp tests
BOOST_AUTO_TEST_CASE(RangeRef)
{
  int ent = 1000, nent = 5;
  o2::dataformats::RangeReference<int, int> rangeII(ent, nent);
  BOOST_CHECK_EQUAL(rangeII.getFirstEntry(), ent);
  rangeII.changeEntriesBy(nent);
  BOOST_CHECK_EQUAL(rangeII.getEntries(), 2 * nent);

  o2::dataformats::RangeRefComp<4> range4(ent, nent);
  BOOST_CHECK_EQUAL(range4.getFirstEntry(), ent);
  range4.changeEntriesBy(nent);
  BOOST_CHECK_EQUAL(range4.getEntries(), 2 * nent);
  LOG(INFO) << "MaxEntryID : " << range4.getMaxFirstEntry() << " MaxEntries: " << range4.getMaxEntries();
  BOOST_CHECK_EQUAL(range4.getMaxFirstEntry(), (0x1 << (32 - 4)) - 1);
  BOOST_CHECK_EQUAL(range4.getMaxEntries(), (0x1 << 4) - 1);
}

} // namespace o2
