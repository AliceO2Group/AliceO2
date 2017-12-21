// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   headerstack.cxx
/// @author Matthias Richter
/// @since  2017-09-21
/// @brief  Unit test for table view abstraction class

#define BOOST_TEST_MODULE Test Algorithm TableView
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <cstring> // memcmp
#include "Headers/DataHeader.h" // hexdump, DataHeader
#include "Headers/HeartbeatFrame.h" // HeartbeatHeader, HeartbeatTrailer
#include "../include/Algorithm/TableView.h"
#include "../include/Algorithm/Parser.h"
#include "StaticSequenceAllocator.h"

using DataHeader = o2::header::DataHeader;
using HeartbeatHeader = o2::header::HeartbeatHeader;
using HeartbeatTrailer = o2::header::HeartbeatTrailer;

template<typename... Targs>
void hexDump(Targs... Fargs) {
  // a simple redirect to enable/disable the hexdump printout
  o2::header::hexDump(Fargs...);
}

BOOST_AUTO_TEST_CASE(test_tableview_reverse)
{
  using FrameT = o2::algorithm::Composite<HeartbeatHeader, HeartbeatTrailer>;
  using TestFrame = o2::algorithm::StaticSequenceAllocator;
  // the length of the data is set in the trailer word
  // the header is used as column description, using slightly different
  // orbit numbers in the to data sets which will result in two complete
  // columns at beginning and end, while the two in the middle only have
  // one row entry
  TestFrame tf1(FrameT({0x1100000000000000}, "heartbeatdata", {0x510000000000000e}),
                FrameT({0x1100000000000001}, "test", {0x5100000000000005}),
                FrameT({0x1100000000000003}, "dummydata", {0x510000000000000a})
                );
  TestFrame tf2(FrameT({0x1100000000000000}, "frame2a", {0x5100000000000008}),
                FrameT({0x1100000000000002}, "frame2b", {0x5100000000000008}),
                FrameT({0x1100000000000003}, "frame2c", {0x5100000000000008})
                );
  hexDump("Test frame 1", tf1.buffer.get(), tf1.size());
  hexDump("Test frame 2", tf2.buffer.get(), tf2.size());

  // the payload length is set in the trailer, so we need a reverse parser
  using ParserT = o2::algorithm::ReverseParser<typename FrameT::HeaderType,
                                               typename FrameT::TrailerType>;

  // define the view type for DataHeader as row descriptor,
  // HeartbeatHeader as column descriptor and the reverse parser
  using ViewType = o2::algorithm::TableView<o2::header::DataHeader,
                                            o2::header::HeartbeatHeader,
                                            ParserT>;
  ViewType heartbeatview;

  o2::header::DataHeader dh1;
  dh1.dataDescription = o2::header::DataDescription("FIRSTROW");
  dh1.dataOrigin = o2::header::DataOrigin("TST");
  dh1.subSpecification = 0;
  dh1.payloadSize = 0;

  o2::header::DataHeader dh2;
  dh2.dataDescription = o2::header::DataDescription("SECONDROW");
  dh2.dataOrigin = o2::header::DataOrigin("TST");
  dh2.subSpecification = 0xdeadbeef;
  dh2.payloadSize = 0;

  heartbeatview.addRow(dh1, tf1.buffer.get(), tf1.size());
  heartbeatview.addRow(dh2, tf2.buffer.get(), tf2.size());

  std::cout << "slots: " << heartbeatview.getNRows()
            << " columns: " << heartbeatview.getNColumns()
            << std::endl;

  // definitions for the data check
  const char* dataset1[] = {
    "heartbeatdata",
    "test",
    "dummydata"
  };
  const char* dataset2[] = {
    "frame2a",
    "frame2b",
    "frame2c"
  };

  // four orbits are populated, 0 and 3 with 2 rows, 1 and 2 with one row
  BOOST_REQUIRE(heartbeatview.getNColumns() == 4);
  BOOST_REQUIRE(heartbeatview.getNRows() == 2);
  unsigned requiredNofRowsInColumn[] = {2, 1, 1, 2};

  unsigned colidx = 0;
  unsigned dataset1idx = 0;
  unsigned dataset2idx = 0;
  for (auto columnIt = heartbeatview.begin(), end = heartbeatview.end();
       columnIt != end; ++columnIt, ++colidx) {
    unsigned rowidx = 0;
    std::cout << "---------------------------------------" << std::endl;
    for (auto row : columnIt) {
      auto dataset = (rowidx == 1 || colidx == 2)? dataset2 : dataset1;
      auto & datasetidx = (rowidx == 1 || colidx == 2)? dataset2idx : dataset1idx;
      hexDump("Entry", row.buffer, row.size);
      BOOST_CHECK(memcmp(row.buffer, dataset[datasetidx++], row.size) == 0);
      ++rowidx;
    }
    BOOST_CHECK(rowidx == requiredNofRowsInColumn[colidx]);
  }
}
