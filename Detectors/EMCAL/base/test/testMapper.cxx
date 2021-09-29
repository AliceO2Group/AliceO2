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
#define BOOST_TEST_MODULE Test EMCAL Base
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <fmt/format.h>
#include <DataFormatsEMCAL/Constants.h>
#include "EMCALBase/Mapper.h"
#include <array>
#include <iostream>
#include <fstream>
#include <vector>
#include "RStringView.h"

struct refchannel {
  int mAddress;
  int mRow;
  int mCol;
  int mCellType;
  int mAmbiguous;
};

std::vector<refchannel> loadReferenceMapping(const std::string_view filename);

/// \macro Test implementation of the EMCAL mapper
///
/// Test coverage:
/// - Row, column and channel type from hardware address: all channels
/// - Hardware address from row, column and channel type (inverse mapping): all channels
/// - Invalid hardware address: exception test
/// - Invalid channel row / column: exception test
BOOST_AUTO_TEST_CASE(Mapper_test)
{

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env) {
    inputDir = aliceO2env;
  }
  inputDir += "/share/Detectors/EMC/files/";

  std::vector<char> sides = {'A', 'C'};
  for (auto side : sides) {
    for (int iddl = 0; iddl < 2; iddl++) {
      std::string mappingbase = fmt::format("RCU{}{}.data", iddl, side);
      std::cout << "Test mapping " << mappingbase << std::endl;
      std::string mappingfile = inputDir + mappingbase;
      o2::emcal::Mapper testmapper(mappingfile);

      // Load reference mapping (same file)
      auto refmapping = loadReferenceMapping(mappingfile);

      // test mapping of channel
      for (const auto& chan : refmapping) {
        BOOST_CHECK_EQUAL(chan.mRow, testmapper.getRow(chan.mAddress));
        BOOST_CHECK_EQUAL(chan.mCol, testmapper.getColumn(chan.mAddress));
        BOOST_CHECK_EQUAL(o2::emcal::intToChannelType(chan.mCellType), testmapper.getChannelType(chan.mAddress));
        if (!chan.mAmbiguous) {
          // Skip channels in inverse mapping for which two hardware adresses are registered
          // (feature of the odd DDL mappings)
          BOOST_CHECK_EQUAL(chan.mAddress, testmapper.getHardwareAddress(chan.mRow, chan.mCol, o2::emcal::intToChannelType(chan.mCellType))); // test of inverse mapping
        }
      }
      if (mappingbase == "RCU0A.data") {
        // test of the error handling:
        // Hardware address outside range
        BOOST_CHECK_EXCEPTION(testmapper.getRow(4000), o2::emcal::Mapper::AddressNotFoundException, [](o2::emcal::Mapper::AddressNotFoundException const& e) { return e.getAddress() == 4000; });
        // Row, and column out of range
        BOOST_CHECK_EXCEPTION(testmapper.getHardwareAddress(16, 0, o2::emcal::ChannelType_t::HIGH_GAIN), o2::emcal::Mapper::ChannelNotFoundException, [](o2::emcal::Mapper::ChannelNotFoundException const& e) { return e.getChannel().mRow == 16; });
        BOOST_CHECK_EXCEPTION(testmapper.getHardwareAddress(0, 128, o2::emcal::ChannelType_t::TRU), o2::emcal::Mapper::ChannelNotFoundException, [](o2::emcal::Mapper::ChannelNotFoundException const& e) { return e.getChannel().mColumn == 128; });
      }
    }
  }
}

/// \brief Load reference mapping from mapping file
/// \param mappingfile Full path to the file with the mapping
/// \return Vector with channel information (as integers)
std::vector<refchannel> loadReferenceMapping(const std::string_view mappingfile)
{
  std::vector<refchannel> mapping;
  std::ifstream in(mappingfile.data());
  std::string tmpstr;
  // skip first two lines (header)
  std::getline(in, tmpstr);
  std::getline(in, tmpstr);
  int address, row, col, caloflag;
  int nline = 0;
  while (std::getline(in, tmpstr)) {
    std::stringstream addressdecoder(tmpstr);
    addressdecoder >> address >> row >> col >> caloflag;
    // check whether the col/row is already registered with a different hardware address
    // Odd-DDLs have several TRU channels listed with different hardware addresses
    // In such cases the inverse mapping cannot be tested reliably due to ambiguity and
    // must be skipped.
    auto channelPresent = std::find_if(mapping.begin(), mapping.end(), [row, col, caloflag](const refchannel& test) {
      return row == test.mRow && col == test.mCol && caloflag == test.mCellType;
    });
    bool ambiguous = false;
    if (channelPresent != mapping.end()) {
      ambiguous = true;
      channelPresent->mAmbiguous = ambiguous;
    }
    mapping.push_back({address, row, col, caloflag, ambiguous});
  }

  return mapping;
}