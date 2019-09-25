// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test EMCAL Reconstruction
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <DataFormatsEMCAL/Constants.h>
#include "EMCALReconstruction/Mapper.h"
#include <array>
#include <iostream>
#include <fstream>
#include <vector>
#include "RStringView.h"

std::vector<std::array<int, 4>> loadReferenceMapping(const std::string_view filename);

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
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string mappingfile = inputDir + "RCU0A.data";
  o2::emcal::Mapper testmapper(mappingfile);

  // Load reference mapping (same file)
  auto refmapping = loadReferenceMapping(mappingfile);

  // test mapping of channel
  for (const auto& chan : refmapping) {
    BOOST_CHECK_EQUAL(chan[1], testmapper.getRow(chan[0]));
    BOOST_CHECK_EQUAL(chan[2], testmapper.getColumn(chan[0]));
    BOOST_CHECK_EQUAL(o2::emcal::intToChannelType(chan[3]), testmapper.getChannelType(chan[0]));
    BOOST_CHECK_EQUAL(chan[0], testmapper.getHardwareAddress(chan[1], chan[2], o2::emcal::intToChannelType(chan[3]))); // test of inverse mapping
  }

  // test of the error handling:
  // Hardware address outside range
  BOOST_CHECK_EXCEPTION(testmapper.getRow(4000), o2::emcal::Mapper::AddressNotFoundException, [](o2::emcal::Mapper::AddressNotFoundException const& e) { return e.getAddress() == 4000; });
  // Row, and column out of range
  BOOST_CHECK_EXCEPTION(testmapper.getHardwareAddress(16, 0, o2::emcal::ChannelType_t::HIGH_GAIN), o2::emcal::Mapper::ChannelNotFoundException, [](o2::emcal::Mapper::ChannelNotFoundException const& e) { return e.getChannel().mRow == 16; });
  BOOST_CHECK_EXCEPTION(testmapper.getHardwareAddress(0, 128, o2::emcal::ChannelType_t::TRU), o2::emcal::Mapper::ChannelNotFoundException, [](o2::emcal::Mapper::ChannelNotFoundException const& e) { return e.getChannel().mColumn == 128; });
}

/// \brief Load reference mapping from mapping file
/// \param mappingfile Full path to the file with the mapping
/// \return Vector with channel information (as integers)
std::vector<std::array<int, 4>> loadReferenceMapping(const std::string_view mappingfile)
{
  std::ifstream in(mappingfile.data());
  std::string tmpstr;
  std::getline(in, tmpstr);
  std::vector<std::array<int, 4>> mapping;
  std::getline(in, tmpstr);
  int address, row, col, caloflag;
  while (std::getline(in, tmpstr)) {
    std::stringstream addressdecoder(tmpstr);
    addressdecoder >> address >> row >> col >> caloflag;
    mapping.push_back({{address, row, col, caloflag}});
  }

  return mapping;
}