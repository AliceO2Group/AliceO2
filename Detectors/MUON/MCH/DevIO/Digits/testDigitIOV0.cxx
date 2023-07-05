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

#include <boost/test/tools/old/interface.hpp>
#include <sstream>
#include "DigitFileFormat.h"
#define BOOST_TEST_MODULE Test MCHWorkflow DigitsIO - V0
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include "DigitIO.h"
#include <algorithm>
#include <fmt/format.h>
#include <fstream>
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <cassert>
#include <array>
#include <stdexcept>
#include "DigitSamplerImpl.h"
#include "TestFileV0.h"
#include "DigitD0.h"
#include "IO.h"

using namespace o2::mch;
using namespace o2::mch::io;

void bin2cpp(const char* filename = "digits.v3.in")
{
  // filename must point to a V0 format
  std::ifstream in(filename);
  int ndig;
  int pos{8};
  int event{0};
  in.seekg(8);
  while ((ndig = o2::mch::io::impl::advance(in, digitFileFormats[0].digitSize, "digits")) >= 0) {
    int next = in.tellg();
    if (ndig < 40 || ndig == 96) {
      std::cout << fmt::format("Event {:4d} {} ndigits between {} and {}\n", event, ndig, pos, next);
    }
    pos = next;
    event++;
  }
  struct a {
    int start, end;
  };
  std::array<a, 2> positions = {a{17707576, 17708160}, a{38707380, 38708064}};
  std::vector<uint8_t> bytes;
  int i{1};
  for (auto p : positions) {
    in.seekg(p.start);
    int n = p.end - p.start;
    bytes.resize(n);
    in.read(reinterpret_cast<char*>(&bytes[0]), n);
    if (in.tellg() != p.end) {
      std::cout << "lost in file!\n";
      exit(1);
    }
    for (auto b : bytes) {
      std::cout << fmt::format("0x{:02X},", b);
      if (i % 16 == 0) {
        std::cout << "\n";
      }
      i++;
    }
  }
  std::cout << "i=" << i << "\n";
}

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE(Bin2Cpp)
{
  bin2cpp();
}
struct V0File {
  V0File()
  {
    std::ostringstream buffer;
    for (auto b : v0_buffer) {
      buffer << b;
    }
    std::istringstream in(buffer.str());
    DigitSampler r(in);
    r.read(digits[0], rofs[0]);
    r.read(digits[1], rofs[1]);
  }
  std::array<std::vector<o2::mch::Digit>, 2> digits;
  std::array<std::vector<o2::mch::ROFRecord>, 2> rofs;
};

BOOST_AUTO_TEST_CASE(CheckStructOffsets)
{
  BOOST_CHECK_EQUAL(offsetof(o2::mch::io::impl::DigitD0, tfTime), 0);
  BOOST_CHECK_EQUAL(offsetof(o2::mch::io::impl::DigitD0, nofSamples), 4);
  BOOST_CHECK_EQUAL(offsetof(o2::mch::io::impl::DigitD0, detID), 8);
  BOOST_CHECK_EQUAL(offsetof(o2::mch::io::impl::DigitD0, padID), 12);
  BOOST_CHECK_EQUAL(offsetof(o2::mch::io::impl::DigitD0, adc), 16);
  BOOST_CHECK_EQUAL(sizeof(o2::mch::io::impl::DigitD0), 20);
}

BOOST_FIXTURE_TEST_SUITE(SpotCheckV0, V0File)

BOOST_AUTO_TEST_CASE(SpotCheckSizesV0)
{
  BOOST_CHECK_EQUAL(digits[0].size(), 29);
  BOOST_CHECK_EQUAL(digits[1].size(), 34);

  BOOST_CHECK_EQUAL(rofs[0].size(), 1);
  BOOST_CHECK_EQUAL(rofs[1].size(), 1);

  BOOST_CHECK_EQUAL(digits[0][20].getDetID(), 712);
  BOOST_CHECK_EQUAL(digits[1][31].getDetID(), 1024);
}

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE(SpotCheckDumpV0)
{
  DigitSink w(std::cout);
  w.write(digits[0], rofs[0]);
  w.write(digits[1], rofs[1]);
}

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE(SpotCheckDetIDV0)
{
  BOOST_CHECK_EQUAL(digits[0][20].getDetID(), 712);
  BOOST_CHECK_EQUAL(digits[1][31].getDetID(), 1024);
}

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE(SpotCheckPadIDV0)
{
  BOOST_CHECK_EQUAL(digits[0][20].getPadID(), 1661387464);
  BOOST_CHECK_EQUAL(digits[1][31].getPadID(), 268444672);
}

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE(SpotCheckADCV0)
{
  BOOST_CHECK_EQUAL(digits[0][20].getADC(), 1063648290);
  BOOST_CHECK_EQUAL(digits[1][31].getADC(), 1065400059);
}

BOOST_AUTO_TEST_SUITE_END()
