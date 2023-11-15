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
#define BOOST_TEST_MODULE Test MCHWorkflow DigitsIO
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
#include "IO.h"

using namespace o2::mch;
using namespace o2::mch::io;

DigitFileFormat createFormat(uint8_t fileVersion,
                             uint8_t digitVersion,
                             uint8_t digitSize,
                             uint8_t rofVersion,
                             uint8_t rofSize,
                             bool run2ids,
                             bool hasRof)
{
  DigitFileFormat df;
  df.fileVersion = fileVersion;
  df.digitVersion = digitVersion;
  df.digitSize = digitSize;
  df.rofVersion = rofVersion;
  df.rofSize = rofSize;
  df.run2ids = run2ids;
  df.hasRof = hasRof;
  return df;
}

BOOST_AUTO_TEST_CASE(DigitFileFormatV0Value)
{
  DigitFileFormat v0 = createFormat(0, 0, 20, 0, 0, true, false);
  BOOST_CHECK_EQUAL(isValid(v0), true);
}

BOOST_AUTO_TEST_CASE(DigitFileFormatV1Value)
{
  DigitFileFormat v1 = createFormat(1, 0, 20, 1, 16, false, true);
  BOOST_CHECK_EQUAL(isValid(v1), true);
}

BOOST_AUTO_TEST_CASE(DigitFileFormatV2Value)
{
  DigitFileFormat v2 = createFormat(2, 0, 19, 1, 14, false, false);
  BOOST_CHECK_EQUAL(isValid(v2), true);
}

BOOST_AUTO_TEST_CASE(DigitFileFormatV3Value)
{
  DigitFileFormat v3 = createFormat(3, 0, 19, 1, 14, false, true);
  BOOST_CHECK_EQUAL(isValid(v3), true);
}

BOOST_AUTO_TEST_CASE(DigitFileFormatV4Value)
{
  DigitFileFormat v4 = createFormat(4, 0, 19, 2, 18, false, true);
  BOOST_CHECK_EQUAL(isValid(v4), true);
}

BOOST_DATA_TEST_CASE(WriteMustReturnFalseIfDigitVectorIsEmpty, digitFileFormats, digitFileFormat)
{
  std::ostringstream str;
  std::vector<ROFRecord> rofs;
  std::vector<Digit> digits;

  rofs.push_back({});

  DigitSink dwText(str);
  bool ok = dwText.write(digits, rofs);

  BOOST_CHECK_EQUAL(ok, false);

  DigitSink dwBinary(str, digitFileFormat);
  ok = dwBinary.write(digits, rofs);

  BOOST_CHECK_EQUAL(ok, false);
}

BOOST_DATA_TEST_CASE(BinaryWriteMustReturnFalseIfRofVectorIsEmpty, digitFileFormats, digitFileFormat)
{
  if (not digitFileFormat.hasRof) {
    return;
  }
  std::ostringstream str;
  std::vector<ROFRecord> rofs;
  std::vector<Digit> digits;

  digits.push_back({});

  DigitSink dwBinary(str, digitFileFormat);
  bool ok = dwBinary.write(digits, rofs);

  BOOST_CHECK_EQUAL(ok, false);
}

BOOST_AUTO_TEST_CASE(DefaultFormatIsTheTag)
{
  o2::mch::io::DigitFileFormat dff;
  BOOST_CHECK_EQUAL(dff.format, o2::mch::io::TAG_DIGITS);
}

BOOST_DATA_TEST_CASE(ReaderMustIdentifyFileFormat, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  buffer.write(reinterpret_cast<const char*>(&digitFileFormat), sizeof(uint64_t));
  buffer.clear();
  buffer.seekg(0);
  DigitSampler dr(buffer);
  BOOST_CHECK_EQUAL(dr.fileFormat(), digitFileFormat);
}

BOOST_DATA_TEST_CASE(BinaryWriterMustTagCorrectly, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  DigitSink dw(buffer, digitFileFormat);
  DigitSampler dr(buffer);
  BOOST_CHECK_EQUAL(dr.fileFormat(), digitFileFormat);
}

std::vector<o2::mch::Digit> createDummyFixedDigits(int n)
{
  assert(n < 100);
  std::vector<Digit> digits;
  int dummyADC{40};
  int32_t dummyTime{1000};
  uint16_t dummySamples{10};
  for (int i = 0; i < n; i++) {
    auto& d = digits.emplace_back(100, i, dummyADC + i, dummyTime + i * 100, dummySamples + i * 10);
    if (i == 7 || i == 23) {
      d.setSaturated(true);
    }
  }
  return digits;
}

struct TF {
  std::vector<Digit> digits;
  std::vector<ROFRecord> rofs;
};

TF createDummyData(int ndigits, int nrofs, uint32_t firstOrbit)
{
  if (nrofs >= ndigits) {
    throw std::invalid_argument("cannot have more rofs than digits!");
  }
  TF tf;
  tf.digits = createDummyFixedDigits(ndigits);
  int step = ndigits / nrofs;
  for (int i = 0; i < nrofs; i++) {
    o2::InteractionRecord ir(0, firstOrbit + i);
    tf.rofs.emplace_back(ir, step * i, i == nrofs - 1 ? ndigits - step * i : step);
  };
  return tf;
}

/** The test data is used to check the internal consistency
 * of the reader and writer (i.e. data written by the writer must be readable
 * by the reader).
 * A complete test requires, in addition, the usage of externally created data
 * (see for instance V0File in testDigitIOV0)
 */
constexpr int NROF_1 = 1;
constexpr int NROF_2 = 1;
constexpr int NROF_3 = 1;
constexpr int NROF_4 = 3;
constexpr int NROFS = NROF_1 + NROF_2 + NROF_3 + NROF_4;

std::vector<TF> testData{
  createDummyData(3, NROF_1, 0),
  createDummyData(5, NROF_2, 10),
  createDummyData(13, NROF_3, 20),
  createDummyData(26, NROF_4, 20),
};

void writeTestData(std::ostream& out, DigitFileFormat dff)
{
  auto tfs = testData;
  DigitSink dw(out, dff);
  for (auto tf : tfs) {
    dw.write(tf.digits, tf.rofs);
  }
}

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_DATA_TEST_CASE(TestDataDump, digitFileFormats, digitFileFormat)
{
  auto tfs = testData;
  DigitSink dw(std::cout);
  for (auto tf : tfs) {
    dw.write(tf.digits, tf.rofs);
  }
  BOOST_CHECK_EQUAL(tfs.size(), testData.size());
}

BOOST_DATA_TEST_CASE(TestDataIsOfExpectedFormat, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  writeTestData(buffer, digitFileFormat);
  DigitSampler dr(buffer);

  BOOST_CHECK_EQUAL(dr.fileFormat(), digitFileFormat);
}

BOOST_DATA_TEST_CASE(TestDataHasExpectedNofROFs, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  writeTestData(buffer, digitFileFormat);
  DigitSampler dr(buffer);

  BOOST_CHECK_EQUAL(dr.nofROFs(), NROFS);
}

BOOST_DATA_TEST_CASE(TestDataHasExpectedNofTFs, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  writeTestData(buffer, digitFileFormat);
  DigitSampler dr(buffer);

  if (not digitFileFormat.hasRof) {
    BOOST_CHECK_EQUAL(dr.nofTimeFrames(), NROFS);
  } else {
    BOOST_CHECK_EQUAL(dr.nofTimeFrames(), testData.size());
  }
}

BOOST_DATA_TEST_CASE(TestDataHasExpectedNumberOfDigitsWhenReading, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  writeTestData(buffer, digitFileFormat);
  DigitSampler dr(buffer);

  std::vector<Digit> digits;
  std::vector<ROFRecord> rofs;
  int ndigits{0};
  while (dr.read(digits, rofs)) {
    ndigits += digits.size();
  }
  BOOST_CHECK_EQUAL(ndigits, 47);
}

BOOST_DATA_TEST_CASE(TestDataHasExpectedNumberOfDigitsWhenCounting, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  writeTestData(buffer, digitFileFormat);
  DigitSampler dr(buffer);

  BOOST_CHECK_EQUAL(dr.nofDigits(), 47);
}

BOOST_DATA_TEST_CASE(CheckReader, digitFileFormats, digitFileFormat)
{
  std::stringstream buffer;
  writeTestData(buffer, digitFileFormat);
  DigitSampler dr(buffer);

  std::vector<Digit> digits;
  std::vector<ROFRecord> rofs;

  std::array<int, 3> digits_per_tf = {3, 5, 13};
  for (int itf = 0; itf < 3; ++itf) {
    dr.read(digits, rofs);
    BOOST_CHECK_EQUAL(digits.size(), digits_per_tf[itf]);
    BOOST_CHECK_EQUAL(rofs.size(), 1);
    BOOST_CHECK_EQUAL(rofs[0].getFirstIdx(), 0);
    BOOST_CHECK_EQUAL(rofs[0].getLastIdx(), digits_per_tf[itf] - 1);
  }
  dr.read(digits, rofs);
  int ndigits_4th_tf = not digitFileFormat.hasRof ? 8 : 26;
  BOOST_CHECK_EQUAL(digits.size(), ndigits_4th_tf);
  if (not digitFileFormat.hasRof) {
    BOOST_CHECK_EQUAL(rofs.size(), 1);
    BOOST_CHECK_EQUAL(rofs[0].getFirstIdx(), 0);
    BOOST_CHECK_EQUAL(rofs[0].getLastIdx(), 7);
    BOOST_CHECK_EQUAL(digits[7].getADC(), 47);
    BOOST_CHECK_EQUAL(digits[7].isSaturated(), true);
  } else {
    BOOST_CHECK_EQUAL(rofs.size(), 3);
    BOOST_CHECK_EQUAL(rofs[0].getFirstIdx(), 0);
    BOOST_CHECK_EQUAL(rofs[0].getLastIdx(), 7);
    BOOST_CHECK_EQUAL(rofs[1].getFirstIdx(), 8);
    BOOST_CHECK_EQUAL(rofs[1].getLastIdx(), 15);
    BOOST_CHECK_EQUAL(rofs[2].getFirstIdx(), 16);
    BOOST_CHECK_EQUAL(rofs[2].getLastIdx(), 25);
    BOOST_CHECK_EQUAL(digits[23].getADC(), 63);
    BOOST_CHECK_EQUAL(digits[23].isSaturated(), true);
  }
}
