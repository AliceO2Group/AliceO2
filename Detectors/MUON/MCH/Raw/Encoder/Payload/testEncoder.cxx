// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche
///
/// In those tests we are mainly concerned about testinng
/// whether the payloads are actually properly simulated.
///
#define BOOST_TEST_MODULE Test MCHRaw Encoder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHRawEncoder/Encoder.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawEncoder/DataBlock.h"
#include <fmt/printf.h>
#include "DumpBuffer.h"
#include <boost/mpl/list.hpp>
#include "CruBufferCreator.h"

using namespace o2::mch::raw;

template <typename FORMAT>
std::unique_ptr<Encoder> defaultEncoder()
{
  return createEncoder<FORMAT, SampleMode, true>();
}

typedef boost::mpl::list<BareFormat, UserLogicFormat> testTypes;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(encoder)

BOOST_AUTO_TEST_CASE_TEMPLATE(StartHBFrameBunchCrossingMustBe12Bits, T, testTypes)
{
  auto encoder = defaultEncoder<T>();
  BOOST_CHECK_THROW(encoder->startHeartbeatFrame(0, 1 << 12), std::invalid_argument);
  BOOST_CHECK_NO_THROW(encoder->startHeartbeatFrame(0, 0xFFF));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(EmptyEncoderHasEmptyBufferIfPhaseIsZero, T, testTypes)
{
  srand(time(nullptr));
  auto encoder = defaultEncoder<T>();
  encoder->startHeartbeatFrame(12345, 123);
  std::vector<uint8_t> buffer;
  encoder->moveToBuffer(buffer);
  BOOST_CHECK_EQUAL(buffer.size(), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(EmptyEncodeIsNotNecessarilyEmptyDependingOnPhase, T, testTypes)
{
  srand(time(nullptr));
  auto encoder = createEncoder<T, SampleMode, false>();
  encoder->startHeartbeatFrame(12345, 123);
  std::vector<uint8_t> buffer;
  encoder->moveToBuffer(buffer);
  BOOST_CHECK_GE(buffer.size(), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MultipleOrbitsWithNoDataIsAnEmptyBufferIfPhaseIsZero, T, testTypes)
{
  srand(time(nullptr));
  auto encoder = defaultEncoder<T>();
  encoder->startHeartbeatFrame(12345, 123);
  encoder->startHeartbeatFrame(12345, 125);
  encoder->startHeartbeatFrame(12345, 312);
  std::vector<uint8_t> buffer;
  encoder->moveToBuffer(buffer);
  BOOST_CHECK_EQUAL(buffer.size(), 0);
}

int estimateUserLogicSize(int nofDS, int maxNofChPerDS)
{
  size_t headerSize = 2; // equivalent to 2 64-bits words
  // one 64-bits header and one 64-bits data per channel
  // plus one sync per DS
  // (assuming data = just one sample)
  size_t ndata = (maxNofChPerDS * 2) + nofDS;
  return 8 * (ndata + headerSize); // size in bytes
}

int estimateBareSize(int nofDS, int maxNofChPerGBT)
{
  size_t headerSize = 2; // equivalent to 2 64-bits words
  size_t nbits = nofDS * 50 + maxNofChPerGBT * 90;
  size_t n128bitwords = nbits / 2;
  size_t n64bitwords = n128bitwords * 2;
  return 8 * (n64bitwords + headerSize); // size in bytes
}

template <typename FORMAT>
int estimateSize();

template <>
int estimateSize<BareFormat>()
{

  return estimateBareSize(1, 3) +
         estimateBareSize(1, 4) +
         estimateBareSize(1, 6);
}

template <>
int estimateSize<UserLogicFormat>()
{
  return estimateUserLogicSize(1, 3) +
         estimateUserLogicSize(1, 4) +
         estimateUserLogicSize(1, 6);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckNumberOfPayloadHeaders, T, testTypes)
{
  auto buffer = test::CruBufferCreator<T, ChargeSumMode>::makeBuffer();
  int nheaders = o2::mch::raw::countHeaders(buffer);
  BOOST_CHECK_EQUAL(nheaders, 3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckSize, T, testTypes)
{
  auto buffer = test::CruBufferCreator<T, ChargeSumMode>::makeBuffer();
  size_t expectedSize = estimateSize<T>();
  BOOST_CHECK_EQUAL(buffer.size(), expectedSize);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
