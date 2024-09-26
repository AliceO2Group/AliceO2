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
#define BOOST_TEST_MODULE Test EMCAL Reconstruction
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <vector>
#include <tuple>
#include <TRandom.h>
#include <EMCALReconstruction/FastORTimeSeries.h>
#include "EMCALReconstruction/TRUDecodingErrors.h"

namespace o2
{

namespace emcal
{

void printBunch(const gsl::span<const uint16_t> adcs)
{
  bool first = true;
  for (auto& adc : adcs) {
    if (!first) {
      std::cout << ", ";
    } else {
      first = false;
    }
    std::cout << adc;
  }
  std::cout << " (size " << adcs.size() << ")" << std::endl;
}

std::vector<uint16_t> getReversed(const std::vector<uint16_t>& original)
{
  std::vector<uint16_t> reversed(13);
  for (std::size_t sample = 0; sample < 13; sample++) {
    reversed[12 - sample] = original[sample];
  }
  return reversed;
}

std::tuple<uint8_t, std::vector<uint16_t>, std::vector<uint16_t>> generatePulseTimeReversed()
{
  std::vector<uint16_t> pulse(13);
  std::fill(pulse.begin(), pulse.end(), 0);
  // calculate forward pulse
  auto peak_signal = static_cast<uint16_t>(gRandom->Uniform(0, 1024));
  pulse[4] = peak_signal;
  auto last = peak_signal;
  for (std::size_t sample = 5; sample < 13; sample++) {
    if (last == 0) {
      break;
    }
    auto current = static_cast<uint16_t>(gRandom->Uniform(0, last));
    pulse[sample] = current;
    last = current;
  }
  last = peak_signal;
  for (std::size_t sample = 3; sample > 0; sample--) {
    if (last == 0) {
      break;
    }
    auto current = static_cast<uint16_t>(gRandom->Uniform(0, last));
    pulse[sample] = current;
    last = current;
  }
  // find start time
  uint8_t starttime = 12;
  for (std::size_t currenttime = 12; currenttime > 0; currenttime--) {
    starttime = currenttime;
    if (pulse[currenttime]) {
      break;
    }
  }
  // time-reverse pulse
  auto reversed = getReversed(pulse);
  // zero-suppress time series
  std::vector<uint16_t> zerosuppressed;
  bool bunchstart = false;
  for (std::size_t sample = 0; sample < 13; sample++) {
    if (reversed[sample] == 0) {
      if (!bunchstart) {
        continue;
      }
      break;
    }
    bunchstart = true;
    zerosuppressed.push_back(reversed[sample]);
  }
  return std::make_tuple(starttime, zerosuppressed, pulse);
}

uint16_t calculateTimesum(const std::vector<uint16_t> samplesOrdered, uint8_t l0time)
{
  uint16_t timesum = 0;
  uint8_t starttime = l0time - 4;
  for (uint8_t sample = starttime; sample < starttime + 4; sample++) {
    timesum += samplesOrdered[sample];
  }
  return timesum;
}

std::vector<uint16_t> generateSmallBunch(uint8_t bunchlength)
{
  std::vector<uint16_t> bunch(bunchlength);
  auto peak_signal = static_cast<uint16_t>(gRandom->Uniform(0, 1024));
  bunch[bunchlength - 2] = peak_signal;
  bunch[bunchlength - 1] = static_cast<uint16_t>(gRandom->Uniform(0, peak_signal));
  auto last = peak_signal;
  for (int sample = bunchlength - 3; sample >= 0; sample--) {
    auto current = static_cast<uint16_t>(gRandom->Uniform(0, last));
    bunch[sample] = current;
    last = current;
  }
  return bunch;
}

void add_bunch_to_buffer(std::vector<uint16_t>& buffer, const std::vector<uint16_t>& bunch, uint8_t starttime)
{
  for (int sample = 0; sample < bunch.size(); sample++) {
    buffer[starttime - sample] = bunch[bunch.size() - 1 - sample];
  }
}

BOOST_AUTO_TEST_CASE(FastORTimeSeries_test)
{
  // test fill and integral
  for (int itest = 0; itest < 500; itest++) {
    auto [starttime, zerosuppressed, reference] = generatePulseTimeReversed();
    FastORTimeSeries testcase(13, zerosuppressed, starttime);
    auto adcs = testcase.getADCs();
    BOOST_CHECK_EQUAL_COLLECTIONS(adcs.begin(), adcs.end(), reference.begin(), reference.end());
    BOOST_CHECK_EQUAL(testcase.calculateL1TimeSum(8), calculateTimesum(reference, 8));
  }

  // test case where a normal FEC channel is identified as TRU channel. FEC channel can have lenght of 15 and would therefore cause an overflow in the FEC channel (max lenght 14)
  auto starttime = 14;
  auto bunch = generateSmallBunch(14);
  BOOST_CHECK_EXCEPTION(FastORTimeSeries(14, bunch, starttime), FastOrStartTimeInvalidException, [starttime](const FastOrStartTimeInvalidException& e) { return e.getStartTime() == starttime; });

  return;

  // test adding 2 bunches
  for (int itest = 0; itest < 500; itest++) {
    auto length_bunch1 = static_cast<int>(gRandom->Uniform(3, 5)),
         length_bunch2 = static_cast<int>(gRandom->Uniform(3, 5));
    auto sumbunchlength = length_bunch1 + length_bunch2;
    auto offset_bunch1 = static_cast<int>(gRandom->Uniform(0, 13 - sumbunchlength)),
         offset_bunch2 = static_cast<int>(gRandom->Uniform(0, 13 - sumbunchlength - offset_bunch1));
    auto bunch1 = generateSmallBunch(length_bunch1),
         bunch2 = generateSmallBunch(length_bunch2);
    auto starttime_bunch1 = offset_bunch1 + length_bunch1,
         starttime_bunch2 = starttime_bunch1 + offset_bunch2 + length_bunch2;
    std::vector<uint16_t> buffer_reversed{13};
    add_bunch_to_buffer(buffer_reversed, bunch2, starttime_bunch2);
    add_bunch_to_buffer(buffer_reversed, bunch1, starttime_bunch1);
    FastORTimeSeries testcase(13, bunch2, starttime_bunch2);
    testcase.setTimeSamples(bunch1, starttime_bunch1);
    auto adcs_timeordered = getReversed(buffer_reversed);
    auto adcs_timeseries_reversed = testcase.getADCs();
    BOOST_CHECK_EQUAL_COLLECTIONS(adcs_timeseries_reversed.begin(), adcs_timeseries_reversed.end(), adcs_timeordered.begin(), adcs_timeordered.end());
  }
}

} // namespace emcal

} // namespace o2
