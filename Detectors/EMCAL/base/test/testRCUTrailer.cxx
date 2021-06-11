// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test EMCAL Base
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <array>
#include <boost/test/unit_test.hpp>
#include "EMCALBase/RCUTrailer.h"

/// \macro Test implementation of the EMCAL RCU trailer
///
/// Test coverage:
/// - ALTRO config
/// - RCU ID
/// - ALTRO buffers
/// - error counters
BOOST_AUTO_TEST_CASE(RCUTrailer_test)
{
  // common settings
  int firmware = 2,
      activeFECA = 0,
      activeFECB = 1,
      baselineCorr = 0,
      presamples = 0,
      postsamples = 0,
      glitchfilter = 0,
      presamplesNoZS = 1,
      postsamplesNoZS = 1,
      samplesChannel = 15,
      samplesPretrigger = 0,
      timesample = 100;
  bool havePolarity = false,
       haveSecBaselineCorr = false,
       haveZS = true,
       haveSpareReadout = true;
  o2::emcal::RCUTrailer::BufferMode_t bufmode = o2::emcal::RCUTrailer::BufferMode_t::NBUFFERS4;

  o2::emcal::RCUTrailer trailer;
  trailer.setRCUID(0);
  trailer.setFirmwareVersion(firmware);
  trailer.setActiveFECsA(activeFECA);
  trailer.setActiveFECsB(activeFECB);
  trailer.setPayloadSize(20);
  trailer.setTimeSamplePhaseNS(425, timesample);
  trailer.setBaselineCorrection(baselineCorr);
  trailer.setPolarity(havePolarity);
  trailer.setNumberOfPresamples(presamples);
  trailer.setNumberOfPostsamples(postsamples);
  trailer.setSecondBaselineCorrection(false);
  trailer.setGlitchFilter(glitchfilter);
  trailer.setNumberOfNonZeroSuppressedPostsamples(postsamplesNoZS);
  trailer.setNumberOfNonZeroSuppressedPresamples(presamplesNoZS);
  trailer.setNumberOfPretriggerSamples(samplesPretrigger);
  trailer.setNumberOfSamplesPerChannel(samplesChannel);
  trailer.setZeroSuppression(haveZS);
  trailer.setSparseReadout(haveSpareReadout);
  trailer.setNumberOfAltroBuffers(bufmode);

  //
  // check ALTRO config
  //
  auto encoded_config = trailer.encode();
  auto trailer_decoded_config = o2::emcal::RCUTrailer::constructFromPayloadWords(encoded_config);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getFirmwareVersion(), firmware);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getNumberOfPresamples(), presamples);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getNumberOfPostsamples(), postsamples);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getNumberOfNonZeroSuppressedPresamples(), presamplesNoZS);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getNumberOfNonZeroSuppressedPostsamples(), postsamplesNoZS);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getGlitchFilter(), glitchfilter);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getNumberOfPretriggerSamples(), samplesPretrigger);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getNumberOfSamplesPerChannel(), samplesChannel);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getPolarity(), havePolarity);
  BOOST_CHECK_EQUAL(trailer_decoded_config.hasSecondBaselineCorr(), haveSecBaselineCorr);
  BOOST_CHECK_EQUAL(trailer_decoded_config.hasZeroSuppression(), haveZS);
  BOOST_CHECK_EQUAL(trailer_decoded_config.isSparseReadout(), haveSpareReadout);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getActiveFECsA(), activeFECA);
  BOOST_CHECK_EQUAL(trailer_decoded_config.getActiveFECsB(), activeFECB);
  BOOST_CHECK_CLOSE(trailer_decoded_config.getTimeSampleNS(), timesample, 1);
  BOOST_CHECK_CLOSE(trailer_decoded_config.getL1PhaseNS(), 25, 1);

  //
  // Check RCU ID
  //
  for (int ircu = 0; ircu < 46; ircu++) {
    trailer.setRCUID(ircu);
    auto encoded_rcu = trailer.encode();
    auto trailer_decoded_rcu = o2::emcal::RCUTrailer::constructFromPayloadWords(encoded_rcu);
    BOOST_CHECK_EQUAL(trailer_decoded_rcu.getRCUID(), ircu);
  }
  trailer.setRCUID(0);

  //
  // Check buffer modes
  //
  std::map<o2::emcal::RCUTrailer::BufferMode_t, int> buffertests = {{o2::emcal::RCUTrailer::BufferMode_t::NBUFFERS4, 4}, {o2::emcal::RCUTrailer::BufferMode_t::NBUFFERS8, 8}};
  for (auto [bufmode, nbuffers] : buffertests) {
    trailer.setNumberOfAltroBuffers(bufmode);
    auto encoded_buffer = trailer.encode();
    auto trailer_decoded_buffer = o2::emcal::RCUTrailer::constructFromPayloadWords(encoded_buffer);
    BOOST_CHECK_EQUAL(trailer_decoded_buffer.getNumberOfAltroBuffers(), nbuffers);
  }
  trailer.setNumberOfAltroBuffers(bufmode);

  //
  // Check error counters
  //
  std::array<int, 10> nerrors = {0, 1, 6, 10, 112, 232, 255, 52, 22, 76};
  for (auto error : nerrors) {
    trailer.setNumberOfChannelLengthMismatch(error);
    auto encoded_error = trailer.encode();
    auto trailer_decoded_error = o2::emcal::RCUTrailer::constructFromPayloadWords(encoded_error);
    BOOST_CHECK_EQUAL(trailer_decoded_error.getNumberOfChannelLengthMismatch(), error);
  }
  trailer.setNumberOfChannelAddressMismatch(0);
  for (auto error : nerrors) {
    trailer.setNumberOfChannelAddressMismatch(error);
    auto encoded_error = trailer.encode();
    auto trailer_decoded_error = o2::emcal::RCUTrailer::constructFromPayloadWords(encoded_error);
    BOOST_CHECK_EQUAL(trailer_decoded_error.getNumberOfChannelAddressMismatch(), error);
  }
  trailer.setNumberOfChannelAddressMismatch(0);
}