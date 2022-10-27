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

#define BOOST_TEST_MODULE Test EMCAL Calib
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <array>
#include <EMCALReconstruction/RawDecodingError.h>

namespace o2
{
namespace emcal
{

void testThrow(RawDecodingError::ErrorType_t errtype, unsigned int feeID)
{
  throw RawDecodingError(errtype, feeID);
}

BOOST_AUTO_TEST_CASE(RawDecodingError_test)
{
  BOOST_CHECK_EQUAL(RawDecodingError::getNumberOfErrorTypes(), 7);
  std::array<std::string, 7> errornames = {{"PageNotFound",
                                            "HeaderDecoding",
                                            "PayloadDecoding",
                                            "HeaderCorruption",
                                            "PageStartInvalid",
                                            "PayloadCorruption",
                                            "TrailerDecoding"}},
                             errortitles = {{"Page not found",
                                             "Header decoding",
                                             "Payload decoding",
                                             "Header corruption",
                                             "Page start invalid",
                                             "Payload corruption",
                                             "Trailer decoding"}},
                             errordescriptions = {{"Page with requested index not found",
                                                   "RDH of page cannot be decoded",
                                                   "Payload of page cannot be decoded",
                                                   "Access to header not belonging to requested superpage",
                                                   "Page decoding starting outside payload size",
                                                   "Access to payload not belonging to requested superpage",
                                                   "Inconsistent trailer in memory"}};
  std::array<RawDecodingError::ErrorType_t, 7> errortypes = {{
    RawDecodingError::ErrorType_t::PAGE_NOTFOUND,
    RawDecodingError::ErrorType_t::HEADER_DECODING,
    RawDecodingError::ErrorType_t::PAYLOAD_DECODING,
    RawDecodingError::ErrorType_t::HEADER_INVALID,
    RawDecodingError::ErrorType_t::PAGE_START_INVALID,
    RawDecodingError::ErrorType_t::PAYLOAD_INVALID,
    RawDecodingError::ErrorType_t::TRAILER_DECODING,
  }};
  for (int errortype = 0; errortype < RawDecodingError::getNumberOfErrorTypes(); errortype++) {
    BOOST_CHECK_EQUAL(RawDecodingError::ErrorTypeToInt(errortypes[errortype]), errortype);
    BOOST_CHECK_EQUAL(RawDecodingError::intToErrorType(errortype), errortypes[errortype]);
    BOOST_CHECK_EQUAL(std::string(RawDecodingError::getErrorCodeNames(errortype)), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(RawDecodingError::getErrorCodeNames(errortypes[errortype])), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(RawDecodingError::getErrorCodeTitles(errortype)), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(RawDecodingError::getErrorCodeTitles(errortypes[errortype])), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(RawDecodingError::getErrorCodeDescription(errortype)), errordescriptions[errortype]);
    BOOST_CHECK_EQUAL(std::string(RawDecodingError::getErrorCodeDescription(errortypes[errortype])), errordescriptions[errortype]);
    for (unsigned int errtype = 0; errtype < 7; errtype++) {
      auto errtypeval = RawDecodingError::intToErrorType(errtype);
      for (unsigned int feeID = 0; feeID < 40; feeID++) {
        auto checker = [errtypeval, feeID](const RawDecodingError& test) {
          return test.getErrorType() == errtypeval && test.getFECID() == feeID;
        };
        BOOST_CHECK_EXCEPTION(testThrow(errtypeval, feeID),
                              RawDecodingError, checker);
      }
    }
  }
}

} // namespace emcal
} // namespace o2