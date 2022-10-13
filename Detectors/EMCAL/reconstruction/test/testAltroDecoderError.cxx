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
#include <EMCALReconstruction/AltroDecoder.h>

namespace o2
{
namespace emcal
{

void testThrow(AltroDecoderError::ErrorType_t errortype)
{
  throw AltroDecoderError(errortype, AltroDecoderError::getErrorTypeDescription(errortype));
}

BOOST_AUTO_TEST_CASE(AltroDecoderError_test)
{
  BOOST_CHECK_EQUAL(AltroDecoderError::getNumberOfErrorTypes(), 8);
  std::array<std::string, 8> errornames = {{"RCUTrailerError",
                                            "RCUTrailerVersionError",
                                            "RCUTrailerSizeError",
                                            "BunchHeaderError",
                                            "BunchLengthError",
                                            "ALTROPayloadError",
                                            "ALTROMappingError",
                                            "ChannelError"}},
                             errortitles = {{"RCU Trailer",
                                             "RCU Version",
                                             "RCU Trailer Size",
                                             "ALTRO Bunch Header",
                                             "ALTRO Bunch Length",
                                             "ALTRO Payload",
                                             "ALTRO Mapping",
                                             "Channel"}},
                             errordescriptions = {{"RCU trailer decoding error",
                                                   "Inconsistent RCU trailer version",
                                                   "Invalid RCU trailer size",
                                                   "Inconsistent bunch header",
                                                   "Bunch length exceeding payload size",
                                                   "Payload could not be decoded",
                                                   "Invalid hardware address in ALTRO mapping",
                                                   "Channels not initizalized"}};
  std::array<AltroDecoderError::ErrorType_t, 8> errortypes = {{AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR,
                                                               AltroDecoderError::ErrorType_t::RCU_VERSION_ERROR,
                                                               AltroDecoderError::ErrorType_t::RCU_TRAILER_SIZE_ERROR,
                                                               AltroDecoderError::ErrorType_t::ALTRO_BUNCH_HEADER_ERROR,
                                                               AltroDecoderError::ErrorType_t::ALTRO_BUNCH_LENGTH_ERROR,
                                                               AltroDecoderError::ErrorType_t::ALTRO_PAYLOAD_ERROR,
                                                               AltroDecoderError::ErrorType_t::ALTRO_MAPPING_ERROR,
                                                               AltroDecoderError::ErrorType_t::CHANNEL_ERROR}};
  for (int errortype = 0; errortype < AltroDecoderError::getNumberOfErrorTypes(); errortype++) {
    BOOST_CHECK_EQUAL(AltroDecoderError::errorTypeToInt(errortypes[errortype]), errortype);
    BOOST_CHECK_EQUAL(AltroDecoderError::intToErrorType(errortype), errortypes[errortype]);
    BOOST_CHECK_EQUAL(std::string(AltroDecoderError::getErrorTypeName(errortype)), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(AltroDecoderError::getErrorTypeName(errortypes[errortype])), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(AltroDecoderError::getErrorTypeTitle(errortype)), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(AltroDecoderError::getErrorTypeTitle(errortypes[errortype])), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(AltroDecoderError::getErrorTypeDescription(errortype)), errordescriptions[errortype]);
    BOOST_CHECK_EQUAL(std::string(AltroDecoderError::getErrorTypeDescription(errortypes[errortype])), errordescriptions[errortype]);
    auto expecttype = errortypes[errortype];
    BOOST_CHECK_EXCEPTION(testThrow(errortypes[errortype]), AltroDecoderError, [expecttype](const AltroDecoderError& ex) { return ex.getErrorType() == expecttype; });
  }
}

} // namespace emcal
} // namespace o2