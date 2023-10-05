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

BOOST_AUTO_TEST_CASE(MinorAltroDecodingError_test)
{
  BOOST_CHECK_EQUAL(MinorAltroDecodingError::getNumberOfErrorTypes(), 8);
  std::array<std::string, 8> errornames = {{"ChannelEndPayloadUnexpected",
                                            "ChannelPayloadExceed",
                                            "ChannelOrderError",
                                            "ChannelHeader",
                                            "BunchHeaderNull",
                                            "BunchLengthExceed",
                                            "BunchLengthAllowExceed",
                                            "BunchStarttimeExceed"}},
                             errortitles = {{"Channel end unexpected",
                                             "Channel exceed",
                                             "FEC order",
                                             "Channel header invalid",
                                             "Bunch header null",
                                             "Bunch length exceed",
                                             "Bunch length impossible",
                                             "Bunch starttime exceed"}},
                             errordescriptions = {{"Unexpected end of payload in altro channel payload!",
                                                   "Trying to access out-of-bound payload!",
                                                   "Invalid FEC order",
                                                   "Invalid channel header",
                                                   "Bunch header 0 or not configured!",
                                                   "Bunch length exceeding channel payload size!",
                                                   "Bunch length exceeding max. possible bunch size!",
                                                   "Bunch start time outside range!"}};
  std::array<MinorAltroDecodingError::ErrorType_t, 8> errortypes = {{MinorAltroDecodingError::ErrorType_t::CHANNEL_END_PAYLOAD_UNEXPECT,
                                                                     MinorAltroDecodingError::ErrorType_t::CHANNEL_PAYLOAD_EXCEED,
                                                                     MinorAltroDecodingError::ErrorType_t::CHANNEL_ORDER,
                                                                     MinorAltroDecodingError::ErrorType_t::CHANNEL_HEADER,
                                                                     MinorAltroDecodingError::ErrorType_t::BUNCH_HEADER_NULL,
                                                                     MinorAltroDecodingError::ErrorType_t::BUNCH_LENGTH_EXCEED,
                                                                     MinorAltroDecodingError::ErrorType_t::BUNCH_LENGTH_ALLOW_EXCEED,
                                                                     MinorAltroDecodingError::ErrorType_t::BUNCH_STARTTIME}};
  for (int errortype = 0; errortype < MinorAltroDecodingError::getNumberOfErrorTypes(); errortype++) {
    BOOST_CHECK_EQUAL(MinorAltroDecodingError::errorTypeToInt(errortypes[errortype]), errortype);
    BOOST_CHECK_EQUAL(MinorAltroDecodingError::intToErrorType(errortype), errortypes[errortype]);
    BOOST_CHECK_EQUAL(std::string(MinorAltroDecodingError::getErrorTypeName(errortype)), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(MinorAltroDecodingError::getErrorTypeName(errortypes[errortype])), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(MinorAltroDecodingError::getErrorTypeTitle(errortype)), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(MinorAltroDecodingError::getErrorTypeTitle(errortypes[errortype])), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(MinorAltroDecodingError::getErrorTypeDescription(errortype)), errordescriptions[errortype]);
    BOOST_CHECK_EQUAL(std::string(MinorAltroDecodingError::getErrorTypeDescription(errortypes[errortype])), errordescriptions[errortype]);
  }
}

} // namespace emcal
} // namespace o2