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
#include <EMCALReconstruction/CaloRawFitter.h>

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(CaloRawFitterError_test)
{
  BOOST_CHECK_EQUAL(CaloRawFitter::getNumberOfErrorTypes(), 5);
  std::array<std::string, 5> errornames = {{"SampleUninitalized",
                                            "NoConvergence",
                                            "Chi2Error",
                                            "BunchRejected",
                                            "LowSignal"}},
                             errortitles = {{"sample uninitalized",
                                             "No convergence",
                                             "Chi2 error",
                                             "Bunch rejected",
                                             "Low signal"}},
                             errordescriptions = {{"Sample for fit not initialzied or bunch length is 0",
                                                   "Fit of the raw bunch was not successful",
                                                   "Chi2 of the fit could not be determined",
                                                   "Calo bunch could not be selected",
                                                   "No ADC value above threshold found"}};
  std::array<CaloRawFitter::RawFitterError_t, 5> errortypes = {{CaloRawFitter::RawFitterError_t::SAMPLE_UNINITIALIZED,
                                                                CaloRawFitter::RawFitterError_t::FIT_ERROR,
                                                                CaloRawFitter::RawFitterError_t::CHI2_ERROR,
                                                                CaloRawFitter::RawFitterError_t::BUNCH_NOT_OK,
                                                                CaloRawFitter::RawFitterError_t::LOW_SIGNAL}};
  for (int errortype = 0; errortype < CaloRawFitter::getNumberOfErrorTypes(); errortype++) {
    BOOST_CHECK_EQUAL(CaloRawFitter::getErrorNumber(errortypes[errortype]), errortype);
    BOOST_CHECK_EQUAL(CaloRawFitter::intToErrorType(errortype), errortypes[errortype]);
    BOOST_CHECK_EQUAL(std::string(CaloRawFitter::getErrorTypeName(errortype)), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(CaloRawFitter::getErrorTypeName(errortypes[errortype])), errornames[errortype]);
    BOOST_CHECK_EQUAL(std::string(CaloRawFitter::getErrorTypeTitle(errortype)), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(CaloRawFitter::getErrorTypeTitle(errortypes[errortype])), errortitles[errortype]);
    BOOST_CHECK_EQUAL(std::string(CaloRawFitter::getErrorTypeDescription(errortype)), errordescriptions[errortype]);
    BOOST_CHECK_EQUAL(std::string(CaloRawFitter::getErrorTypeDescription(errortypes[errortype])), errordescriptions[errortype]);
  }
}

} // namespace emcal
} // namespace o2