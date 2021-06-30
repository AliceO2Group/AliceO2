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
#define BOOST_TEST_MODULE Test Framework InfoLoggerTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <InfoLogger/InfoLogger.hxx>
using namespace AliceO2::InfoLogger;

BOOST_AUTO_TEST_CASE(InfoLoggerTest)
{

  // define infologger output to stdout, as we don't want to use the default infoLoggerD pipe which might not be running here
  setenv("O2_INFOLOGGER_MODE", "stdout", 1);

  // create the infologger interface
  InfoLogger theLog;

  // log a test message
  BOOST_CHECK(theLog.log("This is a log message test to stdout") == 0);
}
