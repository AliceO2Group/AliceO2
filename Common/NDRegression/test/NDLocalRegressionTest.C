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

#define BOOST_TEST_MODULE Test NDLocalRegressionTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "NDLocalRegression/NDLocalRegression.h"

BOOST_AUTO_TEST_CASE(NDLocalRegressionTest_test)
{
  auto fitter = o2::nd_local_regression::NDLocalRegression();
  auto success = fitter.init();

  BOOST_CHECK(success == true);
}
