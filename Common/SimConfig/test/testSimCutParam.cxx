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

#define BOOST_TEST_MODULE Test SimConfig SimCutParam
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimConfig/SimParams.h"
#include "CommonUtils/ConfigurableParam.h"
#include <boost/property_tree/ptree.hpp>

using namespace o2::conf;

BOOST_AUTO_TEST_CASE(test1)
{
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<bool>("SimCutParams.trackSeed"), false);
  BOOST_CHECK_EQUAL(ConfigurableParam::getValueAs<double>("SimCutParams.ZmaxA"), 1E20);

  auto& par = SimCutParams::Instance();
  ConfigurableParam::setValue("SimCutParams", "ZmaxA", 100.);
  BOOST_CHECK_CLOSE(par.ZmaxA, 100., 0.001);
  BOOST_CHECK_CLOSE(ConfigurableParam::getValueAs<double>("SimCutParams.ZmaxA"), 100., 0.001);

  ConfigurableParam::updateFromString("SimCutParams.ZmaxA=20");
  BOOST_CHECK_CLOSE(par.ZmaxA, 20., 0.001);
  BOOST_CHECK_CLOSE(ConfigurableParam::getValueAs<double>("SimCutParams.ZmaxA"), 20., 0.001);
}
