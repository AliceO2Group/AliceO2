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

#define BOOST_TEST_MODULE Test AlpideSimResponse
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include <fairlogger/Logger.h>

using namespace o2::itsmft;

BOOST_AUTO_TEST_CASE(AlpideSimResponse_test)
{
  // test for the templated Descriptor struct
  AlpideSimResponse resp;
  resp.initData(1);
  float vCol = 1.e-4, vRow = 1.e-4, vDepth = 10.e-4;
  LOG(info) << "Checking response from vRow:" << vCol << " vCol:" << vCol
            << " Depth:" << vDepth;
  bool flipCol, flipRow;
  auto respMat = resp.getResponse(vRow, vCol, resp.getDepthMax() - vDepth, flipRow, flipCol);
  BOOST_CHECK(respMat != nullptr);
  respMat->print(flipRow, flipCol);
  // repsonse at central pixel for electron close to the surface should be >>0
  int pixCen = respMat->getNPix() / 2;
  LOG(info) << "Response at central pixel " << pixCen << ":" << pixCen
            << " is " << respMat->getValue(pixCen, pixCen, flipRow, flipCol);
  BOOST_CHECK(respMat->getValue(pixCen, pixCen, flipRow, flipCol) > 1e-6);
  //
  // check normalization
  float norm = 0.f;
  for (int ir = respMat->getNPix(); ir--;) {
    for (int ic = respMat->getNPix(); ic--;) {
      norm += respMat->getValue(ir, ic, flipRow, flipCol);
    }
  }
  LOG(info) << "Total response to 1 electron: " << norm;
  BOOST_CHECK(norm > 0.1);
}
