// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
#include "FairLogger.h"

using namespace o2::ITSMFT;

BOOST_AUTO_TEST_CASE(AlpideSimResponse_test)
{
  // test for the templated Descriptor struct
  AlpideSimResponse resp;
  resp.initData();
  float x=1.e-4, y=1.e-4, z=10.e-4;
  LOG(INFO) << "Checking response from X:" << x << " Y:" << y << " Z:" << z << FairLogger::endl;
  auto rspmat = resp.getResponse(1e-4,1e-4,resp.getZMax()-10e-04);
  BOOST_CHECK(rspmat != nullptr);
  rspmat->print();
  // repsonse at central pixel for electron close to the surface should be >>0
  int pixCen = resp.getNPix()/2;
  LOG(INFO) << "Response at central pixel " << pixCen << ":" << pixCen
	    << " is " << rspmat->getValue(pixCen,pixCen) << FairLogger::endl;
  BOOST_CHECK(rspmat->getValue(pixCen,pixCen) > 1e-6);
  //
  // check normalization
  float norm = 0.f;
  for (int ix=resp.getNPix();ix--;) {
    for (int iy=resp.getNPix();iy--;) {
      norm += rspmat->getValue(ix,iy);
    }
  }
  LOG(INFO) << "Total response to 1 electron: " << norm << FairLogger::endl;
  BOOST_CHECK(norm > 0.1);
}
