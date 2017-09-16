// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  bool flipX, flipY;
  auto respMat = resp.getResponse(1e-4,1e-4,resp.getZMax()-10e-04,flipX,flipY);
  BOOST_CHECK( respMat!=nullptr );
  respMat->print();
  // repsonse at central pixel for electron close to the surface should be >>0
  int pixCen = respMat->getNPix()/2;
  LOG(INFO) << "Response at central pixel " << pixCen << ":" << pixCen
	    << " is " << respMat->getValue(pixCen,pixCen,flipX,flipY) << FairLogger::endl;
  BOOST_CHECK(respMat->getValue(pixCen,pixCen,flipX,flipY) > 1e-6);
  //
  // check normalization
  float norm = 0.f;
  for (int ix=respMat->getNPix();ix--;) {
    for (int iy=respMat->getNPix();iy--;) {
      norm += respMat->getValue(ix,iy,flipX,flipY);
    }
  }
  LOG(INFO) << "Total response to 1 electron: " << norm << FairLogger::endl;
  BOOST_CHECK(norm > 0.1);
}
