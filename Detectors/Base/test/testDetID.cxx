// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test DetID
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "DetectorsBase/DetID.h"

using namespace o2::Base;

BOOST_AUTO_TEST_CASE(DetID_test)
{
  // test for the templated Descriptor struct
  for (DetID::ID id=DetID::First; id<=DetID::Last; id++) {
    DetID det(id);
    std::cout << "#" << id << " Detector " << det.getName()
              << " ID=" << det << " mask: " << det.getMask() <<std::endl;
    BOOST_CHECK(id == det);

    // test that all names are initialized
    BOOST_CHECK(std::strlen(det.getName()) == 3);
  }

  {
    // test specific name access
    DetID det(DetID::ITS);
    BOOST_CHECK(std::strcmp(det.getName(),"ITS") == 0);
  }
}
