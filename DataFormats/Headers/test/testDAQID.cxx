// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test DAQID class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Headers/DAQID.h"
#include <boost/test/unit_test.hpp>
#include <iostream>

// @brief consistency test for O2 origin <-> DAQ Source ID mapping
// @author ruben.shahoyan@cern.ch

using namespace o2::header;

BOOST_AUTO_TEST_CASE(DAQIDTEST)
{

  for (int i = 0; i < DAQID::MAXDAQ + 5; i++) {
    auto vo2 = DAQID::DAQtoO2(i);
    auto daq = DAQID::O2toDAQ(vo2);
    if (vo2 != DAQID::DAQtoO2(DAQID::INVALID)) {
      std::cout << "DAQ SourceID " << i << " <-> " << vo2.str << std::endl;
    }
    BOOST_CHECK(i == daq || vo2 == DAQID::DAQtoO2(DAQID::INVALID));
  }
  std::cout << "DAQ INVALID  " << int(DAQID::INVALID) << " <-> " << DAQID::DAQtoO2(DAQID::INVALID).str << std::endl;
  std::cout << "DAQ UNLOADED " << int(DAQID::UNLOADED) << " <-> " << DAQID::DAQtoO2(DAQID::UNLOADED).str << std::endl;
  BOOST_CHECK(DAQID::DAQtoO2(DAQID::UNLOADED) == o2::header::gDataOriginUnloaded);
  BOOST_CHECK(DAQID::O2toDAQ(o2::header::gDataOriginUnloaded) == DAQID::UNLOADED);
}
