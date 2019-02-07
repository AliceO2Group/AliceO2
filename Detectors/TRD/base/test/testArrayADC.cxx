// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCMapper.cxx
/// \brief This task tests the mapper function
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TRD_ArrayADC
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "TRDBase/TRDArrayADC.h"

namespace o2
{
namespace tcp
{

/// \brief Test the arrayadc class operation
//
/// check the bit manipulations
BOOST_AUTO_TEST_CASE(ArrayADCtest1)
{
  TRDArrayADC data = new TRDArrayADC();
  //set bits as corrupted and see if they are detected correctly
}

} // namespace tcp
} // namespace o2
