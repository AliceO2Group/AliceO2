// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCSpaceChargeBase.cxx
/// \author Ernst Hellbaer

#define BOOST_TEST_MODULE Test TPC Space - Charge Base Class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "AliTPCSpaceCharge3DCalc.h"

/// @brief Basic test if we can create the method class
BOOST_AUTO_TEST_CASE(TPCSpaceChargeBase_test1)
{
  auto spacecharge = new AliTPCSpaceCharge3DCalc;
  delete spacecharge;
}
