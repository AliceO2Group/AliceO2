// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTRDGeometry.cxx
/// \brief This task tests the TRDGeometry
/// \author Sean Murray, murrays@cern.ch

#define BOOST_TEST_MODULE Test TRD_Geometry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "TRDBase/TRDGeometry.h"
#include <iostream>

namespace o2
{
namespace trd
{

/// \brief Test the TRDGeometry class
//
///
BOOST_AUTO_TEST_CASE(TRDGeometry_test1)
{
  //arbitrary chosen
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowPos(1, 1, 3), 154.5, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowSize(1, 1, 3), 7.5, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRow0(1, 1), 177, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowEnd(1, 1), 57, 1e-3);
  //start
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowPos(0, 0, 3), 278.5, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowSize(0, 0, 3), 7.5, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRow0(0, 0), 301, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowEnd(0, 0), 181, 1e-3);
  //end of trd.
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowPos(5, 4, 0), -204, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowSize(5, 4, 3), 9, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRow0(5, 4), -204, 1e-3);
  BOOST_CHECK_CLOSE(TRDGeometry::instance()->getRowEnd(5, 4), -347, 1e-3);
}

} // namespace trd
} // namespace o2
