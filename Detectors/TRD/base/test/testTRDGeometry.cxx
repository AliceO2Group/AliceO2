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

namespace o2
{
namespace trd
{

/// \brief Test the TRDGeometry class
//
///
BOOST_AUTO_TEST_CASE(TRDGeometrytest1)
{
  TRDGeometry& geom = TRDGeometry::instance();
}

} // namespace trd
} // namespace o2
