// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TPC Base
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Sector.h"
#include <cmath>

namespace o2
{
namespace tpc
{

BOOST_AUTO_TEST_CASE(Point3D_test)
{
  Point3D<double> testpoint(2., 3., 4.);
  BOOST_CHECK_CLOSE(testpoint.X(), 2., 1E-12);
  BOOST_CHECK_CLOSE(testpoint.Y(), 3., 1E-12);
  BOOST_CHECK_CLOSE(testpoint.Z(), 4., 1E-12);
}

// test Sector stuff
BOOST_AUTO_TEST_CASE(TPCSectorTest)
{
  // generate a point in each possible sector
  const auto HALFSECTORS = Sector::MAXSECTOR / 2;
  float x[HALFSECTORS];
  float y[HALFSECTORS];

  const auto startphi = 10. * M_PI / 180.;
  const auto deltaphi = 20. * M_PI / 180.;
  for (int s = 0; s < HALFSECTORS; ++s) {
    x[s] = std::cos(s * deltaphi + startphi);
    y[s] = std::sin(s * deltaphi + startphi);
  }

  for (int s = 0; s < HALFSECTORS; ++s) {
    // check for A side
    BOOST_CHECK_EQUAL(Sector::ToSector(x[s], y[s], 1.f), s);
    // check for C side
    BOOST_CHECK_EQUAL(Sector::ToSector(x[s], y[s], -1.f), s + HALFSECTORS);
  }
}
} // namespace tpc
} // namespace o2
