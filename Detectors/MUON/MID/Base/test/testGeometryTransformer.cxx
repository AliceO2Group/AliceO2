// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/test/testGeometryTransformer.cxx
/// \brief  Test geometry transformer for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 March 2018

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>

#include "MIDBase/GeometryTransformer.h"
#include "MathUtils/Cartesian3D.h"

#include <vector>
#include <random>
#include <cmath>

struct GEOM {
  static o2::mid::GeometryTransformer geoTrans;
};

o2::mid::GeometryTransformer GEOM::geoTrans;

BOOST_AUTO_TEST_SUITE(o2_mid_geometryTransformer)
BOOST_FIXTURE_TEST_SUITE(geo, GEOM)

std::vector<Point3D<float>> generatePoints(int ntimes)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> distX(-127.5, 127.5);
  std::uniform_real_distribution<double> distY(-40., 40.);

  std::vector<Point3D<float>> points;

  for (int itime = 0; itime < ntimes; ++itime) {
    points.emplace_back(Point3D<float>(distX(mt), distY(mt), 0.));
  }

  return points;
}

BOOST_DATA_TEST_CASE(InverseTransformation, boost::unit_test::data::xrange(72) * generatePoints(1000), deId, point)
{
  Point3D<float> globalPoint = GEOM::geoTrans.localToGlobal(deId, point.x(), point.y());
  Point3D<float> localPoint = GEOM::geoTrans.globalToLocal(deId, globalPoint.x(), globalPoint.y(), globalPoint.z());
  float relTolerance = 0.001;
  float absTolerance = 1.;
  float minValue = 0.02;
  float tolerance = (std::abs(localPoint.x()) < minValue) ? absTolerance : relTolerance;
  BOOST_TEST(localPoint.x() == point.x(), boost::test_tools::tolerance(tolerance));
  tolerance = (std::abs(localPoint.y()) < minValue) ? absTolerance : relTolerance;
  BOOST_TEST(localPoint.y() == point.y(), boost::test_tools::tolerance(tolerance));
  tolerance = (std::abs(localPoint.z()) < minValue) ? absTolerance : relTolerance;
  BOOST_TEST(localPoint.z() == point.z(), boost::test_tools::tolerance(tolerance));
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
