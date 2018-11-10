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

#define BOOST_TEST_MODULE Test MID geometry transformer
#define BOOST_TEST_DYN_LINK

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

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.00001))
BOOST_DATA_TEST_CASE(InverseTransformation, boost::unit_test::data::xrange(72), deId)
{

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> distX(-127.5, 127.5);
  std::uniform_real_distribution<double> distY(-40., 40.);

  for (int ipt = 0; ipt < 1000; ++ipt) {
    Point3D<double> point(distX(mt), distY(mt), 0.);

    Point3D<double> globalPoint = GEOM::geoTrans.localToGlobal(deId, point.x(), point.y());
    Point3D<double> localPoint = GEOM::geoTrans.globalToLocal(deId, globalPoint.x(), globalPoint.y(), globalPoint.z());

    BOOST_TEST(localPoint.x() == point.x());
    BOOST_TEST(localPoint.y() == point.y());
    BOOST_TEST(localPoint.z() == point.z());
  }
}

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.00001))
BOOST_DATA_TEST_CASE(InverseTransformationSlope, boost::unit_test::data::xrange(72), deId)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> distX(-1., 1.);
  std::uniform_real_distribution<double> distY(-1., 1.);

  for (int ipt = 0; ipt < 1000; ++ipt) {
    Vector3D<double> slope(distX(mt), distY(mt), 1.);
    Vector3D<double> globalSlope = GEOM::geoTrans.localToGlobal(deId, slope);
    Vector3D<double> localSlope = GEOM::geoTrans.globalToLocal(deId, globalSlope);

    BOOST_TEST(localSlope.x() == slope.x());
    BOOST_TEST(localSlope.y() == slope.y());
    BOOST_TEST(localSlope.z() == slope.z());
  }
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
