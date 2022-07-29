// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Javier Castillo

#define BOOST_TEST_MODULE Test MCHSimulation GeometryMisAligner
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "MCHGeometryCreator/Geometry.h"
#include "MCHGeometryTest/Helpers.h"
#include "MCHGeometryMisAligner/MisAligner.h"
#include "MCHGeometryTransformer/Transformations.h"
#include "TGeoManager.h"
#include "CommonConstants/MathConstants.h"
#include "MathUtils/Cartesian.h"
#include "Math/GenVector/Cartesian3D.h"
#include "boost/format.hpp"
#include <boost/test/data/test_case.hpp>
#include <iomanip>
#include <iostream>
#include <fmt/format.h>

namespace but = boost::unit_test;
namespace bdata = boost::unit_test::data;
namespace btools = boost::test_tools;

BOOST_TEST_DONT_PRINT_LOG_VALUE(o2::mch::geo::MisAligner)

struct GEOMETRY {
  GEOMETRY()
  {
    if (!gGeoManager) {
      o2::mch::test::createStandaloneGeometry();
      o2::mch::geo::addAlignableVolumes(*gGeoManager);
    }
  };
}; // namespace boost::test_toolsBOOST_TEST_DONT_PRINT_LOG_VALUE(o2::mch::geo::MisAligner)structGEOMETRY

BOOST_FIXTURE_TEST_SUITE(geometrymisaligner, GEOMETRY)

BOOST_AUTO_TEST_CASE(ZeroMisAlignHalfChambers, *but::tolerance(0.00001))
{
  BOOST_REQUIRE(gGeoManager != nullptr);

  std::vector<o2::detectors::AlignParam> params;

  // The misaligner
  o2::mch::geo::MisAligner aGMA;

  aGMA.misAlign(params);

  int nvols = params.size();
  for (int i = 0; i < nvols; i++) {
    BOOST_TEST(params[i].getX() == 0.0);
    BOOST_TEST(params[i].getY() == 0.0);
    BOOST_TEST(params[i].getZ() == 0.0);
    BOOST_TEST(params[i].getPsi() == 0.0);
    BOOST_TEST(params[i].getTheta() == 0.0);
    BOOST_TEST(params[i].getPhi() == 0.0);
  }
}

BOOST_AUTO_TEST_CASE(MisAlignHalfChambers, *but::tolerance(0.00001))
{
  BOOST_REQUIRE(gGeoManager != nullptr);

  std::vector<o2::detectors::AlignParam> params;

  // The misaligner
  o2::mch::geo::MisAligner aGMA;

  // To generate module mislaignment (not mandatory)
  aGMA.setModuleCartMisAlig(0.1, 0.0, 0.2, 0.0, 0.3, 0.0);
  aGMA.setModuleAngMisAlig(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  auto transformationB = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
  auto tB100 = transformationB(100);
  auto tB1000 = transformationB(1000);
  o2::math_utils::Point3D<double> poB;

  aGMA.misAlign(params);

  auto transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
  auto t100 = transformation(100);
  auto t1000 = transformation(1000);
  o2::math_utils::Point3D<double> po;
  tB100.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, poB);
  t100.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, po);

  BOOST_TEST(po.X() - poB.X() == 0.1);
  BOOST_TEST(po.Y() - poB.Y() == 0.2);
  BOOST_TEST(po.Z() - poB.Z() == 0.3);

  tB1000.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, poB);
  t1000.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, po);
  BOOST_TEST(po.X() - poB.X() == 0.1);
  BOOST_TEST(po.Y() - poB.Y() == 0.2 * std::cos(o2::constants::math::Deg2Rad * -0.794) - 0.3 * std::sin(o2::constants::math::Deg2Rad * -0.794));
  BOOST_TEST(po.Z() - poB.Z() == 0.2 * std::sin(o2::constants::math::Deg2Rad * -0.794) + 0.3 * std::cos(o2::constants::math::Deg2Rad * -0.794));

  int nvols = params.size();
  for (int i = 0; i < nvols; i++) {
    if (params[i].getSymName().find("DE") != std::string::npos) {
      BOOST_TEST(params[i].getX() == 0.0);
      BOOST_TEST(params[i].getY() == 0.0);
      BOOST_TEST(params[i].getZ() == 0.0);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    }
  }
}

BOOST_AUTO_TEST_CASE(MisAlignDetectionElements, *but::tolerance(0.00001))
{
  BOOST_REQUIRE(gGeoManager != nullptr);

  std::vector<o2::detectors::AlignParam> params;

  // The misaligner
  o2::mch::geo::MisAligner aGMA;

  // To generate detection element misalignment
  aGMA.setCartMisAlig(0.01, 0.0, 0.02, 0.0, 0.03, 0.0);
  aGMA.setAngMisAlig(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  aGMA.misAlign(params);

  int nvols = params.size();
  for (int i = 0; i < nvols; i++) {
    std::cout << params[i].getSymName() << std::endl;
    if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 100)) {
      BOOST_TEST(params[i].getX() == 0.01);
      BOOST_TEST(params[i].getY() == 0.02);
      BOOST_TEST(params[i].getZ() == 0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    } else if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 101)) {
      BOOST_TEST(params[i].getX() == -0.01);
      BOOST_TEST(params[i].getY() == 0.02);
      BOOST_TEST(params[i].getZ() == -0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    } else if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 102)) {
      BOOST_TEST(params[i].getX() == -0.01);
      BOOST_TEST(params[i].getY() == -0.02);
      BOOST_TEST(params[i].getZ() == 0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    } else if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 103)) {
      BOOST_TEST(params[i].getX() == 0.01);
      BOOST_TEST(params[i].getY() == -0.02);
      BOOST_TEST(params[i].getZ() == -0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    }
  }
}

BOOST_AUTO_TEST_CASE(MisAlignHCDE, *but::tolerance(0.00001))
{
  BOOST_REQUIRE(gGeoManager != nullptr);

  std::vector<o2::detectors::AlignParam> params;

  // The misaligner
  o2::mch::geo::MisAligner aGMA;

  // To generate half chmaber misalignment
  aGMA.setModuleCartMisAlig(0.1, 0.0, 0.2, 0.0, 0.3, 0.0);
  aGMA.setModuleAngMisAlig(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  // To generate detection element misalignment
  aGMA.setCartMisAlig(0.01, 0.0, 0.02, 0.0, 0.03, 0.0);
  aGMA.setAngMisAlig(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  auto transformationB = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
  auto tB100 = transformationB(100);
  auto tB1000 = transformationB(1000);
  o2::math_utils::Point3D<double> poB;

  aGMA.misAlign(params);

  auto transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
  auto t100 = transformation(100);
  auto t1000 = transformation(1000);
  o2::math_utils::Point3D<double> po;
  tB100.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, poB);
  t100.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, po);

  BOOST_TEST(po.X() - poB.X() == 0.1 + 0.01);
  BOOST_TEST(po.Y() - poB.Y() == 0.2 + 0.02);
  BOOST_TEST(po.Z() - poB.Z() == 0.3 + 0.03);

  tB1000.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, poB);
  t1000.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, po);
  BOOST_TEST(po.X() - poB.X() == 0.1 + 0.01);
  BOOST_TEST(po.Y() - poB.Y() == (0.2 * std::cos(o2::constants::math::Deg2Rad * -0.794) - 0.3 * std::sin(o2::constants::math::Deg2Rad * -0.794)) -
                                   (0.02 * std::cos(o2::constants::math::Deg2Rad * -0.794) - 0.03 * std::sin(o2::constants::math::Deg2Rad * -0.794)));
  BOOST_TEST(po.Z() - poB.Z() == (0.2 * std::sin(o2::constants::math::Deg2Rad * -0.794) + 0.3 * std::cos(o2::constants::math::Deg2Rad * -0.794)) -
                                   (0.02 * std::sin(o2::constants::math::Deg2Rad * -0.794) + 0.03 * std::cos(o2::constants::math::Deg2Rad * -0.794)));

  int nvols = params.size();
  for (int i = 0; i < nvols; i++) {
    std::cout << params[i].getSymName() << std::endl;
    if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 100)) {
      BOOST_TEST(params[i].getX() == 0.01);
      BOOST_TEST(params[i].getY() == 0.02);
      BOOST_TEST(params[i].getZ() == 0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    } else if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 101)) {
      BOOST_TEST(params[i].getX() == -0.01);
      BOOST_TEST(params[i].getY() == 0.02);
      BOOST_TEST(params[i].getZ() == -0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    } else if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 102)) {
      BOOST_TEST(params[i].getX() == -0.01);
      BOOST_TEST(params[i].getY() == -0.02);
      BOOST_TEST(params[i].getZ() == 0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    } else if (params[i].getSymName() == fmt::format("MCH/HC{}/DE{}", 0, 103)) {
      BOOST_TEST(params[i].getX() == 0.01);
      BOOST_TEST(params[i].getY() == -0.02);
      BOOST_TEST(params[i].getZ() == -0.03);
      BOOST_TEST(params[i].getPsi() == 0.0);
      BOOST_TEST(params[i].getTheta() == 0.0);
      BOOST_TEST(params[i].getPhi() == 0.0);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
