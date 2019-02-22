// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Diego Stocco

#define BOOST_TEST_MODULE Test MID simulation
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

// #include "boost/format.hpp"
// #include <boost/test/data/monomorphic.hpp>
// #include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iostream>
#include "MathUtils/Cartesian3D.h"
#include "TGeoManager.h"
#include "MIDSimulation/Geometry.h"
#include "MIDBase/GeometryTransformer.h"

BOOST_AUTO_TEST_SUITE(o2_mid_simulation)

void createStandaloneGeometry(const char* name)
{
  /// Create standalone geometry for geometry transformation tests
  if (gGeoManager && gGeoManager->GetTopVolume()) {
    std::cerr << "Can only call this function with an empty geometry, i.e. gGeoManager==nullptr "
              << " or gGeoManager->GetTopVolume()==nullptr\n";
  }
  TGeoManager* gm = new TGeoManager("MID-ONLY", "ALICE MID Standalone Geometry");
  TGeoMaterial* mat = new TGeoMaterial("Vacuum", 0, 0, 0);
  TGeoMedium* med = new TGeoMedium("Vacuum", 1, mat);
  TGeoVolume* top = gGeoManager->MakeBox(name, med, 2000.0, 2000.0, 3000.0);
  gm->SetTopVolume(top);
  o2::mid::createGeometry(*top);
}

struct LocalPoint {
  LocalPoint(int ide, double x, double y) : deId(ide), xPos(x), yPos(y) {}
  int deId;
  double xPos;
  double yPos;
};

std::vector<LocalPoint> getPositions(int npoints)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> distX(-127.5, 127.5);
  std::uniform_real_distribution<double> distY(-40., 40.);
  std::uniform_int_distribution<int> distDE(0, 71);

  std::vector<LocalPoint> localPoints;

  for (int ipt = 0; ipt < npoints; ++ipt) {
    localPoints.emplace_back(distDE(mt), distX(mt), distY(mt));
  }
  return localPoints;
}

bool areEqual(double a, double b)
{
  return std::fabs(b - a) < 1E-4; // 1E-4 cm = 1 micron
}

bool areEqual(const Point3D<double>& p1, const Point3D<double>& p2)
{
  if (!areEqual(p1.x(), p2.x()))
    return false;
  if (!areEqual(p1.y(), p2.y()))
    return false;
  if (!areEqual(p1.z(), p2.z()))
    return false;
  return true;
}

int testOnePosition(const LocalPoint& localPt, const o2::mid::GeometryTransformer& geoTrans, const o2::mid::GeometryTransformer& geoTransFromManager)
{
  int deId = localPt.deId;
  auto p1 = geoTrans.localToGlobal(deId, localPt.xPos, localPt.yPos);
  auto p2 = geoTransFromManager.localToGlobal(deId, localPt.xPos, localPt.yPos);
  if (!areEqual(p1, p2)) {
    std::cout << "got different positions for deId " << deId << " : got (" << p1.x() << ", " << p1.y() << ", " << p1.z() << ")  expected (" << p2.x() << ", " << p2.y() << ", " << p2.z() << ")\n";
    return 1;
  }
  return 0;
}

BOOST_AUTO_TEST_CASE(TestTransformations)
{
  o2::mid::GeometryTransformer geoTrans = o2::mid::createDefaultTransformer();
  createStandaloneGeometry("cave");

  o2::mid::GeometryTransformer geoTransFromManager = o2::mid::createTransformationFromManager(gGeoManager);

  auto positions = getPositions(1000);

  int notok = 0;

  for (auto& tp : positions) {
    notok += testOnePosition(tp, geoTrans, geoTransFromManager);
  }
  BOOST_TEST(notok == 0);
}

BOOST_AUTO_TEST_SUITE_END()
