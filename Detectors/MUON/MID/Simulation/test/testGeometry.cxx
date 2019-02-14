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
#include "Geometry.h"
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

bool areEqual(std::array<double, 3>& p1, std::array<double, 3>& p2)
{
  for (int idim = 0; idim < 3; ++idim) {
    if (!areEqual(p1[idim], p2[idim])) {
      return false;
    }
  }
  return true;
}

int testOnePosition(const LocalPoint& localPt, const o2::mid::GeometryTransformer& geoTrans, const TGeoManager* geoManager)
{
  int deId = localPt.deId;
  double lpt[3] = { localPt.xPos, localPt.yPos, 0. };
  auto globPt = geoTrans.localToGlobal(deId, lpt[0], lpt[1]);
  std::array<double, 3> p1 = { globPt.x(), globPt.y(), globPt.z() };
  TGeoNavigator* navig = gGeoManager->GetCurrentNavigator();
  std::string volPath = gGeoManager->GetTopVolume()->GetName() + o2::mid::getRPCVolumePath(deId);
  navig->cd(volPath.c_str());
  std::array<double, 3> p2;
  navig->GetCurrentMatrix()->LocalToMaster(lpt, p2.data());
  if (!areEqual(p1, p2)) {
    std::cout << "got different positions for deId " << deId << " : got (" << p1[0] << ", " << p1[1] << ", " << p1[2] << ")  expected (" << p2[0] << ", " << p2[1] << ", " << p2[2] << ")\n";
    return 1;
  }
  return 0;
}

BOOST_AUTO_TEST_CASE(TestTransformations)
{
  o2::mid::GeometryTransformer geoTrans = o2::mid::createDefaultTransformer();
  createStandaloneGeometry("cave");

  auto positions = getPositions(1000);

  int notok = 0;

  for (auto& tp : positions) {
    notok += testOnePosition(tp, geoTrans, gGeoManager);
  }
  BOOST_TEST(notok == 0);
}

BOOST_AUTO_TEST_SUITE_END()
