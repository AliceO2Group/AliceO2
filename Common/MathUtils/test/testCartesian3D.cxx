// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Cartesian3D
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <TGeoMatrix.h>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "MathUtils/Cartesian3D.h"

using namespace o2;

BOOST_AUTO_TEST_CASE(Cartesian3D_test)
{
  // we create Transform3D by conversion from TGeoHMatrix
  TGeoRotation rotg("r", 10., 20., 30.);
  TGeoTranslation trag("g", 100., 200., 300.);
  TGeoHMatrix hmat = trag;
  hmat *= rotg;

  Transform3D tr(hmat);
  Point3D<double> pd(10., 20., 30.);
  Point3D<float> pf(10.f, 20.f, 30.f);
  //
  // local to master
  auto pdt = tr(pd); // operator form
  Point3D<float> pft;
  tr.LocalToMaster(pf, pft); // TGeoHMatrix form

  std::cout << "Create Transform3D " << std::endl
            << tr << std::endl
            << "from" << std::endl;
  hmat.Print();

  std::cout << " Transforming " << pd << " to master" << std::endl;
  // compare difference between float and double vector transform
  std::cout << "Float:  " << pft << std::endl;
  std::cout << "Double: " << pdt << std::endl;
  std::cout << "Diff:   " << pdt.X() - pft.X() << "," << pdt.Y() - pft.Y() << "," << pdt.Z() - pft.Z() << std::endl;

  const double toler = 1e-4;
  BOOST_CHECK(std::abs(pdt.X() - pft.X()) < toler);
  BOOST_CHECK(std::abs(pdt.Y() - pft.Y()) < toler);
  BOOST_CHECK(std::abs(pdt.Z() - pft.Z()) < toler);

  // inverse transform
  auto pfti = tr ^ (pft); // operator form
  Point3D<double> pdti;
  tr.MasterToLocal(pdt, pdti); // TGeoHMatrix form

  std::cout << " Transforming back to local" << std::endl;
  std::cout << "Float:  " << pfti << std::endl;
  std::cout << "Double: " << pdti << std::endl;
  std::cout << "Diff:   " << pd.X() - pfti.X() << ", " << pd.Y() - pfti.Y() << ", " << pd.Z() - pfti.Z() << std::endl;

  BOOST_CHECK(std::abs(pd.X() - pfti.X()) < toler);
  BOOST_CHECK(std::abs(pd.Y() - pfti.Y()) < toler);
  BOOST_CHECK(std::abs(pd.Z() - pfti.Z()) < toler);

  BOOST_CHECK(std::abs(pdti.X() - pfti.X()) < toler);
  BOOST_CHECK(std::abs(pdti.Y() - pfti.Y()) < toler);
  BOOST_CHECK(std::abs(pdti.Z() - pfti.Z()) < toler);
}
