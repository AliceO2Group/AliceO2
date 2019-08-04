// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test BasicHits class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "Math/GenVector/Transform3D.h"
#include "SimulationDataFormat/BaseHits.h"
#include "TFile.h"

namespace o2
{

BOOST_AUTO_TEST_CASE(BasicXYZHit)
{
  using HitType = BasicXYZEHit<float>;
  HitType hit(1., 2., 3., 0.01, -1.1, -1, 1);

  BOOST_CHECK_CLOSE(hit.GetX(), 1., 1E-4);
  BOOST_CHECK_CLOSE(hit.GetY(), 2., 1E-4);
  BOOST_CHECK_CLOSE(hit.GetZ(), 3., 1E-4);
  BOOST_CHECK_CLOSE(hit.GetEnergyLoss(), -1.1, 1E-4);
  BOOST_CHECK_CLOSE(hit.GetTime(), 0.01, 1E-4);

  hit.SetX(0.);
  BOOST_CHECK_CLOSE(hit.GetX(), 0., 1E-4);

  // check coordinate transformation of the hit coordinate
  // check that is works with float + double
  // note that ROOT transformations are always double valued
  using ROOT::Math::Transform3D;
  Transform3D idtransf; // defaults to identity transformation

  auto transformed = idtransf(hit.GetPos());
  BOOST_CHECK_CLOSE(transformed.Y(), hit.GetY(), 1E-4);
}

BOOST_AUTO_TEST_CASE(BasicXYZHit_ROOTIO)
{
  using HitType = BasicXYZEHit<float>;
  HitType hit(1., 2., 3., 0.01, -1.1, -1, 1);

  // try writing hit to a TBuffer
  {
    TFile fout("HitsIO.root", "RECREATE");
    fout.WriteObject(&hit, "TestObject");
    fout.Close();
  }

  {
    TFile fin("HitsIO.root");
    HitType* obj = nullptr;
    fin.GetObject("TestObject", obj);

    BOOST_CHECK(obj != nullptr);
    fin.Close();
  }

  // same for double valued hits
  using HitTypeD = BasicXYZEHit<double, double>;
  HitTypeD hitD(1., 2., 3., 0.01, -1.1, -1, 1);

  // try writing hit to a TBuffer
  {
    TFile fout("HitsIO.root", "RECREATE");
    fout.WriteObject(&hitD, "TestObject");
    fout.Close();
  }

  {
    TFile fin("HitsIO.root");
    HitTypeD* obj = nullptr;
    fin.GetObject("TestObject", obj);

    BOOST_CHECK(obj != nullptr);
    fin.Close();
  }
}

} // namespace o2
