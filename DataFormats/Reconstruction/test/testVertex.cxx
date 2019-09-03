// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Vertex class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/Vertex.h"
#include <TFile.h>
#include <array>

namespace o2
{
using myTS = o2::dataformats::TimeStampWithError<double, double>;
using myVtx = o2::dataformats::Vertex<myTS>;

// basic Vertex tests
BOOST_AUTO_TEST_CASE(Vertex)
{
  const Point3D<float> pos(0.1, 0.2, 3.0);
  const std::array<float, myVtx::kNCov> cov = {1e-4, -1e-9, 2e-4, -1e-9, 1e-9, 1e-4};
  int nc = 10;
  float chi2 = 5.5f;
  myVtx vtx(pos, cov, nc, chi2);
  myTS ts(1234.567, 0.99);
  vtx.setTimeStamp(ts);
  std::cout << vtx << std::endl;
  BOOST_CHECK_CLOSE(vtx.getX() + vtx.getY() + vtx.getZ(), pos.X() + pos.Y() + pos.Z(), 1e-5);
  for (int i = 0; i < myVtx::kNCov; i++) {
    BOOST_CHECK_CLOSE(vtx.getCov()[i], cov[i], 1e-5);
  }
  BOOST_CHECK_CLOSE(vtx.getChi2(), chi2, 1e-5);
  BOOST_CHECK(vtx.getNContributors() == nc);
  BOOST_CHECK(vtx.getTimeStamp() == ts);

  // test writing
  TFile flOut("tstVtx.root", "recreate");
  flOut.WriteObjectAny(&vtx, vtx.Class(), "vtx");
  flOut.Close();
  // test reading
  std::cout << "reading back written vertex" << std::endl;
  TFile flIn("tstVtx.root");
  auto vtr = static_cast<myVtx*>(gFile->GetObjectUnchecked("vtx"));
  flIn.Close();
  std::cout << *vtr << std::endl;
  BOOST_CHECK(vtx.getNContributors() == vtr->getNContributors());
}

} // namespace o2
