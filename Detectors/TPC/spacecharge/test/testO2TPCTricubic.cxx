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

/// \file  testO2TPCTricubic.cxx
/// \brief this task tests the tricubic interpolation
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#define BOOST_TEST_MODULE Test TPC O2TPCTricubic class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSpaceCharge/TriCubic.h"
#include "TPCSpaceCharge/DataContainer3D.h"
#include "TPCSpaceCharge/PoissonSolverHelpers.h"
#include "TPCSpaceCharge/SpaceChargeHelpers.h"

namespace o2
{
namespace tpc
{

using DataT = double;
static constexpr DataT TOLERANCE = 0.15;     // relative tolerance
static constexpr DataT ABSTOLERANCE = 0.003; // absolute tolerance is taken at small values near 0
static constexpr unsigned short NR = 65;     // grid in r
static constexpr unsigned short NZ = 65;     // grid in z
static constexpr unsigned short NPHI = 90;   // grid in phi

BOOST_AUTO_TEST_CASE(PoissonSolver3D_test)
{
  const ParamSpaceCharge params{NR, NZ, NPHI};

  // define min range
  const DataT zmin = o2::tpc::GridProperties<DataT>::ZMIN;
  const DataT rmin = o2::tpc::GridProperties<DataT>::RMIN;
  const DataT phimin = o2::tpc::GridProperties<DataT>::PHIMIN;

  const DataT rSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingR(NR);
  const DataT zSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingZ(NZ);
  const DataT phiSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingPhi(NPHI);

  // create grid and datacontainer object
  o2::tpc::RegularGrid3D<DataT> grid3D(zmin, rmin, phimin, zSpacing, rSpacing, phiSpacing, params);
  o2::tpc::DataContainer3D<DataT> data3D(NZ, NR, NPHI);

  // function to approximate
  o2::tpc::AnalyticalFields<DataT> field;

  // fill the DataContainer3D with some values
  for (int iz = 0; iz < NZ; ++iz) {
    for (int ir = 0; ir < NR; ++ir) {
      for (int iphi = 0; iphi < NPHI; ++iphi) {
        const DataT z = zSpacing * iz + zmin;
        const DataT r = rSpacing * ir + rmin;
        const DataT phi = phiSpacing * iphi + phimin;
        data3D(iz, ir, iphi) = field.evalPotential(z, r, phi);
      }
    }
  }

  // create tricubic interpolator
  o2::tpc::TriCubicInterpolator<DataT> interpolator(data3D, grid3D);

  const float nFacLoop = 1.4;
  const int nrPointsLoop = NR * nFacLoop;
  const int nzPointsLoop = NZ * nFacLoop;
  const int nphiPointsLoop = NPHI * nFacLoop;
  const DataT rSpacingLoop = GridProperties<DataT>::getGridSpacingR(nrPointsLoop);
  const DataT zSpacingLoop = GridProperties<DataT>::getGridSpacingZ(nzPointsLoop);
  const DataT phiSpacingLoop = GridProperties<DataT>::getGridSpacingZ(nphiPointsLoop);

  for (int iR = -2; iR < nrPointsLoop + 2; ++iR) {
    const DataT r = rmin + iR * rSpacingLoop;
    for (int iZ = 0; iZ < nzPointsLoop; ++iZ) {
      const DataT z = zmin + iZ * zSpacingLoop;
      for (int iPhi = -2; iPhi < nphiPointsLoop + 2; ++iPhi) {
        DataT phi = phimin + iPhi * phiSpacingLoop;
        const DataT interpolatedSparse = interpolator(z, r, phi);
        const DataT trueValue = field.evalPotential(z, r, phi);

        // use larger tolerances at the edges of the grid
        const int facTol = ((iR < nFacLoop) || (iZ < nFacLoop) || (iR >= nrPointsLoop - 1 - nFacLoop) || (iZ >= nzPointsLoop - 1 - nFacLoop)) ? 10 : 1;
        if (std::abs(trueValue) < 0.1) {
          BOOST_CHECK_SMALL(trueValue - interpolatedSparse, facTol * ABSTOLERANCE);
        } else {
          BOOST_CHECK_CLOSE(interpolatedSparse, trueValue, facTol * TOLERANCE);
        }
      }
    }
  }
}

} // namespace tpc
} // namespace o2
