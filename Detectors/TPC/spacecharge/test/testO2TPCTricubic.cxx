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
#include <algorithm>
#include <cmath>

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
  const DataT phiSpacingLoop = GridProperties<DataT>::getGridSpacingPhi(nphiPointsLoop);

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

BOOST_AUTO_TEST_CASE(TriCubic_vertices_test)
{
  // the tricubic interpolation has to reproduce the values at the vertices of the grid: checks the indexing in the inner volume and at the boundaries of the grid (extrapolation)
  const ParamSpaceCharge params{NR, NZ, NPHI};
  const DataT rSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingR(NR);
  const DataT zSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingZ(NZ);
  const DataT phiSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingPhi(NPHI);
  const o2::tpc::RegularGrid3D<DataT> grid3D(GridProperties<DataT>::ZMIN, GridProperties<DataT>::RMIN, GridProperties<DataT>::PHIMIN, zSpacing, rSpacing, phiSpacing, params);
  o2::tpc::DataContainer3D<DataT> data3D(NZ, NR, NPHI);

  // fill with values without any symmetry
  for (int iz = 0; iz < NZ; ++iz) {
    for (int ir = 0; ir < NR; ++ir) {
      for (int iphi = 0; iphi < NPHI; ++iphi) {
        data3D(iz, ir, iphi) = 1 + iz + 1000 * ir + 1e6 * iphi + std::sin(iz * ir + iphi);
      }
    }
  }

  for (const bool extrapolate : {true, false}) {
    o2::tpc::TriCubicInterpolator<DataT> interpolator(data3D, grid3D);
    interpolator.setExtrapolateValues(extrapolate);
    for (int iz = 0; iz < NZ; ++iz) {
      for (int ir = 0; ir < NR; ++ir) {
        for (int iphi = 0; iphi < NPHI; ++iphi) {
          const DataT interpolated = interpolator(grid3D.getZVertex(iz), grid3D.getRVertex(ir), grid3D.getPhiVertex(iphi));
          BOOST_CHECK_CLOSE(interpolated, data3D(iz, ir, iphi), 1e-8);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TriCubic_threeComponents_test)
{
  // the combined interpolation of three data containers has to give the same results as three separate interpolations
  const ParamSpaceCharge params{NR, NZ, NPHI};
  const DataT rSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingR(NR);
  const DataT zSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingZ(NZ);
  const DataT phiSpacing = o2::tpc::GridProperties<DataT>::getGridSpacingPhi(NPHI);
  const o2::tpc::RegularGrid3D<DataT> grid3D(GridProperties<DataT>::ZMIN, GridProperties<DataT>::RMIN, GridProperties<DataT>::PHIMIN, zSpacing, rSpacing, phiSpacing, params);
  o2::tpc::DataContainer3D<DataT> dataR(NZ, NR, NPHI);
  o2::tpc::DataContainer3D<DataT> dataZ(NZ, NR, NPHI);
  o2::tpc::DataContainer3D<DataT> dataPhi(NZ, NR, NPHI);
  o2::tpc::AnalyticalFields<DataT> field;
  for (int iz = 0; iz < NZ; ++iz) {
    for (int ir = 0; ir < NR; ++ir) {
      for (int iphi = 0; iphi < NPHI; ++iphi) {
        const DataT z = grid3D.getZVertex(iz);
        const DataT r = grid3D.getRVertex(ir);
        const DataT phi = grid3D.getPhiVertex(iphi);
        dataR(iz, ir, iphi) = field.evalFieldR(z, r, phi);
        dataZ(iz, ir, iphi) = field.evalFieldZ(z, r, phi);
        dataPhi(iz, ir, iphi) = field.evalFieldPhi(z, r, phi);
      }
    }
  }

  // the combined interpolation uses the same operations as the separate interpolations, but the compiler may contract
  // multiplications and additions (FMA) differently in the two loops: compare with a tolerance at the rounding level
  const auto maxAbs = [](const auto& data) { DataT maxVal = 0; for (const auto val : data.getData()) { maxVal = std::max(maxVal, std::abs(val)); } return maxVal; };
  const DataT tolR = 1e-12 * maxAbs(dataR);
  const DataT tolZ = 1e-12 * maxAbs(dataZ);
  const DataT tolPhi = 1e-12 * maxAbs(dataPhi);

  for (const bool extrapolate : {true, false}) {
    o2::tpc::TriCubicInterpolator<DataT> interpolatorR(dataR, grid3D);
    o2::tpc::TriCubicInterpolator<DataT> interpolatorZ(dataZ, grid3D);
    o2::tpc::TriCubicInterpolator<DataT> interpolatorPhi(dataPhi, grid3D);
    interpolatorR.setExtrapolateValues(extrapolate);
    interpolatorZ.setExtrapolateValues(extrapolate);
    interpolatorPhi.setExtrapolateValues(extrapolate);

    // query points inside and outside of the grid
    const int nPoints = 40;
    for (int iR = -2; iR < nPoints + 2; ++iR) {
      const DataT r = GridProperties<DataT>::RMIN + iR * (GridProperties<DataT>::RMAX - GridProperties<DataT>::RMIN) / nPoints;
      for (int iZ = -2; iZ < nPoints + 2; ++iZ) {
        const DataT z = GridProperties<DataT>::ZMIN + iZ * (GridProperties<DataT>::ZMAX - GridProperties<DataT>::ZMIN) / nPoints;
        for (int iPhi = -2; iPhi < nPoints + 2; ++iPhi) {
          const DataT phi = iPhi * GridProperties<DataT>::PHIMAX / nPoints;
          DataT valR = 0;
          DataT valZ = 0;
          DataT valPhi = 0;
          interpolatorR.interpolate(z, r, phi, interpolatorZ, interpolatorPhi, valR, valZ, valPhi);
          BOOST_CHECK_SMALL(valR - interpolatorR(z, r, phi), tolR);
          BOOST_CHECK_SMALL(valZ - interpolatorZ(z, r, phi), tolZ);
          BOOST_CHECK_SMALL(valPhi - interpolatorPhi(z, r, phi), tolPhi);
        }
      }
    }
  }
}

/// \return number of query points (inside and outside of the grid) for which the combined interpolation of three interpolators differs from the separate interpolations
int getNDifferentCombined(const TriCubicInterpolator<DataT>& interp0, const TriCubicInterpolator<DataT>& interp1, const TriCubicInterpolator<DataT>& interp2, const DataT tolerance)
{
  int nDiff = 0;
  const int nPoints = 20;
  for (int iR = -2; iR < nPoints + 2; ++iR) {
    const DataT r = GridProperties<DataT>::RMIN + iR * (GridProperties<DataT>::RMAX - GridProperties<DataT>::RMIN) / nPoints;
    for (int iZ = -2; iZ < nPoints + 2; ++iZ) {
      const DataT z = GridProperties<DataT>::ZMIN + iZ * (GridProperties<DataT>::ZMAX - GridProperties<DataT>::ZMIN) / nPoints;
      for (int iPhi = -2; iPhi < nPoints + 2; ++iPhi) {
        const DataT phi = iPhi * GridProperties<DataT>::PHIMAX / nPoints;
        DataT val0 = 0;
        DataT val1 = 0;
        DataT val2 = 0;
        interp0.interpolate(z, r, phi, interp1, interp2, val0, val1, val2);
        nDiff += (std::abs(val0 - interp0(z, r, phi)) > tolerance) || (std::abs(val1 - interp1(z, r, phi)) > tolerance) || (std::abs(val2 - interp2(z, r, phi)) > tolerance);
      }
    }
  }
  return nDiff;
}

BOOST_AUTO_TEST_CASE(TriCubic_threeComponentsDifferent_test)
{
  // the combined interpolation of three data containers with different extrapolation settings, grids or data layout has to give the same results as three separate interpolations
  const auto fillData = [](DataContainer3D<DataT>& data, const RegularGrid3D<DataT>& grid, const int component) {
    AnalyticalFields<DataT> field;
    for (size_t iz = 0; iz < data.getNZ(); ++iz) {
      for (size_t ir = 0; ir < data.getNR(); ++ir) {
        for (size_t iphi = 0; iphi < data.getNPhi(); ++iphi) {
          const DataT z = grid.getZVertex(iz);
          const DataT r = grid.getRVertex(ir);
          const DataT phi = grid.getPhiVertex(iphi);
          data(iz, ir, iphi) = (component == 0) ? field.evalFieldR(z, r, phi) : ((component == 1) ? field.evalFieldZ(z, r, phi) : field.evalFieldPhi(z, r, phi));
        }
      }
    }
  };
  const DataT rSpacing = GridProperties<DataT>::getGridSpacingR(NR);
  const DataT zSpacing = GridProperties<DataT>::getGridSpacingZ(NZ);
  const DataT phiSpacing = GridProperties<DataT>::getGridSpacingPhi(NPHI);
  const RegularGrid3D<DataT> grid3D(GridProperties<DataT>::ZMIN, GridProperties<DataT>::RMIN, GridProperties<DataT>::PHIMIN, zSpacing, rSpacing, phiSpacing, ParamSpaceCharge{NR, NZ, NPHI});
  DataContainer3D<DataT> data0(NZ, NR, NPHI);
  DataContainer3D<DataT> data1(NZ, NR, NPHI);
  DataContainer3D<DataT> data2(NZ, NR, NPHI);
  fillData(data0, grid3D, 0);
  fillData(data1, grid3D, 1);
  fillData(data2, grid3D, 2);
  const auto maxAbs = [](const DataContainer3D<DataT>& data) { DataT maxVal = 0; for (const auto val : data.getData()) { maxVal = std::max(maxVal, std::abs(val)); } return maxVal; };
  const DataT tolerance = 1e-12 * std::max({maxAbs(data0), maxAbs(data1), maxAbs(data2)});

  TriCubicInterpolator<DataT> interp0(data0, grid3D);
  TriCubicInterpolator<DataT> interp2(data2, grid3D);

  // different extrapolation type
  {
    TriCubicInterpolator<DataT> interp1(data1, grid3D);
    interp1.setExtrapolationType(TriCubicInterpolator<DataT>::ExtrapolationType::Linear);
    BOOST_CHECK_EQUAL(getNDifferentCombined(interp0, interp1, interp2, tolerance), 0);
  }

  // no extrapolation of query points outside of the grid
  {
    TriCubicInterpolator<DataT> interp1(data1, grid3D);
    interp1.setExtrapolateValues(false);
    BOOST_CHECK_EQUAL(getNDifferentCombined(interp0, interp1, interp2, tolerance), 0);
  }

  // different grid with the same number of vertices
  {
    const RegularGrid3D<DataT> gridShifted(GridProperties<DataT>::ZMIN, GridProperties<DataT>::RMIN + 5, GridProperties<DataT>::PHIMIN, zSpacing, rSpacing, phiSpacing, ParamSpaceCharge{NR, NZ, NPHI});
    DataContainer3D<DataT> data1Shifted(NZ, NR, NPHI);
    fillData(data1Shifted, gridShifted, 1);
    TriCubicInterpolator<DataT> interp1(data1Shifted, gridShifted);
    BOOST_CHECK_EQUAL(getNDifferentCombined(interp0, interp1, interp2, tolerance), 0);
  }

  // different number of vertices in z and phi with the same total number of vertices
  {
    const RegularGrid3D<DataT> gridOther(GridProperties<DataT>::ZMIN, GridProperties<DataT>::RMIN, GridProperties<DataT>::PHIMIN, GridProperties<DataT>::getGridSpacingZ(NPHI), rSpacing, GridProperties<DataT>::getGridSpacingPhi(NZ), ParamSpaceCharge{NR, NPHI, NZ});
    DataContainer3D<DataT> data1Other(NPHI, NR, NZ);
    fillData(data1Other, gridOther, 1);
    TriCubicInterpolator<DataT> interp1(data1Other, gridOther);
    BOOST_CHECK_EQUAL(data1Other.getNDataPoints(), data0.getNDataPoints());
    BOOST_CHECK_EQUAL(getNDifferentCombined(interp0, interp1, interp2, tolerance), 0);
  }
}

} // namespace tpc
} // namespace o2
