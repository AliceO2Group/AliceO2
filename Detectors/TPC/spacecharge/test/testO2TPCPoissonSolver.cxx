// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  testPoissonSolver.cxx
/// \brief this task tests the poisson solver
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#define BOOST_TEST_MODULE Test TPC O2TPCSpaceCharge3DCalc class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSpaceCharge/PoissonSolver.h"
#include "TPCSpaceCharge/SpaceChargeHelpers.h"

namespace o2
{
namespace tpc
{

using DataT = double;                       // using float actually takes alot longer than double (doesnt converge when using float)
static constexpr DataT TOLERANCE = 3;       // relative tolerance for 3D (maximum large error is at phi=90 since there the potential is 0!)
static constexpr DataT TOLERANCE2D = 8.5;   // relative tolerance for 2D TODO check why the difference between numerical and analyticial is larger than for 3D!
static constexpr DataT ABSTOLERANCE = 0.01; // absolute tolerance is taken at small values near 0
static constexpr int NR = 65;               // grid in r
static constexpr int NZ = 65;               // grid in z
static constexpr int NR2D = 129;            // grid in r
static constexpr int NZ2D = 129;            // grid in z
static constexpr int NPHI = 180;            // grid in phi

/// Get phi vertex position for index in phi direction
/// \param indexPhi index in phi direction
template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT getPhiVertex(const size_t indexPhi, const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi>& grid)
{
  return grid.getZVertex(indexPhi);
}

/// Get r vertex position for index in r direction
/// \param indexR index in r direction
template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT getRVertex(const size_t indexR, const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi>& grid)
{
  return grid.getYVertex(indexR);
}

/// Get z vertex position for index in z direction
/// \param indexZ index in z direction
template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT getZVertex(const size_t indexZ, const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi>& grid)
{
  return grid.getXVertex(indexZ);
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void setChargeDensityFromFormula(const AnalyticalFields<DataT>& formulas, const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi>& grid, o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>& density)
{
  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    const DataT phi = getPhiVertex<DataT, Nz, Nr, Nphi>(iPhi, grid);
    for (size_t iR = 0; iR < Nr; ++iR) {
      const DataT radius = getRVertex<DataT, Nz, Nr, Nphi>(iR, grid);
      for (size_t iZ = 0; iZ < Nz; ++iZ) {
        const DataT z = getZVertex<DataT, Nz, Nr, Nphi>(iZ, grid);
        density(iZ, iR, iPhi) = formulas.evalDensity(z, radius, phi);
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void setPotentialFromFormula(const AnalyticalFields<DataT>& formulas, const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi>& grid, o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>& potential)
{
  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    const DataT phi = getPhiVertex<DataT, Nz, Nr, Nphi>(iPhi, grid);
    for (size_t iR = 0; iR < Nr; ++iR) {
      const DataT radius = getRVertex<DataT, Nz, Nr, Nphi>(iR, grid);
      for (size_t iZ = 0; iZ < Nz; ++iZ) {
        const DataT z = getZVertex<DataT, Nz, Nr, Nphi>(iZ, grid);
        potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void setPotentialBoundaryFromFormula(const AnalyticalFields<DataT>& formulas, const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi>& grid, o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>& potential)
{
  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    const DataT phi = getPhiVertex<DataT, Nz, Nr, Nphi>(iPhi, grid);
    for (size_t iZ = 0; iZ < Nz; ++iZ) {
      const DataT z = getZVertex<DataT, Nz, Nr, Nphi>(iZ, grid);
      const size_t iR = 0;
      const DataT radius = getRVertex<DataT, Nz, Nr, Nphi>(iR, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    const DataT phi = getPhiVertex<DataT, Nz, Nr, Nphi>(iPhi, grid);
    for (size_t iZ = 0; iZ < Nz; ++iZ) {
      const DataT z = getZVertex<DataT, Nz, Nr, Nphi>(iZ, grid);
      const size_t iR = Nr - 1;
      const DataT radius = getRVertex<DataT, Nz, Nr, Nphi>(iR, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    const DataT phi = getPhiVertex<DataT, Nz, Nr, Nphi>(iPhi, grid);
    for (size_t iR = 0; iR < Nr; ++iR) {
      const DataT radius = getRVertex<DataT, Nz, Nr, Nphi>(iR, grid);
      const size_t iZ = 0;
      const DataT z = getZVertex<DataT, Nz, Nr, Nphi>(iZ, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    const DataT phi = getPhiVertex<DataT, Nz, Nr, Nphi>(iPhi, grid);
    for (size_t iR = 0; iR < Nr; ++iR) {
      const DataT radius = getRVertex<DataT, Nz, Nr, Nphi>(iR, grid);
      const size_t iZ = Nz - 1;
      const DataT z = getZVertex<DataT, Nz, Nr, Nphi>(iZ, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void testAlmostEqualArray(o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>& analytical, o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>& numerical)
{
  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    for (size_t iR = 0; iR < Nr; ++iR) {
      for (size_t iZ = 0; iZ < Nz; ++iZ) {
        if (std::fabs(analytical(iZ, iR, iPhi)) < ABSTOLERANCE) {
          BOOST_CHECK_SMALL(numerical(iZ, iR, iPhi) - analytical(iZ, iR, iPhi), ABSTOLERANCE);
        } else {
          BOOST_CHECK_CLOSE(numerical(iZ, iR, iPhi), analytical(iZ, iR, iPhi), TOLERANCE);
        }
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void testAlmostEqualArray2D(o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>& analytical, o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>& numerical)
{
  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    for (size_t iR = 0; iR < Nr; ++iR) {
      for (size_t iZ = 0; iZ < Nz; ++iZ) {
        if (std::fabs(analytical(iZ, iR, iPhi)) < ABSTOLERANCE) {
          BOOST_CHECK_SMALL(numerical(iZ, iR, iPhi) - analytical(iZ, iR, iPhi), ABSTOLERANCE);
        } else {
          BOOST_CHECK_CLOSE(numerical(iZ, iR, iPhi), analytical(iZ, iR, iPhi), TOLERANCE2D);
        }
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void poissonSolver3D()
{
  using GridProp = GridProperties<DataT, Nr, Nz, Nphi>;
  const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi> grid3D{GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, GridProp::GRIDSPACINGZ, GridProp::GRIDSPACINGR, GridProp::GRIDSPACINGPHI};

  using DataContainer = o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>;
  DataContainer potentialNumerical{};
  DataContainer potentialAnalytical{};
  DataContainer charge{};

  const o2::tpc::AnalyticalFields<DataT> analyticalFields;
  // set the boudnary and charge for numerical poisson solver
  setChargeDensityFromFormula<DataT, Nz, Nr, Nphi>(analyticalFields, grid3D, charge);
  setPotentialBoundaryFromFormula<DataT, Nz, Nr, Nphi>(analyticalFields, grid3D, potentialNumerical);

  // set analytical potential
  setPotentialFromFormula<DataT, Nz, Nr, Nphi>(analyticalFields, grid3D, potentialAnalytical);

  //calculate numerical potential
  PoissonSolver<DataT, Nz, Nr, Nphi> poissonSolver(grid3D);
  const int symmetry = 0;
  poissonSolver.poissonSolver3D(potentialNumerical, charge, symmetry);

  // compare numerical with analytical solution of the potential
  testAlmostEqualArray<DataT, Nz, Nr, Nphi>(potentialAnalytical, potentialNumerical);
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void poissonSolver2D()
{
  using GridProp = GridProperties<DataT, Nr, Nz, Nphi>;
  const o2::tpc::RegularGrid3D<DataT, Nz, Nr, Nphi> grid3D{GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, GridProp::GRIDSPACINGZ, GridProp::GRIDSPACINGR, GridProp::GRIDSPACINGPHI};

  using DataContainer = o2::tpc::DataContainer3D<DataT, Nz, Nr, Nphi>;
  DataContainer potentialNumerical{};
  DataContainer potentialAnalytical{};
  DataContainer charge{};

  // set the boudnary and charge for numerical poisson solver
  const o2::tpc::AnalyticalFields<DataT> analyticalFields;
  setChargeDensityFromFormula<DataT, Nz, Nr, Nphi>(analyticalFields, grid3D, charge);
  setPotentialBoundaryFromFormula<DataT, Nz, Nr, Nphi>(analyticalFields, grid3D, potentialNumerical);

  // set analytical potential
  setPotentialFromFormula<DataT, Nz, Nr, Nphi>(analyticalFields, grid3D, potentialAnalytical);

  //calculate numerical potential
  PoissonSolver<DataT, Nz, Nr, Nphi> poissonSolver(grid3D);
  poissonSolver.poissonSolver2D(potentialNumerical, charge);

  // compare numerical with analytical solution of the potential
  testAlmostEqualArray2D<DataT, Nz, Nr, Nphi>(potentialAnalytical, potentialNumerical);
}

BOOST_AUTO_TEST_CASE(PoissonSolver3D_test)
{
  o2::tpc::MGParameters::isFull3D = true; //3D
  poissonSolver3D<DataT, NZ, NR, NPHI>();
}

BOOST_AUTO_TEST_CASE(PoissonSolver3D2D_test)
{
  o2::tpc::MGParameters::isFull3D = false; // 3D2D
  poissonSolver3D<DataT, NZ, NR, NPHI>();
}

BOOST_AUTO_TEST_CASE(PoissonSolver2D_test)
{
  const int Nphi = 1;
  poissonSolver2D<DataT, NZ2D, NR2D, Nphi>();
}

} // namespace tpc
} // namespace o2
