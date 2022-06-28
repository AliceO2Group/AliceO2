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
#include "TPCSpaceCharge/PoissonSolverHelpers.h"
#include "TPCSpaceCharge/DataContainer3D.h"

namespace o2
{
namespace tpc
{

using DataT = double;                       // using float actually takes alot longer than double (doesnt converge when using float)
static constexpr DataT TOLERANCE = 3;       // relative tolerance for 3D (maximum large error is at phi=90 since there the potential is 0!)
static constexpr DataT TOLERANCE2D = 8.5;   // relative tolerance for 2D TODO check why the difference between numerical and analyticial is larger than for 3D!
static constexpr DataT ABSTOLERANCE = 0.01; // absolute tolerance is taken at small values near 0
static constexpr unsigned short NR = 65;    // grid in r
static constexpr unsigned short NZ = 65;    // grid in z
static constexpr unsigned short NPHI = 180; // grid in phi
static constexpr unsigned short NR2D = 129; // grid in r
static constexpr unsigned short NZ2D = 129; // grid in z
static constexpr unsigned short NPHI2D = 1; // grid in phi

/// Get phi vertex position for index in phi direction
/// \param indexPhi index in phi direction
template <typename DataT>
DataT getPhiVertex(const size_t indexPhi, const o2::tpc::RegularGrid3D<DataT>& grid)
{
  return grid.getPhiVertex(indexPhi);
}

/// Get r vertex position for index in r direction
/// \param indexR index in r direction
template <typename DataT>
DataT getRVertex(const size_t indexR, const o2::tpc::RegularGrid3D<DataT>& grid)
{
  return grid.getRVertex(indexR);
}

/// Get z vertex position for index in z direction
/// \param indexZ index in z direction
template <typename DataT>
DataT getZVertex(const size_t indexZ, const o2::tpc::RegularGrid3D<DataT>& grid)
{
  return grid.getZVertex(indexZ);
}

template <typename DataT>
void setChargeDensityFromFormula(const AnalyticalFields<DataT>& formulas, const o2::tpc::RegularGrid3D<DataT>& grid, o2::tpc::DataContainer3D<DataT>& density)
{
  for (size_t iPhi = 0; iPhi < density.getNPhi(); ++iPhi) {
    const DataT phi = getPhiVertex<DataT>(iPhi, grid);
    for (size_t iR = 0; iR < density.getNR(); ++iR) {
      const DataT radius = getRVertex<DataT>(iR, grid);
      for (size_t iZ = 0; iZ < density.getNZ(); ++iZ) {
        const DataT z = getZVertex<DataT>(iZ, grid);
        density(iZ, iR, iPhi) = formulas.evalDensity(z, radius, phi);
      }
    }
  }
}

template <typename DataT>
void setPotentialFromFormula(const AnalyticalFields<DataT>& formulas, const o2::tpc::RegularGrid3D<DataT>& grid, o2::tpc::DataContainer3D<DataT>& potential)
{
  for (size_t iPhi = 0; iPhi < potential.getNPhi(); ++iPhi) {
    const DataT phi = getPhiVertex<DataT>(iPhi, grid);
    for (size_t iR = 0; iR < potential.getNR(); ++iR) {
      const DataT radius = getRVertex<DataT>(iR, grid);
      for (size_t iZ = 0; iZ < potential.getNZ(); ++iZ) {
        const DataT z = getZVertex<DataT>(iZ, grid);
        potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
      }
    }
  }
}

template <typename DataT>
void setPotentialBoundaryFromFormula(const AnalyticalFields<DataT>& formulas, const o2::tpc::RegularGrid3D<DataT>& grid, o2::tpc::DataContainer3D<DataT>& potential)
{
  for (size_t iPhi = 0; iPhi < potential.getNPhi(); ++iPhi) {
    const DataT phi = getPhiVertex<DataT>(iPhi, grid);
    for (size_t iZ = 0; iZ < potential.getNZ(); ++iZ) {
      const DataT z = getZVertex<DataT>(iZ, grid);
      const size_t iR = 0;
      const DataT radius = getRVertex<DataT>(iR, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < potential.getNPhi(); ++iPhi) {
    const DataT phi = getPhiVertex<DataT>(iPhi, grid);
    for (size_t iZ = 0; iZ < potential.getNZ(); ++iZ) {
      const DataT z = getZVertex<DataT>(iZ, grid);
      const size_t iR = potential.getNR() - 1;
      const DataT radius = getRVertex<DataT>(iR, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < potential.getNPhi(); ++iPhi) {
    const DataT phi = getPhiVertex<DataT>(iPhi, grid);
    for (size_t iR = 0; iR < potential.getNR(); ++iR) {
      const DataT radius = getRVertex<DataT>(iR, grid);
      const size_t iZ = 0;
      const DataT z = getZVertex<DataT>(iZ, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < potential.getNPhi(); ++iPhi) {
    const DataT phi = getPhiVertex<DataT>(iPhi, grid);
    for (size_t iR = 0; iR < potential.getNR(); ++iR) {
      const DataT radius = getRVertex<DataT>(iR, grid);
      const size_t iZ = potential.getNZ() - 1;
      const DataT z = getZVertex<DataT>(iZ, grid);
      potential(iZ, iR, iPhi) = formulas.evalPotential(z, radius, phi);
    }
  }
}

template <typename DataT>
void testAlmostEqualArray(o2::tpc::DataContainer3D<DataT>& analytical, o2::tpc::DataContainer3D<DataT>& numerical)
{
  for (size_t iPhi = 0; iPhi < numerical.getNPhi(); ++iPhi) {
    for (size_t iR = 0; iR < numerical.getNR(); ++iR) {
      for (size_t iZ = 0; iZ < numerical.getNZ(); ++iZ) {
        if (std::fabs(analytical(iZ, iR, iPhi)) < ABSTOLERANCE) {
          BOOST_CHECK_SMALL(numerical(iZ, iR, iPhi) - analytical(iZ, iR, iPhi), ABSTOLERANCE);
        } else {
          BOOST_CHECK_CLOSE(numerical(iZ, iR, iPhi), analytical(iZ, iR, iPhi), TOLERANCE);
        }
      }
    }
  }
}

template <typename DataT>
void testAlmostEqualArray2D(o2::tpc::DataContainer3D<DataT>& analytical, o2::tpc::DataContainer3D<DataT>& numerical)
{
  for (size_t iPhi = 0; iPhi < numerical.getNPhi(); ++iPhi) {
    for (size_t iR = 0; iR < numerical.getNR(); ++iR) {
      for (size_t iZ = 0; iZ < numerical.getNZ(); ++iZ) {
        if (std::fabs(analytical(iZ, iR, iPhi)) < ABSTOLERANCE) {
          BOOST_CHECK_SMALL(numerical(iZ, iR, iPhi) - analytical(iZ, iR, iPhi), ABSTOLERANCE);
        } else {
          BOOST_CHECK_CLOSE(numerical(iZ, iR, iPhi), analytical(iZ, iR, iPhi), TOLERANCE2D);
        }
      }
    }
  }
}

template <typename DataT>
void poissonSolver3D()
{
  using GridProp = GridProperties<DataT>;
  const o2::tpc::RegularGrid3D<DataT> grid3D{GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, GridProp::getGridSpacingZ(NZ), GridProp::getGridSpacingR(NR), GridProp::getGridSpacingPhi(NPHI)};

  using DataContainer = o2::tpc::DataContainer3D<DataT>;
  DataContainer potentialNumerical(NZ, NR, NPHI);
  DataContainer potentialAnalytical(NZ, NR, NPHI);
  DataContainer charge(NZ, NR, NPHI);

  const o2::tpc::AnalyticalFields<DataT> analyticalFields;
  // set the boudnary and charge for numerical poisson solver
  setChargeDensityFromFormula<DataT>(analyticalFields, grid3D, charge);
  setPotentialBoundaryFromFormula<DataT>(analyticalFields, grid3D, potentialNumerical);

  // set analytical potential
  setPotentialFromFormula<DataT>(analyticalFields, grid3D, potentialAnalytical);

  //calculate numerical potential
  PoissonSolver<DataT> poissonSolver(grid3D);
  const int symmetry = 0;
  poissonSolver.poissonSolver3D(potentialNumerical, charge, symmetry);

  // compare numerical with analytical solution of the potential
  testAlmostEqualArray<DataT>(potentialAnalytical, potentialNumerical);
}

template <typename DataT>
void poissonSolver2D()
{
  using GridProp = GridProperties<DataT>;
  const o2::tpc::RegularGrid3D<DataT> grid3D{GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, GridProp::getGridSpacingZ(NZ2D), GridProp::getGridSpacingR(NR2D), GridProp::getGridSpacingPhi(NPHI2D)};

  using DataContainer = o2::tpc::DataContainer3D<DataT>;
  DataContainer potentialNumerical(NZ2D, NR2D, NPHI2D);
  DataContainer potentialAnalytical(NZ2D, NR2D, NPHI2D);
  DataContainer charge(NZ2D, NR2D, NPHI2D);

  // set the boudnary and charge for numerical poisson solver
  const o2::tpc::AnalyticalFields<DataT> analyticalFields;
  setChargeDensityFromFormula<DataT>(analyticalFields, grid3D, charge);
  setPotentialBoundaryFromFormula<DataT>(analyticalFields, grid3D, potentialNumerical);

  // set analytical potential
  setPotentialFromFormula<DataT>(analyticalFields, grid3D, potentialAnalytical);

  //calculate numerical potential
  PoissonSolver<DataT> poissonSolver(grid3D);
  poissonSolver.poissonSolver2D(potentialNumerical, charge);

  // compare numerical with analytical solution of the potential
  testAlmostEqualArray2D<DataT>(potentialAnalytical, potentialNumerical);
}

BOOST_AUTO_TEST_CASE(PoissonSolver3D_test)
{
  o2::tpc::MGParameters::isFull3D = true; //3D

  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NZVertices", NZ);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NRVertices", NR);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NPhiVertices", NPHI);

  poissonSolver3D<DataT>();
}

BOOST_AUTO_TEST_CASE(PoissonSolver3D2D_test)
{
  o2::tpc::MGParameters::isFull3D = false; // 3D2D
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NZVertices", NZ);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NRVertices", NR);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NPhiVertices", NPHI);
  poissonSolver3D<DataT>();
}

BOOST_AUTO_TEST_CASE(PoissonSolver2D_test)
{
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NZVertices", NZ2D);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NRVertices", NR2D);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NPhiVertices", NPHI2D);
  poissonSolver2D<DataT>();
}

} // namespace tpc
} // namespace o2
