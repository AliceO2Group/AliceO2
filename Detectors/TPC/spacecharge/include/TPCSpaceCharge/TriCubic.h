// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TriCubic.h
/// \brief Definition of TriCubic class
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_TRICUBIC_H_
#define ALICEO2_TPC_TRICUBIC_H_

#include "TPCSpaceCharge/Vector.h"
#include "TPCSpaceCharge/RegularGrid3D.h"
#include "TPCSpaceCharge/DataContainer3D.h"

#if (defined(WITH_OPENMP) || defined(_OPENMP)) && !defined(__CLING__)
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_max_threads() { return 1; }
#endif

namespace o2
{
namespace tpc
{

/// \class TriCubicInterpolator
/// The TriCubic class represents tricubic interpolation on a regular 3-Dim grid.
/// The algorithm which is used is based on the method developed by F. Lekien and J. Marsden and is described
/// in 'Tricubic Interpolation in Three Dimensions (2005)'  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.89.7835
/// In this method in a first step 64 coefficients are computed by using a predefined 64*64 matrix.
/// These coefficients have to be computed for each cell in the grid, but are only computed when querying a point in a given cell.
/// The calculated coefficient is then stored for only the last cell and will be reused if the next query point lies in the same cell.
///
/// Additionally the classical one dimensional approach of interpolating values is implemented. This algorithm is faster when interpolating only a few values inside each cube.
///
/// periodic boundary conditions are used in phi direction.

/// void test()
/// {
///   // define the grid:
///   // define number of vertices per dimension
///   const int zvertices = 40;
///   const int rvertices = 40;
///   const int phivertices = 40;
///
///   // define min range
///   float zmin = 0;
///   float rmin = 0;
///   float phimin = 0;
///
///   // define spacing between grid vertices
///   float zSpacing = 0.25;
///   float rSpacing = 0.25;
///   float phiSpacing = 2 * M_PI / phivertices;
///
///   // create grid and datacontainer object
///   o2::tpc::RegularGrid3D<float, zvertices, rvertices, phivertices> grid3D(zmin, rmin, phimin, zSpacing, rSpacing, phiSpacing);
///   o2::tpc::DataContainer3D<float, zvertices, rvertices, phivertices> data3D;
///
///   // fill the DataContainer3D with some values
///   for (int iz = 0; iz < zvertices; ++iz) {
///     for (int ir = 0; ir < rvertices; ++ir) {
///       for (int iphi = 0; iphi < phivertices; ++iphi) {
///         const float izPos = zSpacing * iz + zmin;
///         const float irPos = rSpacing * ir + rmin;
///         const float iphiPos = phiSpacing * iphi + phimin;
///         data3D(iz, ir, iphi) = std::sin(irPos * izPos / 10.) + std::cos(iphiPos); // some arbitrary function is used here
///       }
///     }
///   }
///
///   // create tricubic interpolator
///   o2::tpc::TriCubicInterpolator<float, zvertices, rvertices, phivertices> interpolator(data3D, grid3D);
///
///   // query some values
///   for (float iz = grid3D.getGridMinX(); iz < grid3D.getGridMaxX(); iz += zSpacing / 3.) {
///     for (float ir = grid3D.getGridMinY(); ir < grid3D.getGridMaxY(); ir += rSpacing / 3.) {
///       for (float iphi = grid3D.getGridMinZ() - 2 * phiSpacing; iphi < grid3D.getGridMaxZ() + 2 * phiSpacing; iphi += phiSpacing / 3.) {
///         const float zQuery = iz;
///         const float rQuery = ir;
///         const float phiQuery = iphi;
///
///         const float interpolatedSparse = interpolator(zQuery, rQuery, phiQuery, o2::tpc::TriCubicInterpolator<float, zvertices, rvertices, phivertices>::InterpolationType::Sparse);
///         const float interpolatedDense = interpolator(zQuery, rQuery, phiQuery, o2::tpc::TriCubicInterpolator<float, zvertices, rvertices, phivertices>::InterpolationType::Dense);
///         const float trueValue = std::sin(rQuery * zQuery / 10.) + std::cos(phiQuery);
///         const float interpolatedDerivative = interpolator(zQuery, rQuery, phiQuery, 1, 1, 0);
///         const float trueDerivative = 1 / 10. * std::cos(rQuery * zQuery / 10.) - rQuery / 10. * std::sin(rQuery * zQuery / 10.) * zQuery / 10.;
///       }
///     }
///   }
/// }

/// \tparam DataT the type of data which is used during the calculations
/// \tparam Nz number of vertices in r direction
/// \tparam Nr number of vertices in phi direction
/// \tparam Nphi number of vertices in phi direction
template <typename DataT = double, size_t Nz = 129, size_t Nr = 129, size_t Nphi = 180>
class TriCubicInterpolator
{
  using Grid3D = RegularGrid3D<DataT, Nz, Nr, Nphi>;
  using DataContainer = DataContainer3D<DataT, Nz, Nr, Nphi>;
  using VDataT = Vc::Vector<DataT>;

 public:
  /// Constructor for a tricubic interpolator
  /// \param gridData struct containing access to the values of the grid
  /// \param gridProperties properties of the 3D grid
  TriCubicInterpolator(const DataContainer& gridData, const Grid3D& gridProperties) : mGridData{gridData}, mGridProperties{gridProperties} {};

  enum class ExtrapolationType {
    Linear = 0,   ///< assume linear dependency at the boundaries of the grid
    Parabola = 1, ///< assume parabolic dependency at the boundaries of the grid
  };

  enum class InterpolationType {
    Sparse = 0, ///< using one dimensional method of interpolation (fast when interpolating only a few values in one cube in the grid)
    Dense = 1,  ///< using three dimensional method of interpolation (fast when interpolating a lot values in one cube in the grid)
  };

  // interpolate value at given coordinate
  /// \param z z coordinate
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param type interpolation algorithm
  /// \return returns the interpolated value at given coordinate
  DataT operator()(const DataT z, const DataT r, const DataT phi, const InterpolationType type = InterpolationType::Sparse) const
  {
    if (type == InterpolationType::Sparse) {
      return interpolateSparse(z, r, phi);
    } else {
      const Vector<DataT, FDim> coordinates{{z, r, phi}}; // vector holding the coordinates
      const auto relPos = processInp(coordinates, false); // vector containing the relative position to
      return interpolateDense(relPos);
    }
  }

  /// interpolate derivative at given coordinate
  /// \param z z coordinate
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param derz order of derivative d/dz: derz=1 -> d/dz f(z,r,phi), derz=2 -> d^2/dz^2 f(z,r,phi), derz=3 -> d^3/dz^3 f(z,r,phi)
  /// \param derphi order of derivative d/dr: derr=1 -> d/dr f(z,r,phi), derr=2 -> d^2/dr^2 f(z,r,phi), derr=3 -> d^3/dr^3 f(z,r,phi)
  /// \param derphi order of derivative d/dphi: derphi=1 -> d/dphi f(z,r,phi), derphi=2 -> d^2/dphi^2 f(z,r,phi), derphi=3 -> d^3/dphi^3 f(z,r,phi)
  /// derz=1 and derr=2 -> d/dz * d^2/dr^2 * f(z,r,phi)
  /// \return returns the interpolated derivative at given coordinate
  DataT operator()(const DataT z, const DataT r, const DataT phi, const size_t derz, const size_t derr, const size_t derphi) const
  {
    const Vector<DataT, FDim> coordinates{{z, r, phi}}; // vector holding the coordinates
    const auto relPos = processInp(coordinates, false);
    return evalDerivative(relPos[0], relPos[1], relPos[2], derz, derr, derphi);
  }

  /// set which type of extrapolation is used at the grid boundaries (linear or parabol can be used with periodic phi axis and non periodic z and r axis).
  /// \param extrapolationType sets type of extrapolation. See enum ExtrapolationType for different types
  void setExtrapolationType(const ExtrapolationType extrapolationType) { mExtrapolationType = extrapolationType; }

  /// \return returns the extrapolation technique for missing boundary values
  ExtrapolationType getExtrapolationType() const { return mExtrapolationType; }

  /// get the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// set the number of threads used for some of the calculations
  static void setNThreads(int nThreads) { sNThreads = nThreads; }

  /// \return returns the number of the thread. Each thread should have an individual thread number
  int getThreadNum() const { return sThreadnum; }

  /// \return performs a check if the interpolator can be used with maximum number of threads
  bool checkThreadSafety() const { return sNThreads <= omp_get_max_threads(); }

 private:
  // matrix containing the 'relationship between the derivatives at the corners of the elements and the coefficients'
  inline static Vc::Memory<VDataT, 64> sMat[64]{
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {9, -9, -9, 9, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-6, 6, 6, -6, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-6, 6, 6, -6, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {4, -4, -4, 4, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
    {-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {9, -9, 0, 0, -9, 9, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-6, 6, 0, 0, 6, -6, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, 0, 0, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0},
    {9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0},
    {-27, 27, 27, -27, 27, -27, -27, 27, -18, -9, 18, 9, 18, 9, -18, -9, -18, 18, -9, 9, 18, -18, 9, -9, -18, 18, 18, -18, -9, 9, 9, -9, -12, -6, -6, -3, 12, 6, 6, 3, -12, -6, 12, 6, -6, -3, 6, 3, -12, 12, -6, 6, -6, 6, -3, 3, -8, -4, -4, -2, -4, -2, -2, -1},
    {18, -18, -18, 18, -18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 12, -12, 6, -6, -12, 12, -6, 6, 12, -12, -12, 12, 6, -6, -6, 6, 6, 6, 3, 3, -6, -6, -3, -3, 6, 6, -6, -6, 3, 3, -3, -3, 8, -8, 4, -4, 4, -4, 2, -2, 4, 4, 2, 2, 2, 2, 1, 1},
    {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0},
    {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 9, -9, 9, -9, -9, 9, -9, 9, 12, -12, -12, 12, 6, -6, -6, 6, 6, 3, 6, 3, -6, -3, -6, -3, 8, 4, -8, -4, 4, 2, -4, -2, 6, -6, 6, -6, 3, -3, 3, -3, 4, 2, 4, 2, 2, 1, 2, 1},
    {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -6, 6, -6, 6, 6, -6, 6, -6, -8, 8, 8, -8, -4, 4, 4, -4, -3, -3, -3, -3, 3, 3, 3, 3, -4, -4, 4, 4, -2, -2, 2, 2, -4, 4, -4, 4, -2, 2, -2, 2, -2, -2, -2, -2, -1, -1, -1, -1},
    {2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-6, 6, 0, 0, 6, -6, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {4, -4, 0, 0, -4, 4, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, 0, 0, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
    {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0},
    {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 12, -12, 6, -6, -12, 12, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 8, 4, 4, 2, -8, -4, -4, -2, 6, 3, -6, -3, 6, 3, -6, -3, 6, -6, 3, -3, 6, -6, 3, -3, 4, 2, 2, 1, 4, 2, 2, 1},
    {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -8, 8, -4, 4, 8, -8, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, -4, -4, -2, -2, 4, 4, 2, 2, -3, -3, 3, 3, -3, -3, 3, 3, -4, 4, -2, 2, -4, 4, -2, 2, -2, -2, -1, -1, -2, -2, -1, -1},
    {4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
    {-12, 12, 12, -12, 12, -12, -12, 12, -8, -4, 8, 4, 8, 4, -8, -4, -6, 6, -6, 6, 6, -6, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, -4, -2, -4, -2, 4, 2, 4, 2, -4, -2, 4, 2, -4, -2, 4, 2, -3, 3, -3, 3, -3, 3, -3, 3, -2, -1, -2, -1, -2, -1, -2, -1},
    {8, -8, -8, 8, -8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 4, -4, 4, -4, -4, 4, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 2, 2, 2, 2, -2, -2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 1, 1, 1, 1, 1, 1, 1, 1}}; ///< matrix containing the 'relationship between the derivatives at the corners of the elements and the coefficients'

  inline static Matrix<DataT, 64> sMatrixA{sMat};                                                        ///< this matrix is used for vectorized operations with the 64*64 matrix
  static constexpr unsigned int FDim = Grid3D::getDim();                                                 ///< dimensions of the grid
  static constexpr unsigned int FZ = Grid3D::getFX();                                                    ///< index for z coordinate
  static constexpr unsigned int FR = Grid3D::getFY();                                                    ///< index for r coordinate
  static constexpr unsigned int FPHI = Grid3D::getFZ();                                                  ///< index for phi coordinate
  const DataContainer& mGridData{};                                                                      ///< adress to the data container of the grid
  const Grid3D& mGridProperties{};                                                                       ///< adress to the properties of the grid
  inline static thread_local const size_t sThreadnum{static_cast<size_t>(omp_get_thread_num())};         ///< save for each thread the thread number to get fast access to the correct array
  inline static int sNThreads{omp_get_max_threads()};                                                    ///< number of threads the tricubic interpolator can be used with
  std::unique_ptr<Vector<DataT, 64>[]> mCoefficients = std::make_unique<Vector<DataT, 64>[]>(sNThreads); ///< coefficients needed to interpolate a value
  std::unique_ptr<Vector<DataT, FDim>[]> mLastInd = std::make_unique<Vector<DataT, FDim>[]>(sNThreads);  ///< stores the index for the cell, where the coefficients are already evaluated (only the coefficients for the last cell are stored)
  std::unique_ptr<bool[]> mInitialized = std::make_unique<bool[]>(sNThreads);                            ///< sets the flag if the coefficients are evaluated at least once
  ExtrapolationType mExtrapolationType = ExtrapolationType::Parabola;                                    ///< sets which type of extrapolation for missing points at boundary is used. Linear and Parabola is only supported for perdiodic phi axis and non periodic z and r axis

  //                 DEFINITION OF enum GridPos
  //========================================================
  //              r
  //              |             6------F---7
  //              |           / |        / |
  //              |         K   G YR   L   H
  //              |       /     |    /     |
  //              |      2---B------3      |
  //              |      |      |   |      |
  //              |      |      4---|---E--5
  //              |      C XL  /    D XR  /
  //              |      |   I  YL  |   J
  //              |      | /        | /
  //              |      0---A------1
  //              |------------------------------- z
  //            /
  //          /
  //        /
  //      phi
  //========================================================

  enum class GridPos {
    None = 27,
    InnerVolume = 26,
    Edge0 = 0,
    Edge1 = 1,
    Edge2 = 2,
    Edge3 = 3,
    Edge4 = 4,
    Edge5 = 5,
    Edge6 = 6,
    Edge7 = 7,
    LineA = 8,
    LineB = 9,
    LineC = 10,
    LineD = 11,
    LineE = 12,
    LineF = 13,
    LineG = 14,
    LineH = 15,
    LineI = 16,
    LineJ = 17,
    LineK = 18,
    LineL = 19,
    SideXRight = 20,
    SideXLeft = 21,
    SideYRight = 22,
    SideYLeft = 23,
    SideZRight = 24,
    SideZLeft = 25
  };

  void setValues(const int iz, const int ir, const int iphi, DataT cVals[64]) const;

  const Vector<DataT, 3> processInp(const Vector<DataT, 3>& coordinates, const bool sparse = false) const;

  // calculate the coefficients needed for the interpolation using the 64*64 matrix.
  // this is the 'slow' part of the code and might be optimized
  void calcCoefficients(const unsigned int iz, const unsigned int ir, const unsigned int iphi) const;

  DataT interpolateDense(const Vector<DataT, 3>& pos) const;

  // interpolate value at given coordinate - this method doesnt compute and stores the coefficients and is faster when quering only a few values per cube
  /// \param z z coordinate
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \return returns the interpolated value at given coordinate
  DataT interpolateSparse(const DataT z, const DataT r, const DataT phi) const;

  DataT evalDerivative(const DataT dz, const DataT dr, const DataT dphi, const size_t derz, const size_t derr, const size_t derphi) const;

  // for periodic boundary conditions
  void getDataIndexCircularArray(const int index0, const int dim, int arr[]) const;

  // for non periodic boundary conditions
  void getDataIndexNonCircularArray(const int index0, const int dim, int arr[]) const;

  // this helps to get circular and non circular padding indices
  int getRegulatedDelta(const int index0, const int delta, const unsigned int dim, const int offs) const
  {
    return mGridProperties.isIndexInGrid(index0 + delta, dim) ? delta : offs;
  }

  void initInterpolator(const unsigned int iz, const unsigned int ir, const unsigned int iphi) const;

  DataT extrapolation(const DataT valk, const DataT valk1, const DataT valk2) const;

  DataT linearExtrapolation(const DataT valk, const DataT valk1) const;

  DataT parabolExtrapolation(const DataT valk, const DataT valk1, const DataT valk2) const;

  GridPos findPos(const int iz, const int ir, const int iphi) const;

  bool isInInnerVolume(const int iz, const int ir, const int iphi, GridPos& posType) const;

  bool findEdge(const int iz, const int ir, const int iphi, GridPos& posType) const;

  bool findLine(const int iz, const int ir, const int iphi, GridPos& posType) const;

  bool findSide(const int iz, const int ir, const int iphi, GridPos& posType) const;

  bool isSideRight(const int ind, const int dim) const;

  bool isSideLeft(const int ind) const;
};

///
/// ========================================================================================================
///                                         Inline implementations
/// ========================================================================================================
///

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void TriCubicInterpolator<DataT, Nz, Nr, Nphi>::initInterpolator(const unsigned int iz, const unsigned int ir, const unsigned int iphi) const
{
  calcCoefficients(iz, ir, iphi);

  // store current cell
  mInitialized[sThreadnum] = true;
  mLastInd[sThreadnum][FZ] = iz;
  mLastInd[sThreadnum][FR] = ir;
  mLastInd[sThreadnum][FPHI] = iphi;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT TriCubicInterpolator<DataT, Nz, Nr, Nphi>::evalDerivative(const DataT dz, const DataT dr, const DataT dphi, const size_t derz, const size_t derr, const size_t derphi) const
{
  //TODO optimize this
  DataT ret{};
  for (size_t i = derz; i < 4; i++) {
    for (size_t j = derr; j < 4; j++) {
      for (size_t k = derphi; k < 4; k++) {

        const size_t index = i + j * 4 + 16 * k;
        DataT cont = mCoefficients[sThreadnum][index] * std::pow(dz, i - derz) * std::pow(dr, j - derr) * std::pow(dphi, k - derphi);
        for (size_t w = 0; w < derz; w++) {
          cont *= (i - w);
        }
        for (size_t w = 0; w < derr; w++) {
          cont *= (j - w);
        }
        for (size_t w = 0; w < derphi; w++) {
          cont *= (k - w);
        }
        ret += cont;
      }
    }
  }
  const DataT norm = std::pow(mGridProperties.getInvSpacingX(), derz) * std::pow(mGridProperties.getInvSpacingY(), derr) * std::pow(mGridProperties.getInvSpacingZ(), derphi);
  return (ret * norm);
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT TriCubicInterpolator<DataT, Nz, Nr, Nphi>::extrapolation(const DataT valk, const DataT valk1, const DataT valk2) const
{
  switch (mExtrapolationType) {
    case ExtrapolationType::Linear:
    default:
      return linearExtrapolation(valk, valk1);
      break;
    case ExtrapolationType::Parabola:
      return parabolExtrapolation(valk, valk1, valk2);
      break;
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT TriCubicInterpolator<DataT, Nz, Nr, Nphi>::linearExtrapolation(const DataT valk, const DataT valk1) const
{
  const DataT val = 2 * valk - valk1;
  return val;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT TriCubicInterpolator<DataT, Nz, Nr, Nphi>::parabolExtrapolation(const DataT valk, const DataT valk1, const DataT valk2) const
{
  const DataT val = 3 * (valk - valk1) + valk2; // legendre polynom with x0=0, x1=1, x2=2 and z=-1
  return val;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void TriCubicInterpolator<DataT, Nz, Nr, Nphi>::calcCoefficients(const unsigned int iz, const unsigned int ir, const unsigned int iphi) const
{
  DataT cVals[64]{};
  setValues(iz, ir, iphi, cVals);

  // needed for first derivative
  const Vector<DataT, 24> vecDeriv1A{
    {cVals[22], cVals[23], cVals[26], cVals[27], cVals[38], cVals[39], cVals[42], cVals[43],
     cVals[25], cVals[26], cVals[29], cVals[30], cVals[41], cVals[42], cVals[45], cVals[46],
     cVals[37], cVals[38], cVals[41], cVals[42], cVals[53], cVals[54], cVals[57], cVals[58]}};

  const Vector<DataT, 24> vecDeriv1B{
    {cVals[20], cVals[21], cVals[24], cVals[25], cVals[36], cVals[37], cVals[40], cVals[41],
     cVals[17], cVals[18], cVals[21], cVals[22], cVals[33], cVals[34], cVals[37], cVals[38],
     cVals[5], cVals[6], cVals[9], cVals[10], cVals[21], cVals[22], cVals[25], cVals[26]}};

  // needed for second derivative
  const Vector<DataT, 24> vecDeriv2A{
    {cVals[26], cVals[27], cVals[30], cVals[31], cVals[42], cVals[43], cVals[46], cVals[47],
     cVals[38], cVals[39], cVals[42], cVals[43], cVals[54], cVals[55], cVals[58], cVals[59],
     cVals[41], cVals[42], cVals[45], cVals[46], cVals[57], cVals[58], cVals[61], cVals[62]}};

  const Vector<DataT, 24> vecDeriv2B{
    {cVals[24], cVals[25], cVals[28], cVals[29], cVals[40], cVals[41], cVals[44], cVals[45],
     cVals[36], cVals[37], cVals[40], cVals[41], cVals[52], cVals[53], cVals[56], cVals[57],
     cVals[33], cVals[34], cVals[37], cVals[38], cVals[49], cVals[50], cVals[53], cVals[54]}};

  const Vector<DataT, 24> vecDeriv2C{
    {cVals[18], cVals[19], cVals[22], cVals[23], cVals[34], cVals[35], cVals[38], cVals[39],
     cVals[6], cVals[7], cVals[10], cVals[11], cVals[22], cVals[23], cVals[26], cVals[27],
     cVals[9], cVals[10], cVals[13], cVals[14], cVals[25], cVals[26], cVals[29], cVals[30]}};

  const Vector<DataT, 24> vecDeriv2D{
    {cVals[16], cVals[17], cVals[20], cVals[21], cVals[32], cVals[33], cVals[36], cVals[37],
     cVals[4], cVals[5], cVals[8], cVals[9], cVals[20], cVals[21], cVals[24], cVals[25],
     cVals[1], cVals[2], cVals[5], cVals[6], cVals[17], cVals[18], cVals[21], cVals[22]}};

  // needed for third derivative
  const Vector<DataT, 8> vecDeriv3A{{cVals[42], cVals[43], cVals[46], cVals[47], cVals[58], cVals[59], cVals[62], cVals[63]}};
  const Vector<DataT, 8> vecDeriv3B{{cVals[40], cVals[41], cVals[44], cVals[45], cVals[56], cVals[57], cVals[60], cVals[61]}};
  const Vector<DataT, 8> vecDeriv3C{{cVals[34], cVals[35], cVals[38], cVals[39], cVals[50], cVals[51], cVals[54], cVals[55]}};
  const Vector<DataT, 8> vecDeriv3D{{cVals[32], cVals[33], cVals[36], cVals[37], cVals[48], cVals[49], cVals[52], cVals[53]}};
  const Vector<DataT, 8> vecDeriv3E{{cVals[10], cVals[11], cVals[14], cVals[15], cVals[26], cVals[27], cVals[30], cVals[31]}};
  const Vector<DataT, 8> vecDeriv3F{{cVals[8], cVals[9], cVals[12], cVals[13], cVals[24], cVals[25], cVals[28], cVals[29]}};
  const Vector<DataT, 8> vecDeriv3G{{cVals[2], cVals[3], cVals[6], cVals[7], cVals[18], cVals[19], cVals[22], cVals[23]}};
  const Vector<DataT, 8> vecDeriv3H{{cVals[0], cVals[1], cVals[4], cVals[5], cVals[16], cVals[17], cVals[20], cVals[21]}};

  // factor for first derivative
  const DataT fac1{0.5};
  const Vector<DataT, 24> vfac1{{fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1, fac1}};

  // factor for second derivative
  const DataT fac2{0.25};
  const Vector<DataT, 24> vfac2{{fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2, fac2}};

  // factor for third derivative
  const DataT fac3{0.125};
  const Vector<DataT, 8> vfac3{{fac3, fac3, fac3, fac3, fac3, fac3, fac3, fac3}};

  // compute the derivatives
  const Vector<DataT, 24> vecDeriv1Res{vfac1 * (vecDeriv1A - vecDeriv1B)};
  const Vector<DataT, 24> vecDeriv2Res{vfac2 * (vecDeriv2A - vecDeriv2B - vecDeriv2C + vecDeriv2D)};
  const Vector<DataT, 8> vecDeriv3Res{vfac3 * (vecDeriv3A - vecDeriv3B - vecDeriv3C + vecDeriv3D - vecDeriv3E + vecDeriv3F + vecDeriv3G - vecDeriv3H)};

  const Vector<DataT, 64> matrixPar{
    {cVals[21], cVals[22], cVals[25], cVals[26], cVals[37], cVals[38], cVals[41], cVals[42],
     vecDeriv1Res[0], vecDeriv1Res[1], vecDeriv1Res[2], vecDeriv1Res[3], vecDeriv1Res[4], vecDeriv1Res[5], vecDeriv1Res[6], vecDeriv1Res[7], vecDeriv1Res[8], vecDeriv1Res[9], vecDeriv1Res[10],
     vecDeriv1Res[11], vecDeriv1Res[12], vecDeriv1Res[13], vecDeriv1Res[14], vecDeriv1Res[15], vecDeriv1Res[16], vecDeriv1Res[17], vecDeriv1Res[18], vecDeriv1Res[19], vecDeriv1Res[20], vecDeriv1Res[21],
     vecDeriv1Res[22], vecDeriv1Res[23], vecDeriv2Res[0], vecDeriv2Res[1], vecDeriv2Res[2], vecDeriv2Res[3], vecDeriv2Res[4], vecDeriv2Res[5], vecDeriv2Res[6], vecDeriv2Res[7], vecDeriv2Res[8], vecDeriv2Res[9],
     vecDeriv2Res[10], vecDeriv2Res[11], vecDeriv2Res[12], vecDeriv2Res[13], vecDeriv2Res[14], vecDeriv2Res[15], vecDeriv2Res[16], vecDeriv2Res[17], vecDeriv2Res[18], vecDeriv2Res[19], vecDeriv2Res[20],
     vecDeriv2Res[21], vecDeriv2Res[22], vecDeriv2Res[23], vecDeriv3Res[0], vecDeriv3Res[1], vecDeriv3Res[2], vecDeriv3Res[3], vecDeriv3Res[4], vecDeriv3Res[5], vecDeriv3Res[6], vecDeriv3Res[7]}};

  // calc coeffiecients
  mCoefficients[sThreadnum] = sMatrixA * matrixPar;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT TriCubicInterpolator<DataT, Nz, Nr, Nphi>::interpolateDense(const Vector<DataT, 3>& pos) const
{
  // the formula for evaluating the interpolation is as follows:
  // f(z,r,phi) = \sum_{i,j,k=0}^3 a_{ijk} * z^{i} * r^{j} * phi^{k}
  // a_{ijk} is stored in  mCoefficients[] and are computed in the function calcCoefficientsX()

  const Vector<DataT, FDim> vals0{{1, 1, 1}};   // z^0, r^0, phi^0
  const Vector<DataT, FDim> vals2{pos * pos};   // z^2, r^2, phi^2
  const Vector<DataT, FDim> vals3{vals2 * pos}; // z^3, r^3, phi^3

  const DataT valX[4]{vals0[FZ], pos[FZ], vals2[FZ], vals3[FZ]};
  const Vector<DataT, 64> vecValX{
    {valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3],
     valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3],
     valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3],
     valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3], valX[0], valX[1], valX[2], valX[3]}};

  const DataT valY[4]{vals0[FR], pos[FR], vals2[FR], vals3[FR]};
  const Vector<DataT, 64> vecValY{
    {valY[0], valY[0], valY[0], valY[0], valY[1], valY[1], valY[1], valY[1], valY[2], valY[2], valY[2], valY[2], valY[3], valY[3], valY[3], valY[3],
     valY[0], valY[0], valY[0], valY[0], valY[1], valY[1], valY[1], valY[1], valY[2], valY[2], valY[2], valY[2], valY[3], valY[3], valY[3], valY[3],
     valY[0], valY[0], valY[0], valY[0], valY[1], valY[1], valY[1], valY[1], valY[2], valY[2], valY[2], valY[2], valY[3], valY[3], valY[3], valY[3],
     valY[0], valY[0], valY[0], valY[0], valY[1], valY[1], valY[1], valY[1], valY[2], valY[2], valY[2], valY[2], valY[3], valY[3], valY[3], valY[3]}};

  const DataT valZ[4]{vals0[FPHI], pos[FPHI], vals2[FPHI], vals3[FPHI]};
  const Vector<DataT, 64> vecValZ{
    {valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0], valZ[0],
     valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1], valZ[1],
     valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2], valZ[2],
     valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3], valZ[3]}};

  // result = f(z,r,phi) = \sum_{i,j,k=0}^3 a_{ijk}    * z^{i}   * r^{j}   * phi^{k}
  const DataT result = sum(mCoefficients[sThreadnum] * vecValX * vecValY * vecValZ);
  return result;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT TriCubicInterpolator<DataT, Nz, Nr, Nphi>::interpolateSparse(const DataT z, const DataT r, const DataT phi) const
{
  const Vector<DataT, FDim> coordinates{{z, r, phi}}; // vector holding the coordinates
  const auto posRel = processInp(coordinates, true);

  DataT cVals[64]{};
  setValues(mLastInd[sThreadnum][FZ], mLastInd[sThreadnum][FR], mLastInd[sThreadnum][FPHI], cVals);

  const Vector<DataT, FDim> vals0{posRel};
  const Vector<DataT, FDim> vals1{vals0 * vals0};
  const Vector<DataT, FDim> vals2{vals0 * vals1};

  const int nPoints = 4;
  const Vector<DataT, nPoints> vecValX{{1, vals0[FZ], vals1[FZ], vals2[FZ]}};
  const Vector<DataT, nPoints> vecValY{{1, vals0[FR], vals1[FR], vals2[FR]}};
  const Vector<DataT, nPoints> vecValZ{{1, vals0[FPHI], vals1[FPHI], vals2[FPHI]}};

  const Vc::Memory<VDataT, nPoints> matrA[nPoints]{
    {0, -0.5, 1, -0.5},
    {1, 0, -2.5, 1.5},
    {0, 0.5, 2., -1.5},
    {0, 0, -0.5, 0.5}};

  const Matrix<DataT, nPoints> matrixA{matrA};
  const Vector<DataT, nPoints> vecValXMult{matrixA * vecValX};
  const Vector<DataT, nPoints> vecValYMult{matrixA * vecValY};
  const Vector<DataT, nPoints> vecValZMult{matrixA * vecValZ};

  DataT result{};
  int ind = 0;
  for (int slice = 0; slice < nPoints; ++slice) {
    const Vector<DataT, nPoints> vecA{vecValZMult[slice] * vecValYMult};
    for (int row = 0; row < nPoints; ++row) {
      const Vector<DataT, nPoints> vecD{{cVals[ind], cVals[++ind], cVals[++ind], cVals[++ind]}};
      ++ind;
      result += sum(vecA[row] * vecValXMult * vecD);
    }
  }
  return result;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
const Vector<DataT, 3> TriCubicInterpolator<DataT, Nz, Nr, Nphi>::processInp(const Vector<DataT, 3>& coordinates, const bool sparse) const
{
  Vector<DataT, FDim> posRel{(coordinates - mGridProperties.getGridMin()) * mGridProperties.getInvSpacing()}; // needed for the grid index
  posRel[FPHI] = mGridProperties.clampToGridCircularRel(posRel[FPHI], FPHI);
  const Vector<DataT, FDim> posRelN{posRel};
  posRel[FZ] = mGridProperties.clampToGridRel(posRel[FZ], FZ);
  posRel[FR] = mGridProperties.clampToGridRel(posRel[FR], FR);

  const Vector<DataT, FDim> index{floor(posRel)};

  if (!sparse && (!mInitialized[sThreadnum] || !(mLastInd[sThreadnum] == index))) {
    initInterpolator(index[FZ], index[FR], index[FPHI]);
  } else if (sparse) {
    mLastInd[sThreadnum][FZ] = index[FZ];
    mLastInd[sThreadnum][FR] = index[FR];
    mLastInd[sThreadnum][FPHI] = index[FPHI];
    mInitialized[sThreadnum] = false;
  }
  return posRelN - index;
}

// for perdiodic boundary condition
template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void TriCubicInterpolator<DataT, Nz, Nr, Nphi>::getDataIndexCircularArray(const int index0, const int dim, int arr[]) const
{
  const int delta_min1 = getRegulatedDelta(index0, -1, dim, mGridProperties.getN(dim) - 1);
  const int delta_plus1 = getRegulatedDelta(index0, +1, dim, 1 - mGridProperties.getN(dim));
  const int delta_plus2 = getRegulatedDelta(index0, +2, dim, 2 - mGridProperties.getN(dim));

  arr[0] = mGridProperties.getDeltaDataIndex(delta_min1, dim);
  arr[1] = mGridProperties.getDeltaDataIndex(delta_plus1, dim);
  arr[2] = mGridProperties.getDeltaDataIndex(delta_plus2, dim);
}

// for non perdiodic boundary condition
template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void TriCubicInterpolator<DataT, Nz, Nr, Nphi>::getDataIndexNonCircularArray(const int index0, const int dim, int arr[]) const
{
  const int delta_min1 = getRegulatedDelta(index0, -1, dim, 0);
  const int delta_plus1 = getRegulatedDelta(index0, +1, dim, 0);
  const int delta_plus2 = getRegulatedDelta(index0, +2, dim, delta_plus1);

  arr[0] = mGridProperties.getDeltaDataIndex(delta_min1, dim);
  arr[1] = mGridProperties.getDeltaDataIndex(delta_plus1, dim);
  arr[2] = mGridProperties.getDeltaDataIndex(delta_plus2, dim);
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
typename TriCubicInterpolator<DataT, Nz, Nr, Nphi>::GridPos TriCubicInterpolator<DataT, Nz, Nr, Nphi>::findPos(const int iz, const int ir, const int iphi) const
{
  GridPos pos = GridPos::None;
  if (isInInnerVolume(iz, ir, iphi, pos)) {
    return pos;
  }

  if (findEdge(iz, ir, iphi, pos)) {
    return pos;
  }

  if (findLine(iz, ir, iphi, pos)) {
    return pos;
  }

  if (findSide(iz, ir, iphi, pos)) {
    return pos;
  }
  return GridPos::None;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
bool TriCubicInterpolator<DataT, Nz, Nr, Nphi>::findEdge(const int iz, const int ir, const int iphi, GridPos& posType) const
{
  const int iR = 2;
  if (iz == 0 && ir == 0) {
    if (iphi == 0) {
      posType = GridPos::Edge0;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::Edge4;
      return true;
    }
  } else if (iz == Nz - iR && ir == 0) {
    if (iphi == 0) {
      posType = GridPos::Edge1;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::Edge5;
      return true;
    }
  } else if (iz == 0 && ir == Nr - iR) {
    if (iphi == 0) {
      posType = GridPos::Edge2;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::Edge6;
      return true;
    }
  } else if (iz == Nz - iR && ir == Nr - iR) {
    if (iphi == 0) {
      posType = GridPos::Edge3;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::Edge7;
      return true;
    }
  }
  return false;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
bool TriCubicInterpolator<DataT, Nz, Nr, Nphi>::findLine(const int iz, const int ir, const int iphi, GridPos& posType) const
{
  const int iR = 2;
  //check line
  if (ir == 0) {
    if (iphi == 0) {
      posType = GridPos::LineA;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::LineE;
      return true;
    }
    if (iz == 0) {
      posType = GridPos::LineI;
      return true;
    } else if (iz == Nz - iR) {
      posType = GridPos::LineJ;
      return true;
    }
  } else if (ir == Nr - iR) {
    if (iphi == 0) {
      posType = GridPos::LineB;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::LineF;
      return true;
    }
    if (iz == 0) {
      posType = GridPos::LineK;
      return true;
    } else if (iz == Nz - iR) {
      posType = GridPos::LineL;
      return true;
    }
  } else if (iz == 0) {
    if (iphi == 0) {
      posType = GridPos::LineC;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::LineG;
      return true;
    }
  } else if (iz == Nz - iR) {
    if (iphi == 0) {
      posType = GridPos::LineD;
      return true;
    } else if (iphi == Nphi - iR) {
      posType = GridPos::LineH;
      return true;
    }
  }
  return false;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
bool TriCubicInterpolator<DataT, Nz, Nr, Nphi>::findSide(const int iz, const int ir, const int iphi, GridPos& posType) const
{
  if (isSideRight(iz, FZ)) {
    posType = GridPos::SideXRight;
    return true;
  } else if (isSideLeft(iz)) {
    posType = GridPos::SideXLeft;
    return true;
  }
  if (isSideRight(ir, FR)) {
    posType = GridPos::SideYRight;
    return true;
  } else if (isSideLeft(ir)) {
    posType = GridPos::SideYLeft;
    return true;
  }
  if (isSideRight(iphi, FPHI)) {
    posType = GridPos::SideZRight;
    return true;
  } else if (isSideLeft(iphi)) {
    posType = GridPos::SideZLeft;
    return true;
  }
  return false;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
bool TriCubicInterpolator<DataT, Nz, Nr, Nphi>::isInInnerVolume(const int iz, const int ir, const int iphi, GridPos& posType) const
{
  if (iz >= 1 && iz < static_cast<int>(Nz - 2) && ir >= 1 && ir < static_cast<int>(Nr - 2) && iphi >= 1 && iphi < static_cast<int>(Nphi - 2)) {
    posType = GridPos::InnerVolume;
    return true;
  }
  return false;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
bool TriCubicInterpolator<DataT, Nz, Nr, Nphi>::isSideRight(const int ind, const int dim) const
{
  if (ind == static_cast<int>(mGridProperties.getN(dim) - 2)) {
    return true;
  }
  return false;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
bool TriCubicInterpolator<DataT, Nz, Nr, Nphi>::isSideLeft(const int ind) const
{
  if (ind == 0) {
    return true;
  }
  return false;
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void TriCubicInterpolator<DataT, Nz, Nr, Nphi>::setValues(const int iz, const int ir, const int iphi, DataT cVals[64]) const
{
  const GridPos location = findPos(iz, ir, iphi);
  const int ii_x_y_z = mGridData.getDataIndex(iz, ir, iphi);
  cVals[21] = mGridData[ii_x_y_z];

  int deltaZ[3]{mGridProperties.getDeltaDataIndex(-1, 0), mGridProperties.getDeltaDataIndex(1, 0), mGridProperties.getDeltaDataIndex(2, 0)};
  int deltaR[3]{mGridProperties.getDeltaDataIndex(-1, 1), mGridProperties.getDeltaDataIndex(1, 1), mGridProperties.getDeltaDataIndex(2, 1)};
  int deltaPhi[3]{};
  getDataIndexCircularArray(iphi, FPHI, deltaPhi);

  const int i0 = 0;
  const int i1 = 1;
  const int i2 = 2;

  switch (location) {
    case GridPos::InnerVolume:
    case GridPos::SideZRight:
    case GridPos::SideZLeft:
    default: {
      const int ind[4][4][4]{
        {{ii_x_y_z + deltaPhi[i0] + deltaR[i0] + deltaZ[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0], ind[0][0][2] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0], ind[0][1][2] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0], ind[0][2][2] - deltaZ[i0]},
         {ind[0][2][0] - deltaR[i0], ind[0][3][0] - deltaZ[i0], ind[0][3][1] - deltaZ[i0], ind[0][3][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaR[i0] + deltaZ[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0], ind[1][0][2] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0], ind[1][1][2] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0], ind[1][2][2] - deltaZ[i0]},
         {ind[1][2][0] - deltaR[i0], ind[1][3][0] - deltaZ[i0], ind[1][3][1] - deltaZ[i0], ind[1][3][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaR[i0] + deltaZ[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0], ind[2][0][2] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0], ind[2][1][2] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0], ind[2][2][2] - deltaZ[i0]},
         {ind[2][2][0] - deltaR[i0], ind[2][3][0] - deltaZ[i0], ind[2][3][1] - deltaZ[i0], ind[2][3][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaR[i0] + deltaZ[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0], ind[3][0][2] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0], ind[3][1][2] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0], ind[3][2][2] - deltaZ[i0]},
         {ind[3][2][0] - deltaR[i0], ind[3][3][0] - deltaZ[i0], ind[3][3][1] - deltaZ[i0], ind[3][3][2] - deltaZ[i0]}}};

      cVals[0] = mGridData[ind[0][0][0]];
      cVals[1] = mGridData[ind[0][0][1]];
      cVals[2] = mGridData[ind[0][0][2]];
      cVals[3] = mGridData[ind[0][0][3]];
      cVals[4] = mGridData[ind[0][1][0]];
      cVals[5] = mGridData[ind[0][1][1]];
      cVals[6] = mGridData[ind[0][1][2]];
      cVals[7] = mGridData[ind[0][1][3]];
      cVals[8] = mGridData[ind[0][2][0]];
      cVals[9] = mGridData[ind[0][2][1]];
      cVals[10] = mGridData[ind[0][2][2]];
      cVals[11] = mGridData[ind[0][2][3]];
      cVals[12] = mGridData[ind[0][3][0]];
      cVals[13] = mGridData[ind[0][3][1]];
      cVals[14] = mGridData[ind[0][3][2]];
      cVals[15] = mGridData[ind[0][3][3]];
      cVals[16] = mGridData[ind[1][0][0]];
      cVals[17] = mGridData[ind[1][0][1]];
      cVals[18] = mGridData[ind[1][0][2]];
      cVals[19] = mGridData[ind[1][0][3]];
      cVals[22] = mGridData[ind[1][1][2]];
      cVals[20] = mGridData[ind[1][1][0]];
      cVals[23] = mGridData[ind[1][1][3]];
      cVals[24] = mGridData[ind[1][2][0]];
      cVals[25] = mGridData[ind[1][2][1]];
      cVals[26] = mGridData[ind[1][2][2]];
      cVals[27] = mGridData[ind[1][2][3]];
      cVals[28] = mGridData[ind[1][3][0]];
      cVals[29] = mGridData[ind[1][3][1]];
      cVals[30] = mGridData[ind[1][3][2]];
      cVals[31] = mGridData[ind[1][3][3]];
      cVals[32] = mGridData[ind[2][0][0]];
      cVals[33] = mGridData[ind[2][0][1]];
      cVals[34] = mGridData[ind[2][0][2]];
      cVals[35] = mGridData[ind[2][0][3]];
      cVals[36] = mGridData[ind[2][1][0]];
      cVals[37] = mGridData[ind[2][1][1]];
      cVals[38] = mGridData[ind[2][1][2]];
      cVals[39] = mGridData[ind[2][1][3]];
      cVals[40] = mGridData[ind[2][2][0]];
      cVals[41] = mGridData[ind[2][2][1]];
      cVals[42] = mGridData[ind[2][2][2]];
      cVals[43] = mGridData[ind[2][2][3]];
      cVals[44] = mGridData[ind[2][3][0]];
      cVals[45] = mGridData[ind[2][3][1]];
      cVals[46] = mGridData[ind[2][3][2]];
      cVals[47] = mGridData[ind[2][3][3]];
      cVals[48] = mGridData[ind[3][0][0]];
      cVals[49] = mGridData[ind[3][0][1]];
      cVals[50] = mGridData[ind[3][0][2]];
      cVals[51] = mGridData[ind[3][0][3]];
      cVals[52] = mGridData[ind[3][1][0]];
      cVals[53] = mGridData[ind[3][1][1]];
      cVals[54] = mGridData[ind[3][1][2]];
      cVals[55] = mGridData[ind[3][1][3]];
      cVals[56] = mGridData[ind[3][2][0]];
      cVals[57] = mGridData[ind[3][2][1]];
      cVals[58] = mGridData[ind[3][2][2]];
      cVals[59] = mGridData[ind[3][2][3]];
      cVals[60] = mGridData[ind[3][3][0]];
      cVals[61] = mGridData[ind[3][3][1]];
      cVals[62] = mGridData[ind[3][3][2]];
      cVals[63] = mGridData[ind[3][3][3]];
    } break;

    case GridPos::SideXRight:
    case GridPos::LineD:
    case GridPos::LineH: {
      const int ind[4][4][3]{
        {{ii_x_y_z + deltaPhi[i0] + deltaR[i0] + deltaZ[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0]},
         {ind[0][2][0] - deltaR[i0], ind[0][3][0] - deltaZ[i0], ind[0][3][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaR[i0] + deltaZ[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0]},
         {ind[1][2][0] - deltaR[i0], ind[1][3][0] - deltaZ[i0], ind[1][3][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaR[i0] + deltaZ[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0]},
         {ind[2][2][0] - deltaR[i0], ind[2][3][0] - deltaZ[i0], ind[2][3][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaR[i0] + deltaZ[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0]},
         {ind[3][2][0] - deltaR[i0], ind[3][3][0] - deltaZ[i0], ind[3][3][1] - deltaZ[i0]}}};

      cVals[0] = mGridData[ind[0][0][0]];
      cVals[1] = mGridData[ind[0][0][1]];
      cVals[2] = mGridData[ind[0][0][2]];
      cVals[3] = extrapolation(mGridData[ind[0][0][2]], mGridData[ind[0][0][1]], mGridData[ind[0][0][0]]);
      cVals[4] = mGridData[ind[0][1][0]];
      cVals[5] = mGridData[ind[0][1][1]];
      cVals[6] = mGridData[ind[0][1][2]];
      cVals[7] = extrapolation(mGridData[ind[0][1][2]], mGridData[ind[0][1][1]], mGridData[ind[0][1][0]]);
      cVals[8] = mGridData[ind[0][2][0]];
      cVals[9] = mGridData[ind[0][2][1]];
      cVals[10] = mGridData[ind[0][2][2]];
      cVals[11] = extrapolation(mGridData[ind[0][2][2]], mGridData[ind[0][2][1]], mGridData[ind[0][2][0]]);
      cVals[12] = mGridData[ind[0][3][0]];
      cVals[13] = mGridData[ind[0][3][1]];
      cVals[14] = mGridData[ind[0][3][2]];
      cVals[15] = extrapolation(mGridData[ind[0][3][2]], mGridData[ind[0][3][1]], mGridData[ind[0][3][0]]);
      cVals[16] = mGridData[ind[1][0][0]];
      cVals[17] = mGridData[ind[1][0][1]];
      cVals[18] = mGridData[ind[1][0][2]];
      cVals[19] = extrapolation(mGridData[ind[1][0][2]], mGridData[ind[1][0][1]], mGridData[ind[1][0][0]]);
      cVals[20] = mGridData[ind[1][1][0]];
      cVals[22] = mGridData[ind[1][1][2]];
      cVals[23] = extrapolation(mGridData[ind[1][1][2]], mGridData[ii_x_y_z], mGridData[ind[1][1][0]]);
      cVals[24] = mGridData[ind[1][2][0]];
      cVals[25] = mGridData[ind[1][2][1]];
      cVals[26] = mGridData[ind[1][2][2]];
      cVals[27] = extrapolation(mGridData[ind[1][2][2]], mGridData[ind[1][2][1]], mGridData[ind[1][2][0]]);
      cVals[28] = mGridData[ind[1][3][0]];
      cVals[29] = mGridData[ind[1][3][1]];
      cVals[30] = mGridData[ind[1][3][2]];
      cVals[31] = extrapolation(mGridData[ind[1][3][2]], mGridData[ind[1][3][1]], mGridData[ind[1][3][0]]);
      cVals[32] = mGridData[ind[2][0][0]];
      cVals[33] = mGridData[ind[2][0][1]];
      cVals[34] = mGridData[ind[2][0][2]];
      cVals[35] = extrapolation(mGridData[ind[2][0][2]], mGridData[ind[2][0][1]], mGridData[ind[2][0][0]]);
      cVals[36] = mGridData[ind[2][1][0]];
      cVals[37] = mGridData[ind[2][1][1]];
      cVals[38] = mGridData[ind[2][1][2]];
      cVals[39] = extrapolation(mGridData[ind[2][1][2]], mGridData[ind[2][1][1]], mGridData[ind[2][1][0]]);
      cVals[40] = mGridData[ind[2][2][0]];
      cVals[41] = mGridData[ind[2][2][1]];
      cVals[42] = mGridData[ind[2][2][2]];
      cVals[43] = extrapolation(mGridData[ind[2][2][2]], mGridData[ind[2][2][1]], mGridData[ind[2][2][0]]);
      cVals[44] = mGridData[ind[2][3][0]];
      cVals[45] = mGridData[ind[2][3][1]];
      cVals[46] = mGridData[ind[2][3][2]];
      cVals[47] = extrapolation(mGridData[ind[2][3][2]], mGridData[ind[2][3][1]], mGridData[ind[2][3][0]]);
      cVals[48] = mGridData[ind[3][0][0]];
      cVals[49] = mGridData[ind[3][0][1]];
      cVals[50] = mGridData[ind[3][0][2]];
      cVals[52] = mGridData[ind[3][1][0]];
      cVals[51] = extrapolation(mGridData[ind[3][0][2]], mGridData[ind[3][0][1]], mGridData[ind[3][0][0]]);
      cVals[53] = mGridData[ind[3][1][1]];
      cVals[54] = mGridData[ind[3][1][2]];
      cVals[55] = extrapolation(mGridData[ind[3][1][2]], mGridData[ind[3][1][1]], mGridData[ind[3][1][0]]);
      cVals[56] = mGridData[ind[3][2][0]];
      cVals[57] = mGridData[ind[3][2][1]];
      cVals[58] = mGridData[ind[3][2][2]];
      cVals[59] = extrapolation(mGridData[ind[3][2][2]], mGridData[ind[3][2][1]], mGridData[ind[3][2][0]]);
      cVals[60] = mGridData[ind[3][3][0]];
      cVals[61] = mGridData[ind[3][3][1]];
      cVals[62] = mGridData[ind[3][3][2]];
      cVals[63] = extrapolation(mGridData[ind[3][3][2]], mGridData[ind[3][3][1]], mGridData[ind[3][3][0]]);
    } break;

    case GridPos::SideYRight:
    case GridPos::LineB:
    case GridPos::LineF: {
      const int ind[4][3][4]{
        {{ii_x_y_z + deltaPhi[i0] + deltaR[i0] + deltaZ[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0], ind[0][0][2] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0], ind[0][1][2] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0], ind[0][2][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaR[i0] + deltaZ[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0], ind[1][0][2] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0], ind[1][1][2] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0], ind[1][2][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaR[i0] + deltaZ[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0], ind[2][0][2] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0], ind[2][1][2] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0], ind[2][2][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaR[i0] + deltaZ[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0], ind[3][0][2] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0], ind[3][1][2] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0], ind[3][2][2] - deltaZ[i0]}}};

      cVals[0] = mGridData[ind[0][0][0]];
      cVals[1] = mGridData[ind[0][0][1]];
      cVals[2] = mGridData[ind[0][0][2]];
      cVals[3] = mGridData[ind[0][0][3]];
      cVals[4] = mGridData[ind[0][1][0]];
      cVals[5] = mGridData[ind[0][1][1]];
      cVals[6] = mGridData[ind[0][1][2]];
      cVals[7] = mGridData[ind[0][1][3]];
      cVals[8] = mGridData[ind[0][2][0]];
      cVals[9] = mGridData[ind[0][2][1]];
      cVals[10] = mGridData[ind[0][2][2]];
      cVals[11] = mGridData[ind[0][2][3]];
      cVals[12] = extrapolation(mGridData[ind[0][2][0]], mGridData[ind[0][1][0]], mGridData[ind[0][0][0]]);
      cVals[13] = extrapolation(mGridData[ind[0][2][1]], mGridData[ind[0][1][1]], mGridData[ind[0][0][1]]);
      cVals[14] = extrapolation(mGridData[ind[0][2][2]], mGridData[ind[0][1][2]], mGridData[ind[0][0][2]]);
      cVals[15] = extrapolation(mGridData[ind[0][2][3]], mGridData[ind[0][1][3]], mGridData[ind[0][0][3]]);
      cVals[16] = mGridData[ind[1][0][0]];
      cVals[17] = mGridData[ind[1][0][1]];
      cVals[18] = mGridData[ind[1][0][2]];
      cVals[19] = mGridData[ind[1][0][3]];
      cVals[20] = mGridData[ind[1][1][0]];
      cVals[22] = mGridData[ind[1][1][2]];
      cVals[23] = mGridData[ind[1][1][3]];
      cVals[24] = mGridData[ind[1][2][0]];
      cVals[25] = mGridData[ind[1][2][1]];
      cVals[26] = mGridData[ind[1][2][2]];
      cVals[27] = mGridData[ind[1][2][3]];
      cVals[28] = extrapolation(mGridData[ind[1][2][0]], mGridData[ind[1][1][0]], mGridData[ind[1][0][0]]);
      cVals[29] = extrapolation(mGridData[ind[1][2][1]], mGridData[ii_x_y_z], mGridData[ind[1][0][1]]);
      cVals[30] = extrapolation(mGridData[ind[1][2][2]], mGridData[ind[1][1][2]], mGridData[ind[1][0][2]]);
      cVals[31] = extrapolation(mGridData[ind[1][2][3]], mGridData[ind[1][1][3]], mGridData[ind[1][0][3]]);
      cVals[32] = mGridData[ind[2][0][0]];
      cVals[33] = mGridData[ind[2][0][1]];
      cVals[34] = mGridData[ind[2][0][2]];
      cVals[35] = mGridData[ind[2][0][3]];
      cVals[36] = mGridData[ind[2][1][0]];
      cVals[37] = mGridData[ind[2][1][1]];
      cVals[38] = mGridData[ind[2][1][2]];
      cVals[39] = mGridData[ind[2][1][3]];
      cVals[40] = mGridData[ind[2][2][0]];
      cVals[41] = mGridData[ind[2][2][1]];
      cVals[42] = mGridData[ind[2][2][2]];
      cVals[43] = mGridData[ind[2][2][3]];
      cVals[44] = extrapolation(mGridData[ind[2][2][0]], mGridData[ind[2][1][0]], mGridData[ind[2][0][0]]);
      cVals[45] = extrapolation(mGridData[ind[2][2][1]], mGridData[ind[2][1][1]], mGridData[ind[2][0][1]]);
      cVals[46] = extrapolation(mGridData[ind[2][2][2]], mGridData[ind[2][1][2]], mGridData[ind[2][0][2]]);
      cVals[47] = extrapolation(mGridData[ind[2][2][3]], mGridData[ind[2][1][3]], mGridData[ind[2][0][3]]);
      cVals[48] = mGridData[ind[3][0][0]];
      cVals[49] = mGridData[ind[3][0][1]];
      cVals[50] = mGridData[ind[3][0][2]];
      cVals[51] = mGridData[ind[3][0][3]];
      cVals[52] = mGridData[ind[3][1][0]];
      cVals[53] = mGridData[ind[3][1][1]];
      cVals[54] = mGridData[ind[3][1][2]];
      cVals[55] = mGridData[ind[3][1][3]];
      cVals[56] = mGridData[ind[3][2][0]];
      cVals[57] = mGridData[ind[3][2][1]];
      cVals[58] = mGridData[ind[3][2][2]];
      cVals[59] = mGridData[ind[3][2][3]];
      cVals[60] = extrapolation(mGridData[ind[3][2][0]], mGridData[ind[3][1][0]], mGridData[ind[3][0][0]]);
      cVals[61] = extrapolation(mGridData[ind[3][2][1]], mGridData[ind[3][1][1]], mGridData[ind[3][0][1]]);
      cVals[62] = extrapolation(mGridData[ind[3][2][2]], mGridData[ind[3][1][2]], mGridData[ind[3][0][2]]);
      cVals[63] = extrapolation(mGridData[ind[3][2][3]], mGridData[ind[3][1][3]], mGridData[ind[3][0][3]]);
    } break;

    case GridPos::SideYLeft:
    case GridPos::LineA:
    case GridPos::LineE: {
      const int ind[4][3][4]{
        {{ii_x_y_z + deltaPhi[i0] + deltaZ[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0], ind[0][0][2] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0], ind[0][1][2] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0], ind[0][2][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaZ[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0], ind[1][0][2] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0], ind[1][1][2] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0], ind[1][2][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaZ[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0], ind[2][0][2] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0], ind[2][1][2] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0], ind[2][2][2] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaZ[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0], ind[3][0][2] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0], ind[3][1][2] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0], ind[3][2][2] - deltaZ[i0]}}};

      cVals[0] = extrapolation(mGridData[ind[0][0][0]], mGridData[ind[0][1][0]], mGridData[ind[0][2][0]]);
      cVals[1] = extrapolation(mGridData[ind[0][0][1]], mGridData[ind[0][1][1]], mGridData[ind[0][2][1]]);
      cVals[2] = extrapolation(mGridData[ind[0][0][2]], mGridData[ind[0][1][2]], mGridData[ind[0][2][2]]);
      cVals[3] = extrapolation(mGridData[ind[0][0][3]], mGridData[ind[0][1][3]], mGridData[ind[0][2][3]]);
      cVals[4] = mGridData[ind[0][0][0]];
      cVals[5] = mGridData[ind[0][0][1]];
      cVals[6] = mGridData[ind[0][0][2]];
      cVals[7] = mGridData[ind[0][0][3]];
      cVals[8] = mGridData[ind[0][1][0]];
      cVals[9] = mGridData[ind[0][1][1]];
      cVals[10] = mGridData[ind[0][1][2]];
      cVals[11] = mGridData[ind[0][1][3]];
      cVals[12] = mGridData[ind[0][2][0]];
      cVals[13] = mGridData[ind[0][2][1]];
      cVals[14] = mGridData[ind[0][2][2]];
      cVals[15] = mGridData[ind[0][2][3]];
      cVals[16] = extrapolation(mGridData[ind[1][0][0]], mGridData[ind[1][1][0]], mGridData[ind[1][2][0]]);
      cVals[17] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][1][1]], mGridData[ind[1][2][1]]);
      cVals[18] = extrapolation(mGridData[ind[1][0][2]], mGridData[ind[1][1][2]], mGridData[ind[1][2][2]]);
      cVals[19] = extrapolation(mGridData[ind[1][0][3]], mGridData[ind[1][1][3]], mGridData[ind[1][2][3]]);
      cVals[20] = mGridData[ind[1][0][0]];
      cVals[22] = mGridData[ind[1][0][2]];
      cVals[23] = mGridData[ind[1][0][3]];
      cVals[24] = mGridData[ind[1][1][0]];
      cVals[25] = mGridData[ind[1][1][1]];
      cVals[26] = mGridData[ind[1][1][2]];
      cVals[27] = mGridData[ind[1][1][3]];
      cVals[28] = mGridData[ind[1][2][0]];
      cVals[29] = mGridData[ind[1][2][1]];
      cVals[30] = mGridData[ind[1][2][2]];
      cVals[31] = mGridData[ind[1][2][3]];
      cVals[32] = extrapolation(mGridData[ind[2][0][0]], mGridData[ind[2][1][0]], mGridData[ind[2][2][0]]);
      cVals[33] = extrapolation(mGridData[ind[2][0][1]], mGridData[ind[2][1][1]], mGridData[ind[2][2][1]]);
      cVals[34] = extrapolation(mGridData[ind[2][0][2]], mGridData[ind[2][1][2]], mGridData[ind[2][2][2]]);
      cVals[35] = extrapolation(mGridData[ind[2][0][3]], mGridData[ind[2][1][3]], mGridData[ind[2][2][3]]);
      cVals[36] = mGridData[ind[2][0][0]];
      cVals[37] = mGridData[ind[2][0][1]];
      cVals[38] = mGridData[ind[2][0][2]];
      cVals[39] = mGridData[ind[2][0][3]];
      cVals[40] = mGridData[ind[2][1][0]];
      cVals[41] = mGridData[ind[2][1][1]];
      cVals[42] = mGridData[ind[2][1][2]];
      cVals[43] = mGridData[ind[2][1][3]];
      cVals[44] = mGridData[ind[2][2][0]];
      cVals[45] = mGridData[ind[2][2][1]];
      cVals[46] = mGridData[ind[2][2][2]];
      cVals[47] = mGridData[ind[2][2][3]];
      cVals[48] = extrapolation(mGridData[ind[3][0][0]], mGridData[ind[3][1][0]], mGridData[ind[3][2][0]]);
      cVals[49] = extrapolation(mGridData[ind[3][0][1]], mGridData[ind[3][1][1]], mGridData[ind[3][2][1]]);
      cVals[50] = extrapolation(mGridData[ind[3][0][2]], mGridData[ind[3][1][2]], mGridData[ind[3][2][2]]);
      cVals[51] = extrapolation(mGridData[ind[3][0][3]], mGridData[ind[3][1][3]], mGridData[ind[3][2][3]]);
      cVals[52] = mGridData[ind[3][0][0]];
      cVals[53] = mGridData[ind[3][0][1]];
      cVals[54] = mGridData[ind[3][0][2]];
      cVals[55] = mGridData[ind[3][0][3]];
      cVals[56] = mGridData[ind[3][1][0]];
      cVals[57] = mGridData[ind[3][1][1]];
      cVals[58] = mGridData[ind[3][1][2]];
      cVals[59] = mGridData[ind[3][1][3]];
      cVals[60] = mGridData[ind[3][2][0]];
      cVals[61] = mGridData[ind[3][2][1]];
      cVals[62] = mGridData[ind[3][2][2]];
      cVals[63] = mGridData[ind[3][2][3]];
    } break;

    case GridPos::SideXLeft:
    case GridPos::LineC:
    case GridPos::LineG: {
      const int ind[4][4][3]{
        {{ii_x_y_z + deltaPhi[i0] + deltaR[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0]},
         {ind[0][2][0] - deltaR[i0], ind[0][3][0] - deltaZ[i0], ind[0][3][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaR[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0]},
         {ind[1][2][0] - deltaR[i0], ind[1][3][0] - deltaZ[i0], ind[1][3][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaR[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0]},
         {ind[2][2][0] - deltaR[i0], ind[2][3][0] - deltaZ[i0], ind[2][3][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaR[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0]},
         {ind[3][2][0] - deltaR[i0], ind[3][3][0] - deltaZ[i0], ind[3][3][1] - deltaZ[i0]}}};

      cVals[0] = extrapolation(mGridData[ind[0][0][0]], mGridData[ind[0][0][1]], mGridData[ind[0][0][2]]);
      cVals[1] = mGridData[ind[0][0][0]];
      cVals[2] = mGridData[ind[0][0][1]];
      cVals[3] = mGridData[ind[0][0][2]];
      cVals[4] = extrapolation(mGridData[ind[0][1][0]], mGridData[ind[0][1][1]], mGridData[ind[0][1][2]]);
      cVals[5] = mGridData[ind[0][1][0]];
      cVals[6] = mGridData[ind[0][1][1]];
      cVals[7] = mGridData[ind[0][1][2]];
      cVals[8] = extrapolation(mGridData[ind[0][2][0]], mGridData[ind[0][2][1]], mGridData[ind[0][2][2]]);
      cVals[9] = mGridData[ind[0][2][0]];
      cVals[10] = mGridData[ind[0][2][1]];
      cVals[11] = mGridData[ind[0][2][2]];
      cVals[12] = extrapolation(mGridData[ind[0][3][0]], mGridData[ind[0][3][1]], mGridData[ind[0][3][2]]);
      cVals[13] = mGridData[ind[0][3][0]];
      cVals[14] = mGridData[ind[0][3][1]];
      cVals[15] = mGridData[ind[0][3][2]];
      cVals[16] = extrapolation(mGridData[ind[1][0][0]], mGridData[ind[1][0][1]], mGridData[ind[1][0][2]]);
      cVals[17] = mGridData[ind[1][0][0]];
      cVals[18] = mGridData[ind[1][0][1]];
      cVals[19] = mGridData[ind[1][0][2]];
      cVals[20] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][1][1]], mGridData[ind[1][1][2]]);
      cVals[22] = mGridData[ind[1][1][1]];
      cVals[23] = mGridData[ind[1][1][2]];
      cVals[24] = extrapolation(mGridData[ind[1][2][0]], mGridData[ind[1][2][1]], mGridData[ind[1][2][2]]);
      cVals[25] = mGridData[ind[1][2][0]];
      cVals[26] = mGridData[ind[1][2][1]];
      cVals[27] = mGridData[ind[1][2][2]];
      cVals[28] = extrapolation(mGridData[ind[1][3][0]], mGridData[ind[1][3][1]], mGridData[ind[1][3][2]]);
      cVals[29] = mGridData[ind[1][3][0]];
      cVals[30] = mGridData[ind[1][3][1]];
      cVals[31] = mGridData[ind[1][3][2]];
      cVals[32] = extrapolation(mGridData[ind[2][0][0]], mGridData[ind[2][0][1]], mGridData[ind[2][0][2]]);
      cVals[33] = mGridData[ind[2][0][0]];
      cVals[34] = mGridData[ind[2][0][1]];
      cVals[35] = mGridData[ind[2][0][2]];
      cVals[36] = extrapolation(mGridData[ind[2][1][0]], mGridData[ind[2][1][1]], mGridData[ind[2][1][2]]);
      cVals[37] = mGridData[ind[2][1][0]];
      cVals[38] = mGridData[ind[2][1][1]];
      cVals[39] = mGridData[ind[2][1][2]];
      cVals[40] = extrapolation(mGridData[ind[2][2][0]], mGridData[ind[2][2][1]], mGridData[ind[2][2][2]]);
      cVals[41] = mGridData[ind[2][2][0]];
      cVals[42] = mGridData[ind[2][2][1]];
      cVals[43] = mGridData[ind[2][2][2]];
      cVals[44] = extrapolation(mGridData[ind[2][3][0]], mGridData[ind[2][3][1]], mGridData[ind[2][3][2]]);
      cVals[45] = mGridData[ind[2][3][0]];
      cVals[46] = mGridData[ind[2][3][1]];
      cVals[47] = mGridData[ind[2][3][2]];
      cVals[48] = extrapolation(mGridData[ind[3][0][0]], mGridData[ind[3][0][1]], mGridData[ind[3][0][2]]);
      cVals[49] = mGridData[ind[3][0][0]];
      cVals[50] = mGridData[ind[3][0][1]];
      cVals[51] = mGridData[ind[3][0][2]];
      cVals[52] = extrapolation(mGridData[ind[3][1][0]], mGridData[ind[3][1][1]], mGridData[ind[3][1][2]]);
      cVals[53] = mGridData[ind[3][1][0]];
      cVals[54] = mGridData[ind[3][1][1]];
      cVals[55] = mGridData[ind[3][1][2]];
      cVals[56] = extrapolation(mGridData[ind[3][2][0]], mGridData[ind[3][2][1]], mGridData[ind[3][2][2]]);
      cVals[57] = mGridData[ind[3][2][0]];
      cVals[58] = mGridData[ind[3][2][1]];
      cVals[59] = mGridData[ind[3][2][2]];
      cVals[60] = extrapolation(mGridData[ind[3][3][0]], mGridData[ind[3][3][1]], mGridData[ind[3][3][2]]);
      cVals[61] = mGridData[ind[3][3][0]];
      cVals[62] = mGridData[ind[3][3][1]];
      cVals[63] = mGridData[ind[3][3][2]];
    } break;

    case GridPos::Edge0:
    case GridPos::Edge4:
    case GridPos::LineI: {
      const int ind[4][3][3]{
        {{ii_x_y_z + deltaPhi[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0]}},
        {{ii_x_y_z, ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0]}}};

      cVals[0] = extrapolation(mGridData[ind[0][0][0]], mGridData[ind[0][1][1]], mGridData[ind[0][2][2]]);
      cVals[1] = extrapolation(mGridData[ind[0][0][0]], mGridData[ind[0][1][0]], mGridData[ind[0][2][0]]);
      cVals[2] = extrapolation(mGridData[ind[0][0][1]], mGridData[ind[0][1][1]], mGridData[ind[0][2][1]]);
      cVals[3] = extrapolation(mGridData[ind[0][0][2]], mGridData[ind[0][1][2]], mGridData[ind[0][2][2]]);
      cVals[4] = extrapolation(mGridData[ind[0][0][0]], mGridData[ind[0][0][1]], mGridData[ind[0][0][2]]);
      cVals[5] = mGridData[ind[0][0][0]];
      cVals[6] = mGridData[ind[0][0][1]];
      cVals[7] = mGridData[ind[0][0][2]];
      cVals[8] = extrapolation(mGridData[ind[0][1][0]], mGridData[ind[0][1][1]], mGridData[ind[0][1][2]]);
      cVals[9] = mGridData[ind[0][1][0]];
      cVals[10] = mGridData[ind[0][1][1]];
      cVals[11] = mGridData[ind[0][1][2]];
      cVals[12] = extrapolation(mGridData[ind[0][2][0]], mGridData[ind[0][2][1]], mGridData[ind[0][2][2]]);
      cVals[13] = mGridData[ind[0][2][0]];
      cVals[14] = mGridData[ind[0][2][1]];
      cVals[15] = mGridData[ind[0][2][2]];
      cVals[16] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][1][1]], mGridData[ind[1][2][2]]);
      cVals[17] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][1][0]], mGridData[ind[1][2][0]]);
      cVals[18] = extrapolation(mGridData[ind[1][0][1]], mGridData[ind[1][1][1]], mGridData[ind[1][2][1]]);
      cVals[19] = extrapolation(mGridData[ind[1][0][2]], mGridData[ind[1][1][2]], mGridData[ind[1][2][2]]);
      cVals[20] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][0][1]], mGridData[ind[1][0][2]]);
      cVals[22] = mGridData[ind[1][0][1]];
      cVals[23] = mGridData[ind[1][0][2]];
      cVals[24] = extrapolation(mGridData[ind[1][1][0]], mGridData[ind[1][1][1]], mGridData[ind[1][1][2]]);
      cVals[25] = mGridData[ind[1][1][0]];
      cVals[26] = mGridData[ind[1][1][1]];
      cVals[27] = mGridData[ind[1][1][2]];
      cVals[28] = extrapolation(mGridData[ind[1][2][0]], mGridData[ind[1][2][1]], mGridData[ind[1][2][2]]);
      cVals[29] = mGridData[ind[1][2][0]];
      cVals[30] = mGridData[ind[1][2][1]];
      cVals[31] = mGridData[ind[1][2][2]];
      cVals[32] = extrapolation(mGridData[ind[2][0][0]], mGridData[ind[2][1][1]], mGridData[ind[2][2][2]]);
      cVals[33] = extrapolation(mGridData[ind[2][0][0]], mGridData[ind[2][1][0]], mGridData[ind[2][2][0]]);
      cVals[34] = extrapolation(mGridData[ind[2][0][1]], mGridData[ind[2][1][1]], mGridData[ind[2][2][1]]);
      cVals[35] = extrapolation(mGridData[ind[2][0][2]], mGridData[ind[2][1][2]], mGridData[ind[2][2][2]]);
      cVals[36] = extrapolation(mGridData[ind[2][0][0]], mGridData[ind[2][0][1]], mGridData[ind[2][0][2]]);
      cVals[37] = mGridData[ind[2][0][0]];
      cVals[38] = mGridData[ind[2][0][1]];
      cVals[39] = mGridData[ind[2][0][2]];
      cVals[40] = extrapolation(mGridData[ind[2][1][0]], mGridData[ind[2][1][1]], mGridData[ind[2][1][2]]);
      cVals[41] = mGridData[ind[2][1][0]];
      cVals[42] = mGridData[ind[2][1][1]];
      cVals[43] = mGridData[ind[2][1][2]];
      cVals[44] = extrapolation(mGridData[ind[2][2][0]], mGridData[ind[2][2][1]], mGridData[ind[2][2][2]]);
      cVals[45] = mGridData[ind[2][2][0]];
      cVals[46] = mGridData[ind[2][2][1]];
      cVals[47] = mGridData[ind[2][2][2]];
      cVals[48] = extrapolation(mGridData[ind[3][0][0]], mGridData[ind[3][1][1]], mGridData[ind[3][2][2]]);
      cVals[49] = extrapolation(mGridData[ind[3][0][0]], mGridData[ind[3][1][0]], mGridData[ind[3][2][0]]);
      cVals[50] = extrapolation(mGridData[ind[3][0][1]], mGridData[ind[3][1][1]], mGridData[ind[3][2][1]]);
      cVals[51] = extrapolation(mGridData[ind[3][0][2]], mGridData[ind[3][1][2]], mGridData[ind[3][2][2]]);
      cVals[52] = extrapolation(mGridData[ind[3][0][0]], mGridData[ind[3][0][1]], mGridData[ind[3][0][2]]);
      cVals[53] = mGridData[ind[3][0][0]];
      cVals[54] = mGridData[ind[3][0][1]];
      cVals[55] = mGridData[ind[3][0][2]];
      cVals[56] = extrapolation(mGridData[ind[3][1][0]], mGridData[ind[3][1][1]], mGridData[ind[3][1][2]]);
      cVals[57] = mGridData[ind[3][1][0]];
      cVals[58] = mGridData[ind[3][1][1]];
      cVals[59] = mGridData[ind[3][1][2]];
      cVals[60] = extrapolation(mGridData[ind[3][2][0]], mGridData[ind[3][2][1]], mGridData[ind[3][2][2]]);
      cVals[61] = mGridData[ind[3][2][0]];
      cVals[62] = mGridData[ind[3][2][1]];
      cVals[63] = mGridData[ind[3][2][2]];
    } break;

    case GridPos::Edge1:
    case GridPos::Edge5:
    case GridPos::LineJ: {
      const int ind[4][3][3]{
        {{ii_x_y_z + deltaPhi[i0] + deltaZ[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaZ[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaZ[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaZ[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0]}}};

      cVals[0] = extrapolation(mGridData[ind[0][0][0]], mGridData[ind[0][1][0]], mGridData[ind[0][2][0]]);
      cVals[1] = extrapolation(mGridData[ind[0][0][1]], mGridData[ind[0][1][1]], mGridData[ind[0][2][1]]);
      cVals[2] = extrapolation(mGridData[ind[0][0][1]], mGridData[ind[0][1][0]], mGridData[ind[0][2][0] + deltaZ[i0]]);
      cVals[3] = extrapolation(mGridData[ind[0][0][2]], mGridData[ind[0][1][1]], mGridData[ind[0][2][0]]);
      cVals[4] = mGridData[ind[0][0][0]];
      cVals[5] = mGridData[ind[0][0][1]];
      cVals[6] = mGridData[ind[0][0][2]];
      cVals[7] = extrapolation(mGridData[ind[0][0][2]], mGridData[ind[0][0][1]], mGridData[ind[0][0][0]]);
      cVals[8] = mGridData[ind[0][1][0]];
      cVals[9] = mGridData[ind[0][1][1]];
      cVals[10] = mGridData[ind[0][1][2]];
      cVals[11] = extrapolation(mGridData[ind[0][1][2]], mGridData[ind[0][1][1]], mGridData[ind[0][1][0]]);
      cVals[12] = mGridData[ind[0][2][0]];
      cVals[13] = mGridData[ind[0][2][1]];
      cVals[14] = mGridData[ind[0][2][2]];
      cVals[15] = extrapolation(mGridData[ind[0][2][2]], mGridData[ind[0][2][1]], mGridData[ind[0][2][0]]);
      cVals[16] = extrapolation(mGridData[ind[1][0][0]], mGridData[ind[1][1][0]], mGridData[ind[1][2][0]]);
      cVals[17] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][1][1]], mGridData[ind[1][2][1]]);
      cVals[18] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][1][0]], mGridData[ind[1][2][0] + deltaZ[i0]]);
      cVals[19] = extrapolation(mGridData[ind[1][0][2]], mGridData[ind[1][1][1]], mGridData[ind[1][2][0]]);
      cVals[20] = mGridData[ind[1][0][0]];
      cVals[22] = mGridData[ind[1][0][2]];
      cVals[23] = extrapolation(mGridData[ind[1][0][2]], mGridData[ii_x_y_z], mGridData[ind[1][0][0]]);
      cVals[24] = mGridData[ind[1][1][0]];
      cVals[25] = mGridData[ind[1][1][1]];
      cVals[26] = mGridData[ind[1][1][2]];
      cVals[27] = extrapolation(mGridData[ind[1][1][2]], mGridData[ind[1][1][1]], mGridData[ind[1][1][0]]);
      cVals[28] = mGridData[ind[1][2][0]];
      cVals[29] = mGridData[ind[1][2][1]];
      cVals[30] = mGridData[ind[1][2][2]];
      cVals[31] = extrapolation(mGridData[ind[1][2][2]], mGridData[ind[1][2][1]], mGridData[ind[1][2][0]]);
      cVals[32] = extrapolation(mGridData[ind[2][0][0]], mGridData[ind[2][1][0]], mGridData[ind[2][2][0]]);
      cVals[33] = extrapolation(mGridData[ind[2][0][1]], mGridData[ind[2][1][1]], mGridData[ind[2][2][1]]);
      cVals[34] = extrapolation(mGridData[ind[2][0][1]], mGridData[ind[2][1][0]], mGridData[ind[2][2][0] + deltaZ[i0]]);
      cVals[35] = extrapolation(mGridData[ind[2][0][2]], mGridData[ind[2][1][1]], mGridData[ind[2][2][0]]);
      cVals[36] = mGridData[ind[2][0][0]];
      cVals[37] = mGridData[ind[2][0][1]];
      cVals[38] = mGridData[ind[2][0][2]];
      cVals[39] = extrapolation(mGridData[ind[2][0][2]], mGridData[ind[2][0][1]], mGridData[ind[2][0][0]]);
      cVals[40] = mGridData[ind[2][1][0]];
      cVals[41] = mGridData[ind[2][1][1]];
      cVals[42] = mGridData[ind[2][1][2]];
      cVals[43] = extrapolation(mGridData[ind[2][1][2]], mGridData[ind[2][1][1]], mGridData[ind[2][1][0]]);
      cVals[44] = mGridData[ind[2][2][0]];
      cVals[45] = mGridData[ind[2][2][1]];
      cVals[46] = mGridData[ind[2][2][2]];
      cVals[47] = extrapolation(mGridData[ind[2][2][2]], mGridData[ind[2][2][1]], mGridData[ind[2][2][0]]);
      cVals[48] = extrapolation(mGridData[ind[3][0][0]], mGridData[ind[3][1][0]], mGridData[ind[3][2][0]]);
      cVals[49] = extrapolation(mGridData[ind[3][0][1]], mGridData[ind[3][1][1]], mGridData[ind[3][2][1]]);
      cVals[50] = extrapolation(mGridData[ind[3][0][1]], mGridData[ind[3][1][0]], mGridData[ind[3][2][0] + deltaZ[i0]]);
      cVals[51] = extrapolation(mGridData[ind[3][0][2]], mGridData[ind[3][1][1]], mGridData[ind[3][2][0]]);
      cVals[52] = mGridData[ind[3][0][0]];
      cVals[53] = mGridData[ind[3][0][1]];
      cVals[54] = mGridData[ind[3][0][2]];
      cVals[55] = extrapolation(mGridData[ind[3][0][2]], mGridData[ind[3][0][1]], mGridData[ind[3][0][0]]);
      cVals[56] = mGridData[ind[3][1][0]];
      cVals[57] = mGridData[ind[3][1][1]];
      cVals[58] = mGridData[ind[3][1][2]];
      cVals[59] = extrapolation(mGridData[ind[3][1][2]], mGridData[ind[3][1][1]], mGridData[ind[3][1][0]]);
      cVals[60] = mGridData[ind[3][2][0]];
      cVals[61] = mGridData[ind[3][2][1]];
      cVals[62] = mGridData[ind[3][2][2]];
      cVals[63] = extrapolation(mGridData[ind[3][2][2]], mGridData[ind[3][2][1]], mGridData[ind[3][2][0]]);
    } break;

    case GridPos::Edge2:
    case GridPos::Edge6:
    case GridPos::LineK: {
      const int ind[4][3][3]{
        {{ii_x_y_z + deltaPhi[i0] + deltaR[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaR[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaR[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaR[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0]}}};

      cVals[0] = extrapolation(mGridData[ind[0][0][0]], mGridData[ind[0][0][1]], mGridData[ind[0][0][2]]);
      cVals[1] = mGridData[ind[0][0][0]];
      cVals[2] = mGridData[ind[0][0][1]];
      cVals[3] = mGridData[ind[0][0][2]];
      cVals[4] = extrapolation(mGridData[ind[0][1][0]], mGridData[ind[0][1][1]], mGridData[ind[0][1][2]]);
      cVals[5] = mGridData[ind[0][1][0]];
      cVals[6] = mGridData[ind[0][1][1]];
      cVals[7] = mGridData[ind[0][1][2]];
      cVals[8] = extrapolation(mGridData[ind[0][1][0]], mGridData[ind[0][0][1]], mGridData[ind[0][0][2] + deltaR[i0]]);
      cVals[9] = mGridData[ind[0][2][0]];
      cVals[10] = mGridData[ind[0][2][1]];
      cVals[11] = mGridData[ind[0][2][2]];
      cVals[12] = extrapolation(mGridData[ind[0][2][0]], mGridData[ind[0][1][1]], mGridData[ind[0][0][2]]);
      cVals[13] = extrapolation(mGridData[ind[0][2][0]], mGridData[ind[0][1][0]], mGridData[ind[0][0][0]]);
      cVals[14] = extrapolation(mGridData[ind[0][2][1]], mGridData[ind[0][1][1]], mGridData[ind[0][0][1]]);
      cVals[15] = extrapolation(mGridData[ind[0][2][2]], mGridData[ind[0][1][2]], mGridData[ind[0][0][2]]);
      cVals[16] = extrapolation(mGridData[ind[1][0][0]], mGridData[ind[1][0][1]], mGridData[ind[1][0][2]]);
      cVals[17] = mGridData[ind[1][0][0]];
      cVals[18] = mGridData[ind[1][0][1]];
      cVals[19] = mGridData[ind[1][0][2]];
      cVals[20] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][1][1]], mGridData[ind[1][1][2]]);
      cVals[22] = mGridData[ind[1][1][1]];
      cVals[23] = mGridData[ind[1][1][2]];
      cVals[24] = extrapolation(mGridData[ii_x_y_z], mGridData[ind[1][0][1]], mGridData[ind[1][0][2] + deltaR[i0]]);
      cVals[25] = mGridData[ind[1][2][0]];
      cVals[26] = mGridData[ind[1][2][1]];
      cVals[27] = mGridData[ind[1][2][2]];
      cVals[28] = extrapolation(mGridData[ind[1][2][0]], mGridData[ind[1][1][1]], mGridData[ind[1][0][2]]);
      cVals[29] = extrapolation(mGridData[ind[1][2][0]], mGridData[ii_x_y_z], mGridData[ind[1][0][0]]);
      cVals[30] = extrapolation(mGridData[ind[1][2][1]], mGridData[ind[1][1][1]], mGridData[ind[1][0][1]]);
      cVals[32] = extrapolation(mGridData[ind[2][0][0]], mGridData[ind[2][0][1]], mGridData[ind[2][0][2]]);
      cVals[31] = extrapolation(mGridData[ind[1][2][2]], mGridData[ind[1][1][2]], mGridData[ind[1][0][2]]);
      cVals[33] = mGridData[ind[2][0][0]];
      cVals[34] = mGridData[ind[2][0][1]];
      cVals[35] = mGridData[ind[2][0][2]];
      cVals[36] = extrapolation(mGridData[ind[2][1][0]], mGridData[ind[2][1][1]], mGridData[ind[2][1][2]]);
      cVals[37] = mGridData[ind[2][1][0]];
      cVals[38] = mGridData[ind[2][1][1]];
      cVals[39] = mGridData[ind[2][1][2]];
      cVals[40] = extrapolation(mGridData[ind[2][1][0]], mGridData[ind[2][0][1]], mGridData[ind[2][0][2] + deltaR[i0]]);
      cVals[41] = mGridData[ind[2][2][0]];
      cVals[42] = mGridData[ind[2][2][1]];
      cVals[43] = mGridData[ind[2][2][2]];
      cVals[44] = extrapolation(mGridData[ind[2][2][0]], mGridData[ind[2][1][1]], mGridData[ind[2][0][2]]);
      cVals[45] = extrapolation(mGridData[ind[2][2][0]], mGridData[ind[2][1][0]], mGridData[ind[2][0][0]]);
      cVals[46] = extrapolation(mGridData[ind[2][2][1]], mGridData[ind[2][1][1]], mGridData[ind[2][0][1]]);
      cVals[47] = extrapolation(mGridData[ind[2][2][2]], mGridData[ind[2][1][2]], mGridData[ind[2][0][2]]);
      cVals[48] = extrapolation(mGridData[ind[3][0][0]], mGridData[ind[3][0][1]], mGridData[ind[3][0][2]]);
      cVals[49] = mGridData[ind[3][0][0]];
      cVals[50] = mGridData[ind[3][0][1]];
      cVals[51] = mGridData[ind[3][0][2]];
      cVals[52] = extrapolation(mGridData[ind[3][1][0]], mGridData[ind[3][1][1]], mGridData[ind[3][1][2]]);
      cVals[53] = mGridData[ind[3][1][0]];
      cVals[54] = mGridData[ind[3][1][1]];
      cVals[55] = mGridData[ind[3][1][2]];
      cVals[56] = extrapolation(mGridData[ind[3][1][0]], mGridData[ind[3][0][1]], mGridData[ind[3][0][2] + deltaR[i0]]);
      cVals[57] = mGridData[ind[3][2][0]];
      cVals[58] = mGridData[ind[3][2][1]];
      cVals[59] = mGridData[ind[3][2][2]];
      cVals[60] = extrapolation(mGridData[ind[3][2][0]], mGridData[ind[3][1][1]], mGridData[ind[3][0][2]]);
      cVals[61] = extrapolation(mGridData[ind[3][2][0]], mGridData[ind[3][1][0]], mGridData[ind[3][0][0]]);
      cVals[62] = extrapolation(mGridData[ind[3][2][1]], mGridData[ind[3][1][1]], mGridData[ind[3][0][1]]);
      cVals[63] = extrapolation(mGridData[ind[3][2][2]], mGridData[ind[3][1][2]], mGridData[ind[3][0][2]]);
    } break;

    case GridPos::Edge3:
    case GridPos::Edge7:
    case GridPos::LineL: {
      const int ind[4][3][3]{
        {{ii_x_y_z + deltaPhi[i0] + deltaR[i0] + deltaZ[i0], ind[0][0][0] - deltaZ[i0], ind[0][0][1] - deltaZ[i0]},
         {ind[0][0][0] - deltaR[i0], ind[0][1][0] - deltaZ[i0], ind[0][1][1] - deltaZ[i0]},
         {ind[0][1][0] - deltaR[i0], ind[0][2][0] - deltaZ[i0], ind[0][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaR[i0] + deltaZ[i0], ind[1][0][0] - deltaZ[i0], ind[1][0][1] - deltaZ[i0]},
         {ind[1][0][0] - deltaR[i0], ind[1][1][0] - deltaZ[i0], ind[1][1][1] - deltaZ[i0]},
         {ind[1][1][0] - deltaR[i0], ind[1][2][0] - deltaZ[i0], ind[1][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i1] + deltaR[i0] + deltaZ[i0], ind[2][0][0] - deltaZ[i0], ind[2][0][1] - deltaZ[i0]},
         {ind[2][0][0] - deltaR[i0], ind[2][1][0] - deltaZ[i0], ind[2][1][1] - deltaZ[i0]},
         {ind[2][1][0] - deltaR[i0], ind[2][2][0] - deltaZ[i0], ind[2][2][1] - deltaZ[i0]}},
        {{ii_x_y_z + deltaPhi[i2] + deltaR[i0] + deltaZ[i0], ind[3][0][0] - deltaZ[i0], ind[3][0][1] - deltaZ[i0]},
         {ind[3][0][0] - deltaR[i0], ind[3][1][0] - deltaZ[i0], ind[3][1][1] - deltaZ[i0]},
         {ind[3][1][0] - deltaR[i0], ind[3][2][0] - deltaZ[i0], ind[3][2][1] - deltaZ[i0]}}};

      cVals[0] = mGridData[ind[0][0][0]];
      cVals[1] = mGridData[ind[0][0][1]];
      cVals[2] = mGridData[ind[0][0][2]];
      cVals[3] = extrapolation(mGridData[ind[0][0][2]], mGridData[ind[0][0][1]], mGridData[ind[0][0][0]]);
      cVals[4] = mGridData[ind[0][1][0]];
      cVals[5] = mGridData[ind[0][1][1]];
      cVals[6] = mGridData[ind[0][1][2]];
      cVals[7] = extrapolation(mGridData[ind[0][1][2]], mGridData[ind[0][1][1]], mGridData[ind[0][1][0]]);
      cVals[8] = mGridData[ind[0][2][0]];
      cVals[9] = mGridData[ind[0][2][1]];
      cVals[10] = mGridData[ind[0][2][2]];
      cVals[11] = extrapolation(mGridData[ind[0][2][2]], mGridData[ind[0][2][1]], mGridData[ind[0][2][0]]);
      cVals[12] = extrapolation(mGridData[ind[0][2][0]], mGridData[ind[0][1][0]], mGridData[ind[0][0][0]]);
      cVals[13] = extrapolation(mGridData[ind[0][2][1]], mGridData[ind[0][1][1]], mGridData[ind[0][0][1]]);
      cVals[14] = extrapolation(mGridData[ind[0][2][2]], mGridData[ind[0][1][2]], mGridData[ind[0][0][2]]);
      cVals[15] = extrapolation(mGridData[ind[0][2][2]], mGridData[ind[0][1][1]], mGridData[ind[0][0][0]]);
      cVals[16] = mGridData[ind[1][0][0]];
      cVals[17] = mGridData[ind[1][0][1]];
      cVals[18] = mGridData[ind[1][0][2]];
      cVals[19] = extrapolation(mGridData[ind[1][0][2]], mGridData[ind[1][0][1]], mGridData[ind[1][0][0]]);
      cVals[20] = mGridData[ind[1][1][0]];
      cVals[22] = mGridData[ind[1][1][2]];
      cVals[23] = extrapolation(mGridData[ind[1][1][2]], mGridData[ii_x_y_z], mGridData[ind[1][1][0]]);
      cVals[24] = mGridData[ind[1][2][0]];
      cVals[25] = mGridData[ind[1][2][1]];
      cVals[26] = mGridData[ind[1][2][2]];
      cVals[27] = extrapolation(mGridData[ind[1][2][2]], mGridData[ind[1][2][1]], mGridData[ind[1][2][0]]);
      cVals[28] = extrapolation(mGridData[ind[1][2][0]], mGridData[ind[1][1][0]], mGridData[ind[1][0][0]]);
      cVals[29] = extrapolation(mGridData[ind[1][2][1]], mGridData[ii_x_y_z], mGridData[ind[1][0][1]]);
      cVals[30] = extrapolation(mGridData[ind[1][2][2]], mGridData[ind[1][1][2]], mGridData[ind[1][0][2]]);
      cVals[31] = extrapolation(mGridData[ind[1][2][2]], mGridData[ind[1][1][1]], mGridData[ind[1][0][0]]);
      cVals[32] = mGridData[ind[2][0][0]];
      cVals[33] = mGridData[ind[2][0][1]];
      cVals[34] = mGridData[ind[2][0][2]];
      cVals[35] = extrapolation(mGridData[ind[2][0][2]], mGridData[ind[2][0][1]], mGridData[ind[2][0][0]]);
      cVals[36] = mGridData[ind[2][1][0]];
      cVals[37] = mGridData[ind[2][1][1]];
      cVals[38] = mGridData[ind[2][1][2]];
      cVals[39] = extrapolation(mGridData[ind[2][1][2]], mGridData[ind[2][1][1]], mGridData[ind[2][1][0]]);
      cVals[40] = mGridData[ind[2][2][0]];
      cVals[41] = mGridData[ind[2][2][1]];
      cVals[42] = mGridData[ind[2][2][2]];
      cVals[43] = extrapolation(mGridData[ind[2][2][2]], mGridData[ind[2][2][1]], mGridData[ind[2][2][0]]);
      cVals[44] = extrapolation(mGridData[ind[2][2][0]], mGridData[ind[2][1][0]], mGridData[ind[2][0][0]]);
      cVals[45] = extrapolation(mGridData[ind[2][2][1]], mGridData[ind[2][1][1]], mGridData[ind[2][0][1]]);
      cVals[46] = extrapolation(mGridData[ind[2][2][2]], mGridData[ind[2][1][2]], mGridData[ind[2][0][2]]);
      cVals[47] = extrapolation(mGridData[ind[2][2][2]], mGridData[ind[2][1][1]], mGridData[ind[2][0][0]]);
      cVals[48] = mGridData[ind[3][0][0]];
      cVals[49] = mGridData[ind[3][0][1]];
      cVals[50] = mGridData[ind[3][0][2]];
      cVals[51] = extrapolation(mGridData[ind[3][0][2]], mGridData[ind[3][0][1]], mGridData[ind[3][0][0]]);
      cVals[52] = mGridData[ind[3][1][0]];
      cVals[53] = mGridData[ind[3][1][1]];
      cVals[54] = mGridData[ind[3][1][2]];
      cVals[55] = extrapolation(mGridData[ind[3][1][2]], mGridData[ind[3][1][1]], mGridData[ind[3][1][0]]);
      cVals[56] = mGridData[ind[3][2][0]];
      cVals[57] = mGridData[ind[3][2][1]];
      cVals[58] = mGridData[ind[3][2][2]];
      cVals[59] = extrapolation(mGridData[ind[3][2][2]], mGridData[ind[3][2][1]], mGridData[ind[3][2][0]]);
      cVals[60] = extrapolation(mGridData[ind[3][2][0]], mGridData[ind[3][1][0]], mGridData[ind[3][0][0]]);
      cVals[61] = extrapolation(mGridData[ind[3][2][1]], mGridData[ind[3][1][1]], mGridData[ind[3][0][1]]);
      cVals[62] = extrapolation(mGridData[ind[3][2][2]], mGridData[ind[3][1][2]], mGridData[ind[3][0][2]]);
      cVals[63] = extrapolation(mGridData[ind[3][2][2]], mGridData[ind[3][1][1]], mGridData[ind[3][0][0]]);
    } break;
  }
}

} // namespace tpc
} // namespace o2

#endif
