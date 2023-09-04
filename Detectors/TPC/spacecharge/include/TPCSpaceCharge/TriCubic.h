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

/// \file  TriCubic.h
/// \brief Definition of TriCubic class
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_TRICUBIC_H_
#define ALICEO2_TPC_TRICUBIC_H_

#include "TPCSpaceCharge/RegularGrid3D.h"

// forward declare VC Memory
template <typename DataT, size_t, size_t, bool>
class Memory;

namespace o2
{
namespace tpc
{

template <typename DataT>
class DataContainer3D;

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

template <typename DataT, size_t N>
class Vector;

/// \tparam DataT the type of data which is used during the calculations
template <typename DataT = double>
class TriCubicInterpolator
{
  using Grid3D = RegularGrid3D<DataT>;
  using DataContainer = DataContainer3D<DataT>;

 public:
  /// Constructor for a tricubic interpolator
  /// \param gridData struct containing access to the values of the grid
  /// \param gridProperties properties of the 3D grid
  TriCubicInterpolator(const DataContainer& gridData, const Grid3D& gridProperties) : mGridData{&gridData}, mGridProperties{&gridProperties} {};

  /// move constructor
  TriCubicInterpolator(TriCubicInterpolator<DataT>&&);

  /// move assignment
  TriCubicInterpolator<DataT>& operator=(TriCubicInterpolator<DataT>&&);

  enum class ExtrapolationType {
    Linear = 0,   ///< assume linear dependency at the boundaries of the grid
    Parabola = 1, ///< assume parabolic dependency at the boundaries of the grid
  };

  // interpolate value at given coordinate
  /// \param z z coordinate
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param type interpolation algorithm
  /// \return returns the interpolated value at given coordinate
  DataT operator()(const DataT z, const DataT r, const DataT phi) const { return interpolateSparse(z, r, phi); }

  /// set which type of extrapolation is used at the grid boundaries (linear or parabol can be used with periodic phi axis and non periodic z and r axis).
  /// \param extrapolationType sets type of extrapolation. See enum ExtrapolationType for different types
  void setExtrapolationType(const ExtrapolationType extrapolationType) { mExtrapolationType = extrapolationType; }

  /// \return returns the extrapolation technique for missing boundary values
  ExtrapolationType getExtrapolationType() const { return mExtrapolationType; }

  /// enable or disable extraolating values outside of the grid
  void setExtrapolateValues(const bool extraPolate) { mExtraPolateValues = extraPolate; }

  /// \return returns whethere values outside of the grid will be extrapolated
  bool getExtrapolateValues() const { return mExtraPolateValues; }

 private:
  static constexpr unsigned int FDim = Grid3D::getDim();              ///< dimensions of the grid
  static constexpr unsigned int FZ = Grid3D::getFZ();                 ///< index for z coordinate
  static constexpr unsigned int FR = Grid3D::getFR();                 ///< index for r coordinate
  static constexpr unsigned int FPHI = Grid3D::getFPhi();             ///< index for phi coordinate
  const DataContainer* mGridData{};                                   ///< adress to the data container of the grid
  const Grid3D* mGridProperties{};                                    ///< adress to the properties of the grid
  ExtrapolationType mExtrapolationType = ExtrapolationType::Parabola; ///< sets which type of extrapolation for missing points at boundary is used
  bool mExtraPolateValues = true;                                     ///< extrapolating values outside of the grid or restricting query to inside of the grid

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

  void setValues(const int iz, const int ir, const int iphi, std::array<Vector<DataT, 4>, 16>& cVals) const;

  // interpolate value at given coordinate - this method doesnt compute and stores the coefficients and is faster when quering only a few values per cube
  /// \param z z coordinate
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \return returns the interpolated value at given coordinate
  DataT interpolateSparse(const DataT z, const DataT r, const DataT phi) const;

  // for periodic boundary conditions
  void getDataIndexCircularArray(const int index0, const int dim, int arr[]) const;

  // this helps to get circular and non circular padding indices
  int getRegulatedDelta(const int index0, const int delta, const unsigned int dim, const int offs) const { return mGridProperties->isIndexInGrid(index0 + delta, dim) ? delta : offs; }

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

} // namespace tpc
} // namespace o2

#endif
