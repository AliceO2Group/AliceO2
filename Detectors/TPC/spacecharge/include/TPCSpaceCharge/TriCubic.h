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

namespace o2
{
namespace tpc
{

template <typename DataT>
class DataContainer3D;

/// \class TriCubicInterpolator
/// The TriCubic class represents tricubic interpolation on a regular 3-Dim grid (z, r, phi).
/// The interpolation is separable: the 4 x 4 x 4 grid points around the query point are weighted with the 1D cubic (Catmull-Rom) weights in each dimension.
/// Periodic boundary conditions are used in phi direction. Grid points outside of the grid in r and z are extrapolated (linear or parabolic) from the next
/// three points towards the inside of the grid. Query points outside of the grid are extrapolated or, with setExtrapolateValues(false), restricted to the grid.
/// Several data containers on the same grid can be interpolated at the same position with interpolate(), which calculates the position in the grid and
/// the weights only once (e.g. for the three components of the electric field).

/// cell of the grid which is used for the tricubic interpolation of a query point and the weights of the interpolation (independent of the interpolated data)
template <typename DataT>
struct TriCubicCell {
  int indexZ{};          ///< index of the cell in z
  int indexR{};          ///< index of the cell in r
  int indexPhi{};        ///< index of the cell in phi
  bool inGrid{};         ///< all 64 points used for the interpolation are on the grid (no extrapolation needed)
  int dataIndex{};       ///< inGrid: index of the data at (iz - 1, ir - 1, iphi)
  int sliceOffset[4]{};  ///< inGrid: offsets of the data for the 4 phi slices
  int rowStride{};       ///< inGrid: offset of the data between the rows in r direction
  DataT weightsZ[4]{};   ///< weights of the interpolation in z
  DataT weightsR[4]{};   ///< weights of the interpolation in r
  DataT weightsPhi[4]{}; ///< weights of the interpolation in phi
};

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

  /// interpolate value at given coordinate
  /// \param z z coordinate
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \return returns the interpolated value at given coordinate
  DataT operator()(const DataT z, const DataT r, const DataT phi) const { return interpolateSparse(z, r, phi); }

  /// interpolate the data of this and of two other interpolators at the same coordinate (e.g. the three components of a vector field).
  /// The position in the grid and the weights of the interpolation are calculated only once, which is faster than three separate interpolations.
  /// If the data layout, the grid or the extrapolation settings of the other interpolators are different, the three values are interpolated separately.
  /// \param z z coordinate
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param other1 second interpolator
  /// \param other2 third interpolator
  /// \param val0 interpolated value of this interpolator (output)
  /// \param val1 interpolated value of other1 (output)
  /// \param val2 interpolated value of other2 (output)
  void interpolate(const DataT z, const DataT r, const DataT phi, const TriCubicInterpolator<DataT>& other1, const TriCubicInterpolator<DataT>& other2, DataT& val0, DataT& val1, DataT& val2) const;

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

  /// \return returns whether the other interpolator (with a different grid object) has the same data layout and grid (number of vertices, minimum and spacing)
  bool isSameGrid(const TriCubicInterpolator<DataT>& other) const;

  /// \return returns the cell and the weights of the interpolation for given coordinate
  TriCubicCell<DataT> getCell(const DataT z, const DataT r, const DataT phi) const;

  /// \return returns the interpolated value of the data for given cell (the data has to be defined on the grid of this interpolator)
  DataT interpolateData(const DataContainer& data, const TriCubicCell<DataT>& cell) const;

  /// set the 64 values (4 x 4 x 4) used for the interpolation in the cell with given indices. Values outside of the grid in r and z are extrapolated
  /// \param data data which is interpolated
  /// \param cVals values of the points: cVals[4 * iphi + ir][iz] with iphi, ir, iz the relative index (-1, 0, 1, 2) + 1 to the cell
  void setValues(const DataContainer& data, const int iz, const int ir, const int iphi, DataT (&cVals)[16][4]) const;

  /// interpolate value at given coordinate
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
};

} // namespace tpc
} // namespace o2

#endif
