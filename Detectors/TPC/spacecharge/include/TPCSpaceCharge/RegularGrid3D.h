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

/// \file  RegularGrid3D.h
/// \brief Definition of RegularGrid3D class
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_REGULARGRID3D_H_
#define ALICEO2_TPC_REGULARGRID3D_H_

#include "TPCSpaceCharge/Vector.h"
#include "Rtypes.h" // for ClassDefNV
#include "TPCSpaceCharge/SpaceChargeParameter.h"
#include <vector>

namespace o2
{
namespace tpc
{

/// helper struct to store and set RegularGrid3D
template <typename DataT = double>
struct RegularGridHelper {
  DataT zmin;
  DataT rmin;
  DataT phimin;
  DataT spacingZ;
  DataT spacingR;
  DataT spacingPhi;
  ParamSpaceCharge params;
  ClassDefNV(RegularGridHelper, 1)
};

/// \class RegularGrid3D
/// This class implements basic properties of a regular 3D-Grid like the spacing for each dimension and min and max coordinates.
/// \tparam DataT the type of data which is used during the calculations
template <typename DataT = double>
struct RegularGrid3D {

 public:
  RegularGrid3D(const DataT zmin, const DataT rmin, const DataT phimin, const DataT spacingZ, const DataT spacingR, const DataT spacingPhi, const ParamSpaceCharge& params) : mParamGrid{params}, mMin{{zmin, rmin, phimin}}, mMax{{zmin + (mParamGrid.NZVertices - 1) * spacingZ, rmin + (mParamGrid.NRVertices - 1) * spacingR, phimin + (mParamGrid.NPhiVertices - 1) * spacingPhi}}, mSpacing{{spacingZ, spacingR, spacingPhi}}, mInvSpacing{{static_cast<DataT>(1 / spacingZ), static_cast<DataT>(1 / spacingR), static_cast<DataT>(1 / spacingPhi)}}
  {
    mRVertices.reserve(mParamGrid.NRVertices);
    mPhiVertices.reserve(mParamGrid.NPhiVertices);
    mZVertices.reserve(mParamGrid.NZVertices);
    initLists();
  }

  /// \param deltaX delta x index
  /// \return returns the delta index (where the data is stored) for given deltaX
  int getDeltaXDataIndex(const int deltaX) const { return deltaX; }

  /// \param deltaY delta y index
  /// \return returns the delta index (where the data is stored) for given deltaY
  int getDeltaYDataIndex(const int deltaY) const { return mParamGrid.NZVertices * deltaY; }

  /// \param deltaZ delta z index
  /// \return returns the delta index (where the data is stored) for given deltaZ
  int getDeltaZDataIndex(const int deltaZ) const { return deltaZ * mParamGrid.NRVertices * mParamGrid.NZVertices; }

  // same as above
  /// \param delta delta index
  /// \param dim dimension of interest
  /// \return returns the delta index (where the data is stored) for given delta and dim
  int getDeltaDataIndex(const int delta, const int dim) const;

  // check if the specified index for given dimension lies in the grid
  /// \param index query index
  /// \return returns if the index lies in the grid
  bool isIndexInGrid(const int index, const unsigned int dim) const { return index < 0 ? false : (index > (sNdim[dim] - 1) ? false : true); }

  /// \param dim dimension of interest
  /// \return returns the number of vertices for given dimension for the grid
  size_t getN(unsigned int dim) const { return sNdim[dim]; }
  size_t getNZ() const { return sNdim[FZ]; }
  size_t getNR() const { return sNdim[FR]; }
  size_t getNPhi() const { return sNdim[FPHI]; }

  static constexpr unsigned int getDim() { return FDIM; }  /// \return returns number of dimensions of the grid (3)
  static constexpr unsigned int getFZ() { return FZ; }     /// \return returns the index for dimension x (0)
  static constexpr unsigned int getFR() { return FR; }     /// \return returns the index for dimension y (1)
  static constexpr unsigned int getFPhi() { return FPHI; } /// \return returns the index for dimension z (2)

  const Vector<DataT, 3>& getGridMin() const { return mMin; } /// \return returns the minimum coordinates of the grid in all dimensions
  DataT getGridMinZ() const { return mMin[FZ]; }              /// \return returns the minimum coordinate of the grid in x dimension
  DataT getGridMinR() const { return mMin[FR]; }              /// \return returns the minimum coordinate of the grid in y dimension
  DataT getGridMinPhi() const { return mMin[FPHI]; }          /// \return returns the minimum coordinate of the grid in z dimension

  DataT getGridMaxZ() const { return mMax[FZ]; }
  DataT getGridMaxR() const { return mMax[FR]; }
  DataT getGridMaxPhi() const { return mMax[FPHI]; }

  /// \return returns the inversed spacing of the grid for all dimensions
  const Vector<DataT, 3>& getInvSpacing() const { return mInvSpacing; }
  DataT getInvSpacingZ() const { return mInvSpacing[FZ]; }
  DataT getInvSpacingR() const { return mInvSpacing[FR]; }
  DataT getInvSpacingPhi() const { return mInvSpacing[FPHI]; }

  DataT getSpacingZ() const { return mSpacing[FZ]; }
  DataT getSpacingR() const { return mSpacing[FR]; }
  DataT getSpacingPhi() const { return mSpacing[FPHI]; }

  // clamp coordinates to the grid (not circular)
  /// \param pos query position which will be clamped
  /// \return returns clamped coordinate coordinate
  DataT clampToGrid(const DataT pos, const unsigned int dim) const;

  // clamp coordinates to the grid (not circular)
  /// \param pos relative query position in grid which will be clamped
  /// \return returns clamped coordinate coordinate
  DataT clampToGridRel(const DataT pos, const unsigned int dim) const;

  // clamp coordinates to the grid circular
  /// \param pos query position which will be clamped
  /// \return returns clamped coordinate coordinate
  DataT clampToGridCircular(DataT pos, const unsigned int dim) const;

  // clamp coordinates to the grid circular
  /// \param pos relative query position in grid which will be clamped
  /// \return returns clamped coordinate coordinate
  DataT clampToGridCircularRel(DataT pos, const unsigned int dim) const;

  /// \param vertexX in x dimension
  /// \return returns the x positon for given vertex
  DataT getZVertex(const size_t vertexX) const { return mZVertices[vertexX]; }

  /// \param vertexY in y dimension
  /// \return returns the y positon for given vertex
  DataT getRVertex(const size_t vertexY) const { return mRVertices[vertexY]; }

  /// \param vertexZ in z dimension
  /// \return returns the z positon for given vertex
  DataT getPhiVertex(const size_t vertexZ) const { return mPhiVertices[vertexZ]; }

  const Vector<DataT, 3>& getMaxIndices() const { return sMaxIndex; } /// get max indices for all dimensions

  DataT getMaxIndexZ() const { return sMaxIndex[0]; }   /// get max index in x direction
  DataT getMaxIndexR() const { return sMaxIndex[1]; }   /// get max index in y direction
  DataT getMaxIndexPhi() const { return sMaxIndex[2]; } /// get max index in z direction

  /// returns grid parameters
  const auto& getParamSC() const { return mParamGrid; }

  unsigned short getRVertices() const { return mParamGrid.NRVertices; }
  unsigned short getZVertices() const { return mParamGrid.NZVertices; }
  unsigned short getPhiVertices() const { return mParamGrid.NPhiVertices; }

  /// \return returns helper struct containing the properties to create the object
  RegularGridHelper<double> getHelper() const { return RegularGridHelper<double>{getGridMinZ(), getGridMinR(), getGridMinPhi(), getSpacingZ(), getSpacingR(), getSpacingPhi(), mParamGrid}; }

 private:
  ParamSpaceCharge mParamGrid{};
  static constexpr unsigned int FDIM = 3;                                                                                                                                          ///< dimensions of the grid (only 3 supported)
  static constexpr unsigned int FZ = 0;                                                                                                                                            ///< index for x coordinate
  static constexpr unsigned int FR = 1;                                                                                                                                            ///< index for y coordinate
  static constexpr unsigned int FPHI = 2;                                                                                                                                          ///< index for z coordinate
  Vector<DataT, FDIM> mMin{};                                                                                                                                                      ///<! min vertices positions of the grid
  Vector<DataT, FDIM> mMax{};                                                                                                                                                      ///<! max vertices positions of the grid
  Vector<DataT, FDIM> mSpacing{};                                                                                                                                                  ///<! spacing of the grid
  Vector<DataT, FDIM> mInvSpacing{};                                                                                                                                               ///<! inverse spacing of grid
  Vector<DataT, FDIM> sMaxIndex{{static_cast<DataT>(mParamGrid.NZVertices - 1.), static_cast<DataT>(mParamGrid.NRVertices - 1), static_cast<DataT>(mParamGrid.NPhiVertices - 1)}}; ///<! max index which is on the grid in all dimensions
  Vector<int, FDIM> sNdim{{static_cast<int>(mParamGrid.NZVertices), static_cast<int>(mParamGrid.NRVertices), static_cast<int>(mParamGrid.NPhiVertices)}};                          ///<! number of vertices for each dimension
  std::vector<DataT> mZVertices{};                                                                                                                                                 ///< positions of vertices in x direction
  std::vector<DataT> mRVertices{};                                                                                                                                                 ///< positions of vertices in y direction
  std::vector<DataT> mPhiVertices{};                                                                                                                                               ///< positions of vertices in z direction

  void initLists();

  ClassDefNV(RegularGrid3D, 3)
};

///
/// ========================================================================================================
///       Inline implementations of some methods
/// ========================================================================================================
///

template <typename DataT>
DataT RegularGrid3D<DataT>::clampToGrid(const DataT pos, const unsigned int dim) const
{
  if (mMin[dim] < mMax[dim]) {
    if (pos < mMin[dim]) {
      return mMin[dim];
    } else if (pos > mMax[dim]) {
      return mMax[dim];
    }
  } else {
    if (pos > mMin[dim]) {
      return mMin[dim];
    } else if (pos < mMax[dim]) {
      return mMax[dim];
    }
  }
  return pos;
}

template <typename DataT>
DataT RegularGrid3D<DataT>::clampToGridRel(const DataT pos, const unsigned int dim) const
{
  if (pos < 0) {
    return 0;
  } else if (pos >= sMaxIndex[dim]) {
    return sMaxIndex[dim] - 1; // -1 return second last index. otherwise two additional points have to be extrapolated for tricubic interpolation
  }
  return pos;
}

template <typename DataT>
DataT RegularGrid3D<DataT>::clampToGridCircular(DataT pos, const unsigned int dim) const
{
  while (pos < mMin[dim]) {
    pos += mMax[dim] - mMin[dim] + mSpacing[dim];
  }
  while (pos >= mMax[dim] + mSpacing[dim]) {
    pos -= mMax[dim] + mSpacing[dim] - mMin[dim];
  }
  return pos;
}

template <typename DataT>
DataT RegularGrid3D<DataT>::clampToGridCircularRel(DataT pos, const unsigned int dim) const
{
  while (pos < 0) {
    pos += sNdim[dim];
  }
  while (pos > sNdim[dim]) {
    pos -= sNdim[dim];
  }
  if (pos == sNdim[dim]) {
    pos = 0;
  }
  return pos;
}

template <typename DataT>
void RegularGrid3D<DataT>::initLists()
{
  for (size_t i = 0; i < mParamGrid.NZVertices; ++i) {
    mZVertices.emplace_back(mMin[FZ] + i * mSpacing[FZ]);
  }
  for (size_t i = 0; i < mParamGrid.NRVertices; ++i) {
    mRVertices.emplace_back(mMin[FR] + i * mSpacing[FR]);
  }
  for (size_t i = 0; i < mParamGrid.NPhiVertices; ++i) {
    mPhiVertices.emplace_back(mMin[FPHI] + i * mSpacing[FPHI]);
  }
}

template <typename DataT>
int RegularGrid3D<DataT>::getDeltaDataIndex(const int delta, const int dim) const
{
  const int offset[FDIM]{1, mParamGrid.NZVertices, mParamGrid.NRVertices * mParamGrid.NZVertices};
  const int deltaIndex = delta * offset[dim];
  return deltaIndex;
}

} // namespace tpc
} // namespace o2

#endif
