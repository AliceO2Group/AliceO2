// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace tpc
{

/// \class RegularGrid3D
/// This class implements basic properties of a regular 3D-Grid like the spacing for each dimension and min and max coordinates.

/// \tparam DataT the type of data which is used during the calculations
/// \tparam Nx number of vertices in x direction
/// \tparam Ny number of vertices in y direction
/// \tparam Nz number of vertices in z direction
template <typename DataT = double, unsigned int Nx = 129, unsigned int Ny = 129, unsigned int Nz = 180>
struct RegularGrid3D {

 public:
  RegularGrid3D(const DataT xmin, const DataT ymin, const DataT zmin, const DataT spacingX, const DataT spacingY, const DataT spacingZ) : mMin{{xmin, ymin, zmin}}, mMax{{xmin + (Nx - 1) * spacingX, ymin + (Ny - 1) * spacingY, zmin + (Nz - 1) * spacingZ}}, mSpacing{{spacingX, spacingY, spacingZ}}, mInvSpacing{{static_cast<DataT>(1 / spacingX), static_cast<DataT>(1 / spacingY), static_cast<DataT>(1 / spacingZ)}}
  {
    initLists();
  }

  /// \param deltaX delta x index
  /// \return returns the delta index (where the data is stored) for given deltaX
  int getDeltaXDataIndex(const int deltaX) const { return deltaX; }

  /// \param deltaY delta y index
  /// \return returns the delta index (where the data is stored) for given deltaY
  int getDeltaYDataIndex(const int deltaY) const { return Nx * deltaY; }

  /// \param deltaZ delta z index
  /// \return returns the delta index (where the data is stored) for given deltaZ
  int getDeltaZDataIndex(const int deltaZ) const { return deltaZ * Ny * Nx; }

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
  static constexpr size_t getN(unsigned int dim) { return sNdim[dim]; }
  static constexpr size_t getNX() { return sNdim[FX]; }
  static constexpr size_t getNY() { return sNdim[FY]; }
  static constexpr size_t getNZ() { return sNdim[FZ]; }

  static constexpr unsigned int getDim() { return FDIM; } /// \return returns number of dimensions of the grid (3)
  static constexpr unsigned int getFX() { return FX; }    /// \return returns the index for dimension x (0)
  static constexpr unsigned int getFY() { return FY; }    /// \return returns the index for dimension y (1)
  static constexpr unsigned int getFZ() { return FZ; }    /// \return returns the index for dimension z (2)

  const Vector<DataT, 3>& getGridMin() const { return mMin; } /// \return returns the minimum coordinates of the grid in all dimensions
  DataT getGridMinX() const { return mMin[FX]; }              /// \return returns the minimum coordinate of the grid in x dimension
  DataT getGridMinY() const { return mMin[FY]; }              /// \return returns the minimum coordinate of the grid in y dimension
  DataT getGridMinZ() const { return mMin[FZ]; }              /// \return returns the minimum coordinate of the grid in z dimension

  DataT getGridMaxX() const { return mMax[FX]; }
  DataT getGridMaxY() const { return mMax[FY]; }
  DataT getGridMaxZ() const { return mMax[FZ]; }

  /// \return returns the inversed spacing of the grid for all dimensions
  const Vector<DataT, 3>& getInvSpacing() const { return mInvSpacing; }
  DataT getInvSpacingX() const { return mInvSpacing[FX]; }
  DataT getInvSpacingY() const { return mInvSpacing[FY]; }
  DataT getInvSpacingZ() const { return mInvSpacing[FZ]; }

  DataT getSpacingX() const { return mSpacing[FX]; }
  DataT getSpacingY() const { return mSpacing[FY]; }
  DataT getSpacingZ() const { return mSpacing[FZ]; }

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

  void checkStability(Vector<DataT, 3>& relPos, const Vector<int, 3>& circular) const;

  /// \param vertexX in x dimension
  /// \return returns the x positon for given vertex
  DataT getXVertex(const size_t vertexX) const { return mXVertices[vertexX]; }

  /// \param vertexY in y dimension
  /// \return returns the y positon for given vertex
  DataT getYVertex(const size_t vertexY) const { return mYVertices[vertexY]; }

  /// \param vertexZ in z dimension
  /// \return returns the z positon for given vertex
  DataT getZVertex(const size_t vertexZ) const { return mZVertices[vertexZ]; }

  const Vector<DataT, 3>& getMaxIndices() const { return sMaxIndex; } /// get max indices for all dimensions

  DataT getMaxIndexX() const { return sMaxIndex[0]; } /// get max index in x direction
  DataT getMaxIndexY() const { return sMaxIndex[1]; } /// get max index in y direction
  DataT getMaxIndexZ() const { return sMaxIndex[2]; } /// get max index in z direction

 private:
  static constexpr unsigned int FDIM = 3;                                      ///< dimensions of the grid (only 3 supported)
  static constexpr unsigned int FX = 0;                                        ///< index for x coordinate
  static constexpr unsigned int FY = 1;                                        ///< index for y coordinate
  static constexpr unsigned int FZ = 2;                                        ///< index for z coordinate
  const Vector<DataT, FDIM> mMin{};                                            ///< min vertices positions of the grid
  const Vector<DataT, FDIM> mMax{};                                            ///< max vertices positions of the grid
  const Vector<DataT, FDIM> mSpacing{};                                        ///<  spacing of the grid
  const Vector<DataT, FDIM> mInvSpacing{};                                     ///< inverse spacing of grid
  const inline static Vector<DataT, FDIM> sMaxIndex{{Nx - 1, Ny - 1, Nz - 1}}; ///< max index which is on the grid in all dimensions
  inline static Vector<int, FDIM> sNdim{{Nx, Ny, Nz}};                         ///< number of vertices for each dimension
  DataT mXVertices[Nx]{};                                                      ///< positions of vertices in x direction
  DataT mYVertices[Ny]{};                                                      ///< positions of vertices in y direction
  DataT mZVertices[Nz]{};                                                      ///< positions of vertices in z direction

  void initLists();

  ClassDefNV(RegularGrid3D, 1)
};

///
/// ========================================================================================================
///       Inline implementations of some methods
/// ========================================================================================================
///

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
DataT RegularGrid3D<DataT, Nx, Ny, Nz>::clampToGrid(const DataT pos, const unsigned int dim) const
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

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
DataT RegularGrid3D<DataT, Nx, Ny, Nz>::clampToGridRel(const DataT pos, const unsigned int dim) const
{
  if (pos < 0) {
    return 0;
  } else if (pos >= sMaxIndex[dim]) {
    return sMaxIndex[dim] - 1; // -1 return second last index. otherwise two additional points have to be extrapolated for tricubic interpolation
  }
  return pos;
}

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
DataT RegularGrid3D<DataT, Nx, Ny, Nz>::clampToGridCircular(DataT pos, const unsigned int dim) const
{
  while (pos < mMin[dim]) {
    pos += mMax[dim] - mMin[dim] + mSpacing[dim];
  }
  while (pos >= mMax[dim] + mSpacing[dim]) {
    pos -= mMax[dim] + mSpacing[dim] - mMin[dim];
  }
  return pos;
}

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
DataT RegularGrid3D<DataT, Nx, Ny, Nz>::clampToGridCircularRel(DataT pos, const unsigned int dim) const
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

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
void RegularGrid3D<DataT, Nx, Ny, Nz>::initLists()
{
  for (size_t i = 0; i < Nx; ++i) {
    mXVertices[i] = mMin[FX] + i * mSpacing[FX];
  }
  for (size_t i = 0; i < Ny; ++i) {
    mYVertices[i] = mMin[FY] + i * mSpacing[FY];
  }
  for (size_t i = 0; i < Nz; ++i) {
    mZVertices[i] = mMin[FZ] + i * mSpacing[FZ];
  }
}

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
int RegularGrid3D<DataT, Nx, Ny, Nz>::getDeltaDataIndex(const int delta, const int dim) const
{
  const unsigned int offset[FDIM]{1, Nx, Ny * Nx};
  const int deltaIndex = delta * offset[dim];
  return deltaIndex;
}

} // namespace tpc
} // namespace o2

#endif
