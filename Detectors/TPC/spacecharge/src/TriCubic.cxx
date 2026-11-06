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

/// \file  TriCubic.cxx
/// \brief Definition of TriCubic class
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "TPCSpaceCharge/TriCubic.h"
#include "TPCSpaceCharge/DataContainer3D.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace o2::tpc;

/// default move constructor
template <typename DataT>
TriCubicInterpolator<DataT>::TriCubicInterpolator(TriCubicInterpolator<DataT>&& other)
{
  if (this != &other) {
    mExtrapolationType = other.mExtrapolationType;
    mExtraPolateValues = other.mExtraPolateValues;
    // do not change address of old pointers!
    other.mGridData = nullptr;
    other.mGridProperties = nullptr;
  }
}

template <typename DataT>
TriCubicInterpolator<DataT>& TriCubicInterpolator<DataT>::operator=(TriCubicInterpolator<DataT>&& other)
{
  if (this != &other) {
    mExtrapolationType = other.mExtrapolationType;
    mExtraPolateValues = other.mExtraPolateValues;
    // do not change address of old pointers!
    other.mGridData = nullptr;
    other.mGridProperties = nullptr;
  }
  return *this;
}

template <typename DataT>
DataT TriCubicInterpolator<DataT>::extrapolation(const DataT valk, const DataT valk1, const DataT valk2) const
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

template <typename DataT>
DataT TriCubicInterpolator<DataT>::linearExtrapolation(const DataT valk, const DataT valk1) const
{
  const DataT val = 2 * valk - valk1;
  return val;
}

template <typename DataT>
DataT TriCubicInterpolator<DataT>::parabolExtrapolation(const DataT valk, const DataT valk1, const DataT valk2) const
{
  const DataT val = 3 * (valk - valk1) + valk2; // legendre polynom with x0=0, x1=1, x2=2 and z=-1
  return val;
}

namespace
{
/// calculate the 4 weights of the 1D cubic (Catmull-Rom) interpolation for the relative position t in the cell
template <typename DataT>
inline void getWeights(const DataT t, DataT (&weights)[4])
{
  const DataT t2 = t * t;
  const DataT t3 = t2 * t;
  weights[0] = DataT(0.5) * (-t + 2 * t2 - t3);
  weights[1] = DataT(0.5) * (2 - 5 * t2 + 3 * t3);
  weights[2] = DataT(0.5) * (t + 4 * t2 - 3 * t3);
  weights[3] = DataT(0.5) * (t3 - t2);
}

/// sum_{slice, row, iz} weightsPhi[slice] * weightsR[row] * weightsZ[iz] * vals[sliceOffset[slice] + row * rowStride + iz]
/// \param vals values of the grid
/// \param sliceOffset offset of the 4 phi slices
/// \param rowStride offset between the rows in r direction (the 4 values in z direction are consecutive)
template <typename DataT>
inline DataT interpolateCell(const DataT* vals, const int (&sliceOffset)[4], const int rowStride, const DataT (&weightsZ)[4], const DataT (&weightsR)[4], const DataT (&weightsPhi)[4])
{
  DataT result = 0;
  for (int slice = 0; slice < 4; ++slice) {
    DataT sumR = 0;
    for (int row = 0; row < 4; ++row) {
      const DataT* val = vals + sliceOffset[slice] + row * rowStride;
      sumR += weightsR[row] * (weightsZ[0] * val[0] + weightsZ[1] * val[1] + weightsZ[2] * val[2] + weightsZ[3] * val[3]);
    }
    result += weightsPhi[slice] * sumR;
  }
  return result;
}

/// interpolateCell() for three data sets on the same grid in one pass (same order of the operations as interpolateCell() for each data set)
template <typename DataT>
inline void interpolateCell3(const DataT* vals0, const DataT* vals1, const DataT* vals2, const int (&sliceOffset)[4], const int rowStride, const DataT (&weightsZ)[4], const DataT (&weightsR)[4], const DataT (&weightsPhi)[4], DataT& result0, DataT& result1, DataT& result2)
{
  result0 = 0;
  result1 = 0;
  result2 = 0;
  for (int slice = 0; slice < 4; ++slice) {
    DataT sumR0 = 0;
    DataT sumR1 = 0;
    DataT sumR2 = 0;
    for (int row = 0; row < 4; ++row) {
      const int offset = sliceOffset[slice] + row * rowStride;
      const DataT* val0 = vals0 + offset;
      const DataT* val1 = vals1 + offset;
      const DataT* val2 = vals2 + offset;
      sumR0 += weightsR[row] * (weightsZ[0] * val0[0] + weightsZ[1] * val0[1] + weightsZ[2] * val0[2] + weightsZ[3] * val0[3]);
      sumR1 += weightsR[row] * (weightsZ[0] * val1[0] + weightsZ[1] * val1[1] + weightsZ[2] * val1[2] + weightsZ[3] * val1[3]);
      sumR2 += weightsR[row] * (weightsZ[0] * val2[0] + weightsZ[1] * val2[1] + weightsZ[2] * val2[2] + weightsZ[3] * val2[3]);
    }
    result0 += weightsPhi[slice] * sumR0;
    result1 += weightsPhi[slice] * sumR1;
    result2 += weightsPhi[slice] * sumR2;
  }
}
} // namespace

template <typename DataT>
TriCubicCell<DataT> TriCubicInterpolator<DataT>::getCell(const DataT z, const DataT r, const DataT phi) const
{
  TriCubicCell<DataT> cell;

  // relative position in the grid (in units of the grid spacing)
  DataT posRel[FDim]{(z - mGridProperties->getGridMinZ()) * mGridProperties->getInvSpacingZ(), (r - mGridProperties->getGridMinR()) * mGridProperties->getInvSpacingR(), (phi - mGridProperties->getGridMinPhi()) * mGridProperties->getInvSpacingPhi()};
  if (!std::isfinite(posRel[FZ]) || !std::isfinite(posRel[FR]) || !std::isfinite(posRel[FPHI])) {
    // query position is not finite: use a valid cell to avoid an access outside of the grid, the interpolated value is NaN
    const DataT nan = std::numeric_limits<DataT>::quiet_NaN();
    getWeights<DataT>(nan, cell.weightsZ);
    getWeights<DataT>(nan, cell.weightsR);
    getWeights<DataT>(nan, cell.weightsPhi);
    return cell;
  }
  posRel[FPHI] = mGridProperties->clampToGridCircularRel(posRel[FPHI], FPHI);
  DataT posRelN[FDim]{posRel[FZ], posRel[FR], posRel[FPHI]};
  posRel[FZ] = mGridProperties->clampToGridRel(posRel[FZ], FZ);
  posRel[FR] = mGridProperties->clampToGridRel(posRel[FR], FR);
  if (!mExtraPolateValues) {
    // restrict the query position to the grid. The index of the cell is at most the second last vertex (see clampToGridRel()), but the position has to be clamped to the last vertex
    posRelN[FZ] = std::min(std::max(posRelN[FZ], DataT(0)), mGridProperties->getMaxIndexZ());
    posRelN[FR] = std::min(std::max(posRelN[FR], DataT(0)), mGridProperties->getMaxIndexR());
  }

  // index of the cell: the relative positions are >= 0 after the clamping, so the conversion to int is equal to floor()
  cell.indexZ = static_cast<int>(posRel[FZ]);
  cell.indexR = static_cast<int>(posRel[FR]);
  cell.indexPhi = static_cast<int>(posRel[FPHI]);

  // weights of the interpolation in each dimension
  getWeights<DataT>(posRelN[FZ] - cell.indexZ, cell.weightsZ);
  getWeights<DataT>(posRelN[FR] - cell.indexR, cell.weightsR);
  getWeights<DataT>(posRelN[FPHI] - cell.indexPhi, cell.weightsPhi);

  // all 64 points used for the interpolation are on the grid if the cell does not touch the first or last vertex in z and r (the phi direction is periodic)
  cell.inGrid = (cell.indexZ >= 1) && (cell.indexZ <= static_cast<int>(mGridProperties->getN(FZ)) - 3) && (cell.indexR >= 1) && (cell.indexR <= static_cast<int>(mGridProperties->getN(FR)) - 3);
  if (cell.inGrid) {
    // all 64 values are inside the grid (the phi direction is periodic): the values are directly taken from the grid.
    // The values in z direction are consecutive in memory (DataContainer3D::getDataIndex()): start at (iz - 1, ir - 1, iphi)
    int deltaPhi[3]{};
    getDataIndexCircularArray(cell.indexPhi, FPHI, deltaPhi);
    cell.sliceOffset[0] = deltaPhi[0];
    cell.sliceOffset[1] = 0;
    cell.sliceOffset[2] = deltaPhi[1];
    cell.sliceOffset[3] = deltaPhi[2];
    cell.rowStride = mGridProperties->getDeltaDataIndex(1, FR);
    cell.dataIndex = mGridData->getDataIndex(cell.indexZ, cell.indexR, cell.indexPhi) - cell.rowStride - 1;
  }
  return cell;
}

template <typename DataT>
DataT TriCubicInterpolator<DataT>::interpolateData(const DataContainer& data, const TriCubicCell<DataT>& cell) const
{
  // check if data is empty
  if (!data.getNDataPoints()) {
    return 0;
  }

  if (cell.inGrid) {
    return interpolateCell<DataT>(data.getData().data() + cell.dataIndex, cell.sliceOffset, cell.rowStride, cell.weightsZ, cell.weightsR, cell.weightsPhi);
  }

  // missing values outside of the grid are extrapolated
  DataT cVals[16][4];
  setValues(data, cell.indexZ, cell.indexR, cell.indexPhi, cVals);
  static constexpr int sliceOffset[4]{0, 16, 32, 48};
  return interpolateCell<DataT>(&cVals[0][0], sliceOffset, 4, cell.weightsZ, cell.weightsR, cell.weightsPhi);
}

template <typename DataT>
DataT TriCubicInterpolator<DataT>::interpolateSparse(const DataT z, const DataT r, const DataT phi) const
{
  // check if data is empty
  if (!mGridData->getNDataPoints()) {
    return 0;
  }
  return interpolateData(*mGridData, getCell(z, r, phi));
}

template <typename DataT>
void TriCubicInterpolator<DataT>::interpolate(const DataT z, const DataT r, const DataT phi, const TriCubicInterpolator<DataT>& other1, const TriCubicInterpolator<DataT>& other2, DataT& val0, DataT& val1, DataT& val2) const
{
  const size_t nPoints = mGridData->getNDataPoints();
  if (!nPoints || (other1.mGridData->getNDataPoints() != nPoints) || (other2.mGridData->getNDataPoints() != nPoints)) {
    // empty data or data with different grid: separate interpolation
    val0 = interpolateSparse(z, r, phi);
    val1 = other1.interpolateSparse(z, r, phi);
    val2 = other2.interpolateSparse(z, r, phi);
    return;
  }
  const TriCubicCell<DataT> cell = getCell(z, r, phi);
  if (cell.inGrid) {
    // all values inside of the grid: interpolate the three data sets in one pass
    interpolateCell3<DataT>(mGridData->getData().data() + cell.dataIndex, other1.mGridData->getData().data() + cell.dataIndex, other2.mGridData->getData().data() + cell.dataIndex, cell.sliceOffset, cell.rowStride, cell.weightsZ, cell.weightsR, cell.weightsPhi, val0, val1, val2);
    return;
  }
  val0 = interpolateData(*mGridData, cell);
  val1 = interpolateData(*other1.mGridData, cell);
  val2 = interpolateData(*other2.mGridData, cell);
}

// for perdiodic boundary condition
template <typename DataT>
void TriCubicInterpolator<DataT>::getDataIndexCircularArray(const int index0, const int dim, int arr[]) const
{
  const int delta_min1 = getRegulatedDelta(index0, -1, dim, mGridProperties->getN(dim) - 1);
  const int delta_plus1 = getRegulatedDelta(index0, +1, dim, 1 - mGridProperties->getN(dim));
  const int delta_plus2 = getRegulatedDelta(index0, +2, dim, 2 - mGridProperties->getN(dim));

  arr[0] = mGridProperties->getDeltaDataIndex(delta_min1, dim);
  arr[1] = mGridProperties->getDeltaDataIndex(delta_plus1, dim);
  arr[2] = mGridProperties->getDeltaDataIndex(delta_plus2, dim);
}

template <typename DataT>
void TriCubicInterpolator<DataT>::setValues(const DataContainer& data, const int iz, const int ir, const int iphi, DataT (&cVals)[16][4]) const
{
  const int nZ = mGridProperties->getN(FZ);
  const int nR = mGridProperties->getN(FR);

  // offsets of the data: the phi direction is periodic
  int deltaPhi[3]{};
  getDataIndexCircularArray(iphi, FPHI, deltaPhi);
  const int sliceOffset[4]{deltaPhi[0], 0, deltaPhi[1], deltaPhi[2]};
  const int strideR = mGridProperties->getDeltaDataIndex(1, FR);
  const int strideZ = mGridProperties->getDeltaDataIndex(1, FZ);
  const int dataIndex = data.getDataIndex(iz, ir, iphi);

  // value of the point with the relative indices (row - 1, col - 1) to the cell in r and z
  const auto value = [&](const int slice, const int row, const int col) { return data[dataIndex + sliceOffset[slice] + (row - 1) * strideR + (col - 1) * strideZ]; };

  for (int slice = 0; slice < 4; ++slice) {
    for (int row = 0; row < 4; ++row) {
      // points outside of the grid in r and/or z are extrapolated from the next 3 points towards the inside of the grid (along the diagonal if outside in r and z)
      const int indexR = ir + row - 1;
      const int stepR = (indexR < 0) ? 1 : ((indexR >= nR) ? -1 : 0);
      for (int col = 0; col < 4; ++col) {
        const int indexZ = iz + col - 1;
        const int stepZ = (indexZ < 0) ? 1 : ((indexZ >= nZ) ? -1 : 0);
        if ((stepR == 0) && (stepZ == 0)) {
          cVals[4 * slice + row][col] = value(slice, row, col);
        } else {
          cVals[4 * slice + row][col] = extrapolation(value(slice, row + stepR, col + stepZ), value(slice, row + 2 * stepR, col + 2 * stepZ), value(slice, row + 3 * stepR, col + 3 * stepZ));
        }
      }
    }
  }
}

template class o2::tpc::TriCubicInterpolator<double>;
template class o2::tpc::TriCubicInterpolator<float>;
