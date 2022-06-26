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

/// \file MatLayerCyl.h
/// \brief Declarations for single cylindrical material layer class

#ifndef ALICEO2_MATLAYERCYL_H
#define ALICEO2_MATLAYERCYL_H

#ifndef GPUCA_GPUCODE_DEVICE
#include <cmath>
#include <cstring>
#endif
#include "GPUCommonDef.h"
#include "FlatObject.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonMath.h"
#include "DetectorsBase/MatCell.h"

namespace o2
{
namespace base
{

/**********************************************************************
 *                                                                    *
 * Material data on the cells of cylindrical layer                    *
 * Cell is limited by 2 radial planes at phiMin,phiMax,               *
 * radii radMin,radMax and XY planes at zMin, zMax.                   *
 * This limits are defined in the cell container class                *
 *                                                                    *
 *                                                                    *
 * Cylindrical material layer                                         *
 *                                                                    *
 * Logical division is given by mNZBins and mNPhiBins                 *
 * but the actual number phi slices might be less if neighbouring     *
 * phi bins with similar properties are merged together. The actual   *
 * phi slice should be accessed via mPhiBin2Slice map.                *
 *                                                                    *
 **********************************************************************/
class MatLayerCyl : public o2::gpu::FlatObject
{

 public:
  enum RangeStatus : short { Below = -1,
                             Within = 0,
                             Above = 1 };

  MatLayerCyl();
  MatLayerCyl(const MatLayerCyl& src) CON_DELETE;
  ~MatLayerCyl() CON_DEFAULT;

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
  MatLayerCyl(float rMin, float rMax, float zHalfSpan, float dzMin, float drphiMin);

  void initSegmentation(float rMin, float rMax, float zHalfSpan, int nz, int nphi);
  void initSegmentation(float rMin, float rMax, float zHalfSpan, float dzMin, float drphiMin);
  void populateFromTGeo(int ntrPerCell = 10);
  void populateFromTGeo(int ip, int iz, int ntrPerCell);
  void print(bool data = false) const;
#endif // !GPUCA_ALIGPUCODE

  GPUd() float getRMin() const
  {
    return o2::gpu::CAMath::Sqrt(getRMin2());
  }
  GPUd() float getRMax() const { return o2::gpu::CAMath::Sqrt(getRMax2()); }
  GPUd() float getZMin() const { return -mZHalf; }
  GPUd() float getZMax() const { return mZHalf; }

  GPUd() int getNZBins() const { return mNZBins; }
  GPUd() int getNPhiBins() const { return mNPhiBins; }
  GPUd() int getNPhiSlices() const { return mNPhiSlices; }
  GPUd() int getNPhiBinsInSlice(int iSlice, int& binMin, int& binMax) const;

  GPUd() float getRMin2() const { return mRMin2; }
  GPUd() float getRMax2() const { return mRMax2; }
  GPUd() float getDZ() const { return mDZ; }
  GPUd() float getDPhi() const { return mDPhi; }

  // obtain material cell, cell ID must be valid
  GPUd() const MatCell& getCellPhiBin(int iphi, int iz) const { return mCells[getCellIDPhiBin(iphi, iz)]; }
  GPUd() const MatCell& getCell(int iphiSlice, int iz) const { return mCells[getCellID(iphiSlice, iz)]; }

  // ---------------------- Z slice manipulation
  // convert Z to Zslice
  GPUd() RangeStatus isZOutside(float z) const { return z < getZMin() ? Below : (z > getZMax() ? Above : Within); }
  GPUd() int getZBinID(float z) const
  {
    int idz = int((z - getZMin()) * getDZInv()); // cannot be negative since before isZOutside is applied
    return idz < getNZBins() ? idz : getNZBins() - 1;
  }

  // lower boundary of Z slice
  GPUd() float getZBinMin(int id) const { return getZMin() + id * getDZ(); }

  // upper boundary of Z slice
  GPUd() float getZBinMax(int id) const { return getZMin() + (id + 1) * getDZ(); }

  // ---------------------- Phi slice manipulation (0:2pi convention, no check is done)
  GPUd() int phiBin2Slice(int i) const { return mPhiBin2Slice[i]; }
  GPUd() int getPhiSliceID(float phi) const { return phiBin2Slice(getPhiBinID(phi)); }

  // lower boundary of phi slice
  GPUd() float getPhiBinMin(int id) const { return id * getDPhi(); }

  // upper boundary of phi slice
  GPUd() float getPhiBinMax(int id) const { return (id + 1) * getDPhi(); }

  // sin and cosine of the slice lower angle
  GPUd() float getSliceCos(int i) const { return mSliceCos[i]; }
  GPUd() float getSliceSin(int i) const { return mSliceSin[i]; }

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
  void getMeanRMS(MatCell& mean, MatCell& rms) const;
  bool cellsDiffer(const MatCell& cellA, const MatCell& cellB, float maxRelDiff) const;
  bool canMergePhiSlices(int i, int j, float maxRelDiff = 0.05, int maxDifferent = 1) const;
  void optimizePhiSlices(float maxRelDiff = 0.05);
  void flatten(char* newPtr);
#endif // !GPUCA_ALIGPUCODE

#ifndef GPUCA_GPUCODE
  std::size_t estimateFlatBufferSize() const
  {
    return estimateFlatBufferSize(getNPhiBins(), getNPhiSlices(), getNZBins());
  }
  static std::size_t estimateFlatBufferSize(int nPhiBins, int nPhiSlices, int nZBins)
  {
    size_t sz = 0;
    sz += alignSize(sz + nPhiBins * sizeof(short), getBufferAlignmentBytes());              // mPhiBin2Slice
    sz += alignSize(sz + nPhiSlices * sizeof(float), getBufferAlignmentBytes());            // mSliceCos
    sz += alignSize(sz + nPhiSlices * sizeof(float), getBufferAlignmentBytes());            // mSliceSin
    sz += alignSize(sz + nPhiSlices * nZBins * sizeof(MatCell), getBufferAlignmentBytes()); // mSliceSin
    return sz;
  }
  void fixPointers(char* oldPtr, char* newPtr);
  void setFlatPointer(char* ptr)
  {
    // brute force assignment of new pointers
    mFlatBufferPtr = ptr;
    if (mFlatBufferContainer) {
      mFlatBufferContainer = ptr;
    }
  }

  /// Gives minimal alignment in bytes required for the class object
  static constexpr size_t getClassAlignmentBytes() { return 8; }
  /// Gives minimal alignment in bytes required for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return 8; }
#endif

 protected:
  GPUd() int getNCells() const { return getNZBins() * getNPhiSlices(); }
  GPUd() float getDZInv() const { return mDZInv; }
  GPUd() float getDPhiInv() const { return mDPhiInv; }

  // linearized cell ID from phi slice and z bin
  GPUd() int getCellID(int iphi, int iz) const { return iphi * getNZBins() + iz; }

  // linearized cell ID from phi bin and z bin
  GPUd() int getCellIDPhiBin(int iphi, int iz) const { return getCellID(phiBin2Slice(iphi), iz); }

  // convert Phi (in 0:2pi convention) to PhiBinID
  GPUd() int getPhiBinID(float phi) const
  {
    auto idphi = int(phi * getDPhiInv());
    return idphi < getNPhiBins() ? idphi : getNPhiBins() - 1;
  }

  GPUd() int getEdgePhiBinOfSlice(int phiBin, int dir) const
  {
    // Get edge bin (in direction dir) of the slice, to which phiBin belongs
    // No check for phiBin validity is done
    auto slice = phiBin2Slice(phiBin);
    while (slice == phiBin2Slice((phiBin += dir))) {
      ;
    }
    return phiBin - dir;
  }

  //------------------------------------------------------

  short mNZBins;     ///< number of Z bins
  short mNPhiBins;   ///< number of phi bins (logical)
  short mNPhiSlices; ///< actual number of phi slices
  //
  float mZHalf;   ///< Z half span
  float mRMin2;   ///< squared min r
  float mRMax2;   ///< squared max r
  float mDZ;      ///< Z slice thickness
  float mDZInv;   ///< Z slice thickness inverse
  float mDPhi;    ///< phi slice thickness
  float mDPhiInv; ///< phi slice thickness inverse

  short* mPhiBin2Slice; //! mapping from analytical phi bin ID to real slice ID

  float* mSliceCos; //! cached cos each phi slice
  float* mSliceSin; //! cached sin each phi slice

  MatCell* mCells; //! mat.budget per cell

  ClassDefNV(MatLayerCyl, 1);
};

} // namespace base
} // namespace o2

#endif
