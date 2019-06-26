// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastTransformGeo.h
/// \brief Definition of TPCFastTransformGeo class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMGEO_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMGEO_H

#include "GPUCommonDef.h"
#ifndef __OPENCL__
#include <memory>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The TPCFastTransformGeo class contains TPC geometry needed for the TPCFastTransform
///
class TPCFastTransformGeo
{
 public:
  /// The struct contains necessary info for TPC slice
  struct SliceInfo {
    float sinAlpha;
    float cosAlpha;
  };

  /// The struct contains necessary info about TPC padrow
  struct RowInfo {
    float x;          ///< nominal X coordinate of the row [cm]
    int maxPad;       ///< maximal pad number = n pads - 1
    float padWidth;   ///< width of pads [cm]
    float u0;         ///< min. u coordinate
    float scaleUtoSU; ///< scale for su (scaled u ) coordinate
    float scaleSUtoU; ///< scale for u coordinate
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastTransformGeo();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransformGeo(const TPCFastTransformGeo&) CON_DEFAULT;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransformGeo& operator=(const TPCFastTransformGeo&) CON_DEFAULT;

  /// Destructor
  ~TPCFastTransformGeo() CON_DEFAULT;

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Gives minimal alignment in bytes required for an object of the class
  static constexpr size_t getClassAlignmentBytes() { return 8; }

  /// _______________  Construction interface  ________________________

  /// Starts the initialization procedure, reserves temporary memory
  void startConstruction(int numberOfRows);

  /// Initializes a TPC row
  void setTPCrow(int iRow, float x, int nPads, float padWidth);

  /// Sets TPC geometry
  ///
  /// It must be called once during initialization
  void setTPCzLength(float tpcZlengthSideA, float tpcZlengthSideC);

  /// Sets all drift calibration parameters and the time stamp
  ///
  /// It must be called once during construction,
  /// but also may be called afterwards to reset these parameters.
  void setTPCalignmentZ(float tpcAlignmentZ);

  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  /// Is the object constructed
  bool isConstructed() const { return (mConstructionMask == (unsigned int)ConstructionState::Constructed); }

  /// _______________  Getters _________________________________

  /// Gives number of TPC slices
  GPUd() static constexpr int getNumberOfSlices() { return NumberOfSlices; }

  /// Gives number of TPC slices in A side
  GPUd() static constexpr int getNumberOfSlicesA() { return NumberOfSlicesA; }

  /// Gives number of TPC rows
  GPUd() int getNumberOfRows() const { return mNumberOfRows; }

  /// Gives slice info
  GPUd() const SliceInfo& getSliceInfo(int slice) const;

  /// Gives TPC row info
  GPUd() const RowInfo& getRowInfo(int row) const;

  /// Gives Z length of the TPC, side A
  GPUd() float getTPCzLengthA() const { return mTPCzLengthA; }

  /// Gives Z length of the TPC, side C
  GPUd() float getTPCzLengthC() const { return mTPCzLengthC; }

  /// _______________  Conversion of coordinate systems __________

  /// convert Local -> Global c.s.
  GPUd() void convLocalToGlobal(int slice, float lx, float ly, float lz, float& gx, float& gy, float& gz) const;

  /// convert Global->Local c.s.
  GPUd() void convGlobalToLocal(int slice, float gx, float gy, float gz, float& lx, float& ly, float& lz) const;

  /// convert UV -> Local c.s.
  GPUd() void convUVtoLocal(int slice, float u, float v, float& y, float& z) const;

  /// convert Local-> UV c.s.
  GPUd() void convLocalToUV(int slice, float y, float z, float& u, float& v) const;

  /// convert UV -> Scaled UV
  GPUd() void convUVtoScaledUV(int slice, int row, float u, float v, float& su, float& sv) const;

  /// convert Scaled UV -> UV
  GPUd() void convScaledUVtoUV(int slice, int row, float su, float sv, float& u, float& v) const;

  /// convert Pad coordinate -> U
  GPUd() void convPadToU(int row, float pad, float& u) const;

  /// convert U -> Pad coordinate
  GPUd() void convUtoPad(int row, float u, float& pad) const;

  /// Print method
  void print() const;

  /// Method for testing consistency
  int test(int slice, int row, float ly, float lz) const;

  /// Method for testing consistency
  int test() const;

 private:
  /// _______________  Data members  _______________________________________________

  static constexpr int NumberOfSlices = 36;                  ///< Number of TPC slices ( slice = inner + outer sector )
  static constexpr int NumberOfSlicesA = NumberOfSlices / 2; ///< Number of TPC slices side A
  static constexpr int MaxNumberOfRows = 160;                ///< Max Number of TPC rows in a slice

  /// _______________  Construction control  _______________________________________________

  /// Enumeration of construction states
  enum ConstructionState : unsigned int {
    NotConstructed = 0x0, ///< the object is not constructed
    Constructed = 0x1,    ///< the object is constructed, temporary memory is released
    InProgress = 0x2,     ///< construction started: temporary  memory is reserved
    GeometryIsSet = 0x4,  ///< the TPC geometry is set
    AlignmentIsSet = 0x8  ///< the TPC alignment is set
  };

  unsigned int mConstructionMask = ConstructionState::NotConstructed; ///< mask for constructed object members, first two bytes are used by this class

  /// _______________  Geometry  _______________________________________________

  int mNumberOfRows = 0;        ///< Number of TPC rows. It is different for the Run2 and the Run3 setups
  float mTPCzLengthA = 0.f;     ///< Z length of the TPC, side A
  float mTPCzLengthC = 0.f;     ///< Z length of the TPC, side C
  float mTPCalignmentZ = 0.f;   ///< Global Z shift of the TPC detector. It is applied at the end of the transformation.
  float mScaleVtoSVsideA = 0.f; ///< scale for v->sv for TPC side A
  float mScaleVtoSVsideC = 0.f; ///< scale for v->sv for TPC side C
  float mScaleSVtoVsideA = 0.f; ///< scale for sv->v for TPC side A
  float mScaleSVtoVsideC = 0.f; ///< scale for sv->v for TPC side C

  SliceInfo mSliceInfos[NumberOfSlices + 1]; ///< array of slice information [fixed size]
  RowInfo mRowInfos[MaxNumberOfRows + 1];    ///< array of row information [fixed size]
};

// =======================================================================
//              Inline implementations of some methods
// =======================================================================

GPUdi() const TPCFastTransformGeo::SliceInfo& TPCFastTransformGeo::getSliceInfo(int slice) const
{
  /// Gives slice info
  if (slice < 0 || slice >= NumberOfSlices) { // return zero object
    slice = NumberOfSlices;
  }
  return mSliceInfos[slice];
}

GPUdi() const TPCFastTransformGeo::RowInfo& TPCFastTransformGeo::getRowInfo(int row) const
{
  /// Gives TPC row info
  if (row < 0 || row >= mNumberOfRows) { // return zero object
    row = MaxNumberOfRows;
  }
  return mRowInfos[row];
}

GPUdi() void TPCFastTransformGeo::convLocalToGlobal(int slice, float lx, float ly, float lz, float& gx, float& gy, float& gz) const
{
  /// convert Local -> Global c.s.
  const SliceInfo& sliceInfo = getSliceInfo(slice);
  gx = lx * sliceInfo.cosAlpha - ly * sliceInfo.sinAlpha;
  gy = lx * sliceInfo.sinAlpha + ly * sliceInfo.cosAlpha;
  gz = lz;
}

GPUdi() void TPCFastTransformGeo::convGlobalToLocal(int slice, float gx, float gy, float gz, float& lx, float& ly, float& lz) const
{
  /// convert Global -> Local c.s.
  const SliceInfo& sliceInfo = getSliceInfo(slice);
  lx = gx * sliceInfo.cosAlpha + gy * sliceInfo.sinAlpha;
  ly = -gx * sliceInfo.sinAlpha + gy * sliceInfo.cosAlpha;
  lz = gz;
}

GPUdi() void TPCFastTransformGeo::convUVtoLocal(int slice, float u, float v, float& ly, float& lz) const
{
  /// convert UV -> Local c.s.
  if (slice < NumberOfSlicesA) { // TPC side A
    ly = u;
    lz = mTPCzLengthA - v;
  } else {                 // TPC side C
    ly = -u;               // pads are mirrorred on C-side
    lz = v - mTPCzLengthC; // drift direction is mirrored on C-side
  }
  lz += mTPCalignmentZ; // global TPC alignment
}

GPUdi() void TPCFastTransformGeo::convLocalToUV(int slice, float ly, float lz, float& u, float& v) const
{
  /// convert Local-> UV c.s.
  lz = lz - mTPCalignmentZ;      // global TPC alignment
  if (slice < NumberOfSlicesA) { // TPC side A
    u = ly;
    v = mTPCzLengthA - lz;
  } else {                 // TPC side C
    u = -ly;               // pads are mirrorred on C-side
    v = lz + mTPCzLengthC; // drift direction is mirrored on C-side
  }
}

GPUdi() void TPCFastTransformGeo::convUVtoScaledUV(int slice, int row, float u, float v, float& su, float& sv) const
{
  /// convert UV -> Scaled UV
  const RowInfo& rowInfo = getRowInfo(row);
  su = (u - rowInfo.u0) * rowInfo.scaleUtoSU;
  if (slice < NumberOfSlicesA) {
    sv = v * mScaleVtoSVsideA;
  } else {
    sv = v * mScaleVtoSVsideC;
  }
}

GPUdi() void TPCFastTransformGeo::convScaledUVtoUV(int slice, int row, float su, float sv, float& u, float& v) const
{
  /// convert Scaled UV -> UV
  const RowInfo& rowInfo = getRowInfo(row);
  u = rowInfo.u0 + su * rowInfo.scaleSUtoU;
  if (slice < NumberOfSlicesA) {
    v = sv * mScaleSVtoVsideA;
  } else {
    v = sv * mScaleSVtoVsideC;
  }
}

GPUdi() void TPCFastTransformGeo::convPadToU(int row, float pad, float& u) const
{
  /// convert Pad coordinate -> U
  const RowInfo& rowInfo = getRowInfo(row);
  u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;
}

GPUdi() void TPCFastTransformGeo::convUtoPad(int row, float u, float& pad) const
{
  /// convert U -> Pad coordinate
  const RowInfo& rowInfo = getRowInfo(row);
  pad = u / rowInfo.padWidth + 0.5 * rowInfo.maxPad;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
