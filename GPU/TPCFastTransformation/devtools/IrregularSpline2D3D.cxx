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

/// \file  IrregularSpline2D3D.cxx
/// \brief Implementation of IrregularSpline2D3D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "IrregularSpline2D3D.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

using namespace GPUCA_NAMESPACE::gpu;

IrregularSpline2D3D::IrregularSpline2D3D() : FlatObject(), mGridU(), mGridV()
{
  /// Default constructor. Creates an empty uninitialised object
}

void IrregularSpline2D3D::destroy()
{
  /// See FlatObject for description
  mGridU.destroy();
  mGridV.destroy();
  FlatObject::destroy();
}

void IrregularSpline2D3D::cloneFromObject(const IrregularSpline2D3D& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  char* bufferU = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridU.getFlatBufferPtr());
  mGridU.cloneFromObject(obj.mGridU, bufferU);

  char* bufferV = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridV.getFlatBufferPtr());
  mGridV.cloneFromObject(obj.mGridV, bufferV);
}

void IrregularSpline2D3D::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  FlatObject::moveBufferTo(newFlatBufferPtr);
  setActualBufferAddress(mFlatBufferPtr);
}

void IrregularSpline2D3D::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  size_t vOffset = alignSize(mGridU.getFlatBufferSize(), mGridV.getBufferAlignmentBytes());
  mGridU.setActualBufferAddress(mFlatBufferPtr);
  mGridV.setActualBufferAddress(mFlatBufferPtr + vOffset);
}

void IrregularSpline2D3D::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = mFlatBufferPtr;

  char* bufferU = relocatePointer(oldFlatBufferPtr, futureFlatBufferPtr, mGridU.getFlatBufferPtr());
  mGridU.setFutureBufferAddress(bufferU);

  char* bufferV = relocatePointer(oldFlatBufferPtr, futureFlatBufferPtr, mGridV.getFlatBufferPtr());
  mGridV.setFutureBufferAddress(bufferV);

  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void IrregularSpline2D3D::construct(int numberOfKnotsU, const float knotsU[], int numberOfAxisBinsU, int numberOfKnotsV, const float knotsV[], int numberOfAxisBinsV)
{
  /// Constructor
  ///
  /// Number of knots created and their values may differ from the input values:
  /// - Edge knots 0.f and 1.f will be added if they are not present.
  /// - Knot values are rounded to closest axis bins: k*1./numberOfAxisBins.
  /// - Knots which are too close to each other will be merged
  /// - At least 5 knots and at least 4 axis bins will be created for consistency reason
  ///
  /// \param numberOfKnotsU     U axis: Number of knots in knots[] array
  /// \param knotsU             U axis: Array of knots.
  /// \param numberOfAxisBinsU  U axis: Number of axis bins to map U coordinate to
  ///                           an appropriate [knot(i),knot(i+1)] interval.
  ///                           The knot positions have a "granularity" of 1./numberOfAxisBins
  ///
  /// \param numberOfKnotsV     V axis: Number of knots in knots[] array
  /// \param knotsV             V axis: Array of knots.
  /// \param numberOfAxisBinsV  V axis: Number of axis bins to map U coordinate to
  ///                           an appropriate [knot(i),knot(i+1)] interval.
  ///                           The knot positions have a "granularity" of 1./numberOfAxisBins
  ///

  FlatObject::startConstruction();

  mGridU.construct(numberOfKnotsU, knotsU, numberOfAxisBinsU);
  mGridV.construct(numberOfKnotsV, knotsV, numberOfAxisBinsV);

  size_t vOffset = alignSize(mGridU.getFlatBufferSize(), mGridV.getBufferAlignmentBytes());

  FlatObject::finishConstruction(vOffset + mGridV.getFlatBufferSize());

  mGridU.moveBufferTo(mFlatBufferPtr);
  mGridV.moveBufferTo(mFlatBufferPtr + vOffset);
}

void IrregularSpline2D3D::constructRegular(int numberOfKnotsU, int numberOfKnotsV)
{
  /// Constructor for a regular spline
  /// \param numberOfKnotsU     U axis: Number of knots in knots[] array
  /// \param numberOfKnotsV     V axis: Number of knots in knots[] array
  ///

  FlatObject::startConstruction();

  mGridU.constructRegular(numberOfKnotsU);
  mGridV.constructRegular(numberOfKnotsV);

  size_t vOffset = alignSize(mGridU.getFlatBufferSize(), mGridV.getBufferAlignmentBytes());

  FlatObject::finishConstruction(vOffset + mGridV.getFlatBufferSize());

  mGridU.moveBufferTo(mFlatBufferPtr);
  mGridV.moveBufferTo(mFlatBufferPtr + vOffset);
}

void IrregularSpline2D3D::print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << " Irregular Spline 2D3D: " << std::endl;
  std::cout << " grid U: " << std::endl;
  mGridU.print();
  std::cout << " grid V: " << std::endl;
  mGridV.print();
#endif
}
