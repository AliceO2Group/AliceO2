// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  SemiregularSpline2D3D.cxx
/// \brief Implementation of SemiregularSpline2D3D class
///
/// \author  Felix Lapp
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "SemiregularSpline2D3D.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

SemiregularSpline2D3D::SemiregularSpline2D3D()
  : FlatObject(),
    mGridV(),
    mNumberOfRows(0),
    mNumberOfKnots(0),
    mDataIndexMapOffset(0)
{
  /// Default constructor. Creates an empty uninitialised object
}

void SemiregularSpline2D3D::destroy()
{
  mNumberOfRows = 0;
  mDataIndexMapOffset = 0;
  mNumberOfKnots = 0;
  FlatObject::destroy();
}

void SemiregularSpline2D3D::relocateBufferPointers(const char* oldBuffer, char* actualBuffer)
{
  /// relocate pointers from old to new buffer location

  /*char *bufferV = FlatObject::relocatePointer( oldBuffer, actualBuffer, mGridV.getFlatBufferPtr() );
	mGridV.setActualBufferAddress( bufferV );*/

  /*for( int i=0; i<mNumberOfRows; i++) {
	char *bufferUi = FlatObject::relocatePointer(oldBuffer, actualBuffer, mSplineArray[i].getFlatBufferPtr() );
	mSplineArray[i].setActualBufferAddress( bufferUi );
	}*/
}

void SemiregularSpline2D3D::cloneFromObject(const SemiregularSpline2D3D& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mNumberOfRows = obj.mNumberOfRows;
  mNumberOfKnots = obj.mNumberOfKnots;
  mGridV = obj.mGridV;
  mDataIndexMapOffset = obj.mDataIndexMapOffset;
}

void SemiregularSpline2D3D::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void SemiregularSpline2D3D::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void SemiregularSpline2D3D::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  /*const char* oldFlatBufferPtr = mFlatBufferPtr;
 
	char *bufferV = relocatePointer( oldFlatBufferPtr, futureFlatBufferPtr, mGridV.getFlatBufferPtr() );
	mGridV.setFutureBufferAddress( bufferV );*/

  /*for( int i=0; i<mNumberOfRows; i++ ) {
	char *bufferUi = relocatePointer( oldFlatBufferPtr, futureFlatBufferPtr, mSplineArray[i].getFlatBufferPtr() );
	mSplineArray[i].setFutureBufferAddress( bufferUi );
	}*/

  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void SemiregularSpline2D3D::construct(const int numberOfRowsInput, const int numbersOfKnots[])
{
  /// Constructor
  ///
  /// When the number of rows / knots is less than 5 it will be set to 5
  ///

  int numberOfRows = numberOfRowsInput;
  if (numberOfRows < 5) {
    numberOfRows = 5;
  }

  FlatObject::startConstruction();

  //construct regular grid for v
  mGridV.construct(numberOfRows);

  // For each x element numbersOfKnots may be a single RegularSpline1D with x knots.
  // so first create the array
  RegularSpline1D splineArray[numberOfRows];

  // And construct them
  for (int i = 0; i < numberOfRowsInput; i++) {
    splineArray[i].construct(numbersOfKnots[i]);
  }
  for (int i = numberOfRowsInput; i < numberOfRows; i++) {
    splineArray[i].construct(5);
  }

  // this is the space which is taken just by the RegularSpline1D's
  mDataIndexMapOffset = numberOfRows * sizeof(RegularSpline1D);

  //The buffer size is the size of the array
  FlatObject::finishConstruction(mDataIndexMapOffset + numberOfRows * sizeof(int));

  // Array for the 1D-Splines inside the buffer
  RegularSpline1D* bufferSplines = getSplineArrayNonConst();

  // paste local splineArray to the buffer
  for (int i = 0; i < numberOfRows; i++) {
    bufferSplines[i] = splineArray[i];
  }

  // Just calculating the total number of knots in this 2D3D spline.
  int numberOfKnots = 0;
  for (int i = 0; i < numberOfRows; i++) {
    int knotsU = getGridU(i).getNumberOfKnots();
    numberOfKnots += knotsU;
  }

  //save the numberOfRows and numberOfKnots
  mNumberOfRows = numberOfRows;
  mNumberOfKnots = numberOfKnots;

  // map to save the starting data index for each v-coordinate
  int* dataIndexMap = getDataIndexMapNonConst();

  // this will count the amount of u-knots "under" a v-coordinate
  int uSum = 0;

  //count the amount of knots which are in gridU's lower than i
  for (int dv = 0; dv < mNumberOfRows; dv++) {
    dataIndexMap[dv] = uSum;
    uSum += numbersOfKnots[dv];
  }
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE
