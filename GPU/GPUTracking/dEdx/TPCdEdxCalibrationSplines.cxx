// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  dEdxCalibrationSplines.cxx
/// \brief Definition of dEdxCalibrationSplines class
///
/// \author  Matthias Kleiner <matthias.kleiner@cern.ch>

#include "TPCdEdxCalibrationSplines.h"

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "TFile.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

#if !defined(GPUCA_GPUCODE)
TPCdEdxCalibrationSplines::TPCdEdxCalibrationSplines()
  : FlatObject()
{
  /// Default constructor
  int nKnots[mFSplines];
  for (int i = 0; i < mFSplines; i++) {
    nKnots[i] = 2;
  }
  recreate(nKnots, nKnots);
}

TPCdEdxCalibrationSplines::TPCdEdxCalibrationSplines(const TPCdEdxCalibrationSplines& obj)
  : FlatObject()
{
  /// Copy constructor
  this->cloneFromObject(obj, nullptr);
}

TPCdEdxCalibrationSplines& TPCdEdxCalibrationSplines::operator=(const TPCdEdxCalibrationSplines& obj)
{
  /// Assignment operator
  this->cloneFromObject(obj, nullptr);
  return *this;
}

void TPCdEdxCalibrationSplines::recreate(int nKnotsU1[], int nKnotsU2[])
{
  /// Default constructor

  FlatObject::startConstruction();

  int buffSize = 0;
  int offsets1[mFSplines];
  int offsets2[mFSplines];
  for (int i = 0; i < mFSplines; i++) {
    mCalibSplinesqMax[i].recreate(nKnotsU1[i], nKnotsU2[i]);
    buffSize = alignSize(buffSize, mCalibSplinesqMax[i].getBufferAlignmentBytes());
    offsets1[i] = buffSize;
    buffSize += mCalibSplinesqMax[i].getFlatBufferSize();
  }
  for (int i = 0; i < mFSplines; i++) {
    mCalibSplinesqTot[i].recreate(nKnotsU1[i], nKnotsU2[i]);
    buffSize = alignSize(buffSize, mCalibSplinesqTot[i].getBufferAlignmentBytes());
    offsets2[i] = buffSize;
    buffSize += mCalibSplinesqTot[i].getFlatBufferSize();
  }

  FlatObject::finishConstruction(buffSize);

  for (int i = 0; i < mFSplines; i++) {
    mCalibSplinesqMax[i].moveBufferTo(mFlatBufferPtr + offsets1[i]);
  }
  for (int i = 0; i < mFSplines; i++) {
    mCalibSplinesqTot[i].moveBufferTo(mFlatBufferPtr + offsets2[i]);
  }
}

#endif

#if !defined(GPUCA_GPUCODE)

void TPCdEdxCalibrationSplines::cloneFromObject(const TPCdEdxCalibrationSplines& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  for (int i = 0; i < mFSplines; i++) {
    char* buffer = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mCalibSplinesqMax[i].getFlatBufferPtr());
    mCalibSplinesqMax[i].cloneFromObject(obj.mCalibSplinesqMax[i], buffer);
  }

  for (int i = 0; i < mFSplines; i++) {
    char* buffer = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mCalibSplinesqTot[i].getFlatBufferPtr());
    mCalibSplinesqTot[i].cloneFromObject(obj.mCalibSplinesqTot[i], buffer);
  }
}

void TPCdEdxCalibrationSplines::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}
#endif

void TPCdEdxCalibrationSplines::destroy()
{
  /// See FlatObject for description
  for (int i = 0; i < mFSplines; i++) {
    mCalibSplinesqMax[i].destroy();
    mCalibSplinesqTot[i].destroy();
  }
  FlatObject::destroy();
}

void TPCdEdxCalibrationSplines::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  int offset = 0;
  for (int i = 0; i < mFSplines; i++) {
    offset = alignSize(offset, mCalibSplinesqMax[i].getBufferAlignmentBytes());
    mCalibSplinesqMax[i].setActualBufferAddress(mFlatBufferPtr + offset);
    offset += mCalibSplinesqMax[i].getFlatBufferSize();
  }
  for (int i = 0; i < mFSplines; i++) {
    offset = alignSize(offset, mCalibSplinesqTot[i].getBufferAlignmentBytes());
    mCalibSplinesqTot[i].setActualBufferAddress(mFlatBufferPtr + offset);
    offset += mCalibSplinesqTot[i].getFlatBufferSize();
  }
}

void TPCdEdxCalibrationSplines::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description

  for (int i = 0; i < mFSplines; i++) {
    char* buffer = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mCalibSplinesqMax[i].getFlatBufferPtr());
    mCalibSplinesqMax[i].setFutureBufferAddress(buffer);
  }
  for (int i = 0; i < mFSplines; i++) {
    char* buffer = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mCalibSplinesqTot[i].getFlatBufferPtr());
    mCalibSplinesqTot[i].setFutureBufferAddress(buffer);
  }
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

TPCdEdxCalibrationSplines* TPCdEdxCalibrationSplines::readFromFile(
  TFile& inpf, const char* name)
{
  /// read a class object from the file
  return FlatObject::readFromFile<TPCdEdxCalibrationSplines>(inpf, name);
}

int TPCdEdxCalibrationSplines::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  return FlatObject::writeToFile(*this, outf, name);
}
#endif
