// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawDataFrame.h
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF raw data frame

#ifndef O2_TOF_RAWDATAFRAME
#define O2_TOF_RAWDATAFRAME

namespace o2
{
namespace tof
{

class RawDataFrame
{

 public:
  RawDataFrame(int size = 1048576)
    : mSize(size), mBuffer(new char[size]){};
  RawDataFrame(const RawDataFrame& other)
    : mSize(other.mSize), mBuffer(new char[other.mSize])
  {
    for (int i = 0; i < mSize; ++i)
      mBuffer[i] = other.mBuffer[i];
  };
  RawDataFrame& operator=(const RawDataFrame& other)
  {
    if (&other == this)
      return *this;
    if (mSize != other.mSize) {
      delete[] mBuffer;
      mSize = other.mSize;
      mBuffer = new char[mSize];
    }
    for (int i = 0; i < mSize; ++i)
      mBuffer[i] = other.mBuffer[i];
    return *this;
  };
  ~RawDataFrame() { delete[] mBuffer; };
  int getSize() const { return mSize; };
  char* getBuffer() const { return mBuffer; };

  // private:

  int mSize;
  char* mBuffer; // [mSize]

  ClassDef(RawDataFrame, 1);
};

} // namespace tof
} // namespace o2

#endif /* O2_TOF_RAWDATAFRAME */
