// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/////////////////////////////////////////////////////////
//                                                     //
// Container Class for Signals                         //
//                                                     //
// Author:                                             //
//   Hermes Leon Vargas (hleon@ikf.uni-frankfurt.de)   //
//                                                     //
/////////////////////////////////////////////////////////

#include "TRDBase/TRDArraySignal.h"
#include "TRDBase/TRDFeeParam.h"
#include <fairlogger/Logger.h>

using namespace o2::trd;

//_______________________________________________________________________
TRDArraySignal::TRDArraySignal() = default;

//_______________________________________________________________________
TRDArraySignal::TRDArraySignal(int nrow, int ncol, int ntime)
{
  // TRDArraySignal constructor
  allocate(nrow, ncol, ntime);
}

//_______________________________________________________________________
TRDArraySignal::TRDArraySignal(const TRDArraySignal& d) = default;

//_______________________________________________________________________
TRDArraySignal::~TRDArraySignal() = default;

//________________________________________________________________________________
inline float TRDArraySignal::getData(int nrow, int ncol, int ntime) const
{
  // get the data using the pad numbering.
  // To access data using the mcm scheme use instead
  // the method getDataByAdcCol

  int corrcolumn = TRDFeeParam::instance()->padMcmLUT(ncol);

  return mSignal[(nrow * mNumberOfChannels + corrcolumn) * mNtime + ntime];
}

//________________________________________________________________________________
inline void TRDArraySignal::setData(int nrow, int ncol, int ntime, float value)
{
  // set the data using the pad numbering.
  // To write data using the mcm scheme use instead
  // the method setDataByAdcCol

  int colnumb = TRDFeeParam::instance()->padMcmLUT(ncol);

  mSignal[(nrow * mNumberOfChannels + colnumb) * mNtime + ntime] = value;
}

//_______________________________________________________________________
TRDArraySignal& TRDArraySignal::operator=(const TRDArraySignal& d)
{
  //
  // Assignment operator
  //

  if (this == &d) {
    return *this;
  }

  mSignal.clear();
  mNdet = d.mNdet;
  mNrow = d.mNrow;
  mNcol = d.mNcol;
  mNumberOfChannels = d.mNumberOfChannels;
  mNtime = d.mNtime;
  mNdim = d.mNdim;
  mSignal.clear();
  if (mSignal.size() != mNdim)
    mSignal.resize(mNdim);

  mSignal = d.mSignal;

  return *this;
}

//_______________________________________________________________________
void TRDArraySignal::allocate(int nrow, int ncol, int ntime)
{
  //
  // Allocates memory for an TRDArraySignal object with dimensions
  // Row*NumberOfNecessaryMCMs*ADCchannelsInMCM*Time
  // To be consistent with AliTRDarrayADC
  //

  mNrow = nrow;
  mNcol = ncol;
  mNtime = ntime;
  int adcchannelspermcm = TRDFeeParam::getNadcMcm();
  int padspermcm = TRDFeeParam::getNcolMcm();
  int numberofmcms = mNcol / padspermcm;
  mNumberOfChannels = numberofmcms * adcchannelspermcm;
  mNdim = nrow * mNumberOfChannels * ntime;
  if (mSignal.size() != mNdim)
    mSignal.resize(mNdim);

  memset(&mSignal[0], 0, sizeof(mSignal[0]) * mNdim);
}

//_______________________________________________________________________
int TRDArraySignal::getOverThreshold(float threshold) const
{
  //
  // get the number of entries over the threshold
  //

  int counter = 0;
  for (int i = 0; i < mNdim; i++) {
    if (mSignal[i] > threshold) {
      counter++;
    }
  }
  return counter;
}

//_______________________________________________________________________
void TRDArraySignal::compress(float minval)
{
  //
  // Compress the vector, setting values equal or
  // below minval to zero (minval>=0)
  //

  int counter = 0;
  int newDim = 0;
  int j;
  int r = 0;
  int k = 0;

  std::vector<int> longArr(mNdim);
  memset(&longArr[0], 0, sizeof(longArr[0]) * mNdim);

  //Initialize the vector

  for (int i = 0; i < mNdim; i++) {
    j = 0;
    if (mSignal[i] <= minval) {
      for (k = i; k < mNdim; k++) {
        if (mSignal[k] <= minval) {
          j = j + 1;
          longArr[r] = j;
        } else {
          break;
        }
      }
      r = r + 1;
    }
    i = i + j;
  }

  //Calculate the size of the compressed vector
  for (int i = 0; i < mNdim; i++) {
    if (longArr[i] != 0) {
      counter = counter + longArr[i] - 1;
    }
  }
  newDim = mNdim - counter; //New dimension

  //Fill the buffer of the compressed vector
  std::vector<float> buffer(mNdim);
  memset(&buffer[0], 0, sizeof(buffer[0]) * mNdim);
  int counterTwo = 0;

  //Write the new vector
  int g = 0;
  for (int i = 0; i < newDim; i++) {
    if (counterTwo < mNdim) {
      if (mSignal[counterTwo] > minval) {
        buffer[i] = mSignal[counterTwo];
      }
      if (mSignal[counterTwo] <= minval) {
        buffer[i] = -(longArr[g]);
        counterTwo = counterTwo + longArr[g] - 1;
        g++;
      }
      counterTwo++;
    }
  }

  //Copy the buffer
  if (mSignal.size() != newDim) {
    mSignal.resize(newDim);
  }
  LOG(debug) << "Compressed ArraySignal by " << mNdim / newDim;
  mNdim = newDim;
  mSignal = buffer;
  for (int i = 0; i < newDim; i++) {
    mSignal[i] = buffer[i];
  }
}

//_______________________________________________________________________
void TRDArraySignal::expand()
{
  // Expand the vector

  //Check if the vector has not been already expanded
  int verif = 0;
  for (int i = 0; i < mNdim; i++) {
    if (mSignal[i] < 0) {
      verif++;
    }
  }

  if (verif == 0) {
    return;
  }

  int dimexp = 0;
  std::vector<int> longArr(mNdim);

  memset(&longArr[0], 0, sizeof(longArr[0]) * mNdim);

  int r2 = 0;
  for (int i = 0; i < mNdim; i++) {
    if (mSignal[i] < 0) {
      longArr[r2] = (int)(-mSignal[i]);
      r2++;
    }
  }

  //Calculate new dimensions
  for (int i = 0; i < mNdim; i++) {
    if (longArr[i] != 0) {
      dimexp = dimexp + longArr[i] - 1;
    }
  }
  dimexp = dimexp + mNdim; //Dimension of the expanded vector

  //Write in the buffer the new vector
  int contaexp = 0;
  int h = 0;
  std::vector<float> buffer(dimexp);

  memset(&buffer[0], 0, sizeof(buffer[0]) * mNdim);

  for (int i = 0; i < dimexp; i++) {
    if (mSignal[contaexp] > 0) {
      buffer[i] = mSignal[contaexp];
    }
    if (mSignal[contaexp] < 0) {
      for (int j = 0; j < longArr[h]; j++) {
        buffer[i + j] = 0;
      }
      i = i + longArr[h] - 1;
      h++;
    }
    contaexp++;
  }

  if (mSignal.size() != dimexp)
    mSignal.resize(dimexp);
  mNdim = dimexp;
  mSignal = buffer;
}
//________________________________________________________________________________
void TRDArraySignal::reset()
{
  //
  // Reset the vector, the old contents are deleted
  // The vector keeps the same dimensions as before
  //

  memset(&mSignal[0], 0, sizeof(mSignal[0]) * mNdim);
}
