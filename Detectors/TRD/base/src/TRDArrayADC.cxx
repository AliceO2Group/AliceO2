// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
//
// Container class for ADC values of the TRD Pads.
// originally from aliroot
//
// The interface will remain constant.
// The underlying storage mechanism remains to be optimised as of 2/2/2019
//
// The array is a 3d array mapped to a linear array.
// 2 space coordinates 1 time cordinate
// [row][columb][30 time bins]
//
// There are always 30 times per pad.
//
// This will be converted to a sparse matrix after
// due testing. This will be transparent to users.
//
// Compression and Expansion is a run length encoding
// with encodings in the negative numbers, real data is positive.
//
//
// NB
// Do Not use the getDataAddress as the pointer
// can not be guaranteed to remain constant.
// It will cause a fatal log error
//
//////////////////////////////////////////////////////////

#include <TMath.h>
#include <Rtypes.h>
#include "TRDBase/TRDCommonParam.h"
#include <fairlogger/Logger.h>
#include "TRDBase/TRDArrayADC.h"
#include "TRDBase/TRDCalPadStatus.h"
#include "TRDBase/TRDFeeParam.h"
#include "TRDBase/TRDSignalIndex.h"

using namespace o2::trd;

Short_t* TRDArrayADC::mgLutPadNumbering = nullptr;

//____________________________________________________________________________________
TRDArrayADC::TRDArrayADC()
{
  //
  // TRDArrayADC default constructor
  //

  createLut();
}

//____________________________________________________________________________________
TRDArrayADC::TRDArrayADC(Int_t nrow, Int_t ncol, Int_t ntime)
{
  //
  // TRDArrayADC constructor
  //

  createLut();
  allocate(nrow, ncol, ntime);
}

//____________________________________________________________________________________
TRDArrayADC::TRDArrayADC(const TRDArrayADC& b)
  : mNdet(b.mNdet), mNrow(b.mNrow), mNcol(b.mNcol), mNumberOfChannels(b.mNumberOfChannels), mNtime(b.mNtime), mNAdim(b.mNAdim)
{
  //
  // TRDArrayADC copy constructor
  //

  mADC = new Short_t[mNAdim];
  memcpy(mADC, b.mADC, mNAdim * sizeof(Short_t));
}

//____________________________________________________________________________________
TRDArrayADC::~TRDArrayADC()
{
  //
  // TRDArrayADC destructor
  //

  delete[] mADC;
  mADC = nullptr;
}

//____________________________________________________________________________________
TRDArrayADC& TRDArrayADC::operator=(const TRDArrayADC& b)
{
  //
  // Assignment operator
  //

  if (this == &b) {
    return *this;
  }
  if (mADC) {
    delete[] mADC;
  }
  mNdet = b.mNdet;
  mNrow = b.mNrow;
  mNcol = b.mNcol;
  mNumberOfChannels = b.mNumberOfChannels;
  mNtime = b.mNtime;
  mNAdim = b.mNAdim;
  mADC = new Short_t[mNAdim];
  memcpy(mADC, b.mADC, mNAdim * sizeof(Short_t));

  return *this;
}

//____________________________________________________________________________________
void TRDArrayADC::allocate(Int_t nrow, Int_t ncol, Int_t ntime)
{
  //
  // Allocate memory for an TRDArrayADC array with dimensions
  // Row*NumberOfNecessaryMCMs*ADCchannelsInMCM*Time
  //

  mNrow = nrow;
  mNcol = ncol;
  mNtime = ntime;
  Int_t adcchannelspermcm = TRDFeeParam::getNadcMcm();
  Int_t padspermcm = TRDFeeParam::getNcolMcm();
  Int_t numberofmcms = mNcol / padspermcm;
  mNumberOfChannels = numberofmcms * adcchannelspermcm;
  mNAdim = nrow * mNumberOfChannels * ntime;

  if (mADC) {
    delete[] mADC;
  }

  mADC = new Short_t[mNAdim];
  //std::fill(mADC.begin(), myVector.end(), 0);
  memset(mADC, 0, sizeof(Short_t) * mNAdim);
}

////////////////////////////////////////////////////////////////////
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//________________________________________________________________________________
Short_t* TRDArrayADC::getDataAddress(Int_t nrow, Int_t ncol, Int_t ntime) const
{
  //
  // get the address of the given pad
  //

  Int_t corrcolumn = mgLutPadNumbering[ncol];

  return &mADC[(nrow * mNumberOfChannels + corrcolumn) * mNtime + ntime];
}
//________________________________________________________________________________
Short_t TRDArrayADC::getData(Int_t nrow, Int_t ncol, Int_t ntime) const
{
  //
  // get the data using the pad numbering.
  // To access data using the mcm scheme use instead
  // the method getDataByAdcCol
  //

  Int_t corrcolumn = mgLutPadNumbering[ncol];

  return mADC[(nrow * mNumberOfChannels + corrcolumn) * mNtime + ntime];
}
//________________________________________________________________________________
void TRDArrayADC::setData(Int_t nrow, Int_t ncol, Int_t ntime, Short_t value)
{
  //
  // set the data using the pad numbering.
  // To write data using the mcm scheme use instead
  // the method setDataByAdcCol
  //

  Int_t colnumb = mgLutPadNumbering[ncol];

  mADC[(nrow * mNumberOfChannels + colnumb) * mNtime + ntime] = value;
}

void TRDArrayADC::getData(Int_t r, Int_t c, Int_t t, Int_t n, Short_t* vals) const
{
  Int_t colNum = mgLutPadNumbering[c];
  for (Int_t ic = n, idx = (r * mNumberOfChannels + colNum) * mNtime + t; ic--; idx += mNtime)
    vals[ic] = mADC[idx];
}

//____________________________________________________________________________________
Short_t TRDArrayADC::getDataBits(Int_t row, Int_t col, Int_t time) const
{
  //
  // get the ADC value for a given position: row, col, time
  // Taking bit masking into account
  //
  // Adapted from code of the class TRDclusterizer
  //

  Short_t tempval = getData(row, col, time);
  // Be aware of manipulations introduced by pad masking in the RawReader
  // Only output the manipulated Value
  CLRBIT(tempval, 10);
  CLRBIT(tempval, 11);
  CLRBIT(tempval, 12);
  return tempval;
}

//____________________________________________________________________________________
UChar_t TRDArrayADC::getPadStatus(Int_t row, Int_t col, Int_t time) const
{
  //
  // Returns the pad status stored in the pad signal
  //
  // Output is a UChar_t value
  // Status Codes:
  //               Noisy Masking:           2
  //               Bridged Left Masking     8
  //               Bridged Right Masking    8
  //               Not Connected Masking Digits
  //
  // Adapted from code of the class TRDclusterizer
  //

  UChar_t padstatus = 0;
  Short_t signal = getData(row, col, time);
  if (signal > 0 && TESTBIT(signal, 10)) {
    if (signal & 0x800)    //TESTBIT(signal, 11))
      if (signal & 0x1000) //TESTBIT(signal, 12))
        padstatus = TRDCalPadStatus::kPadBridgedRight;
      else
        padstatus = TRDCalPadStatus::kNotConnected;
    else if (signal & 0x1000) //TESTBIT(signal, 12))
      padstatus = TRDCalPadStatus::kPadBridgedLeft;
    else
      padstatus = TRDCalPadStatus::kMasked;
  }

  return padstatus;
}

//____________________________________________________________________________________
void TRDArrayADC::setPadStatus(Int_t row, Int_t col, Int_t time, UChar_t status)
{
  //
  // Setting the pad status into the signal using the Bits 10 to 14
  // (currently used: 10 to 12)
  //
  // Input codes (Unsigned char):
  //               Noisy Masking:           2
  //               Bridged Left Masking     8
  //               Bridged Right Masking    8
  //               Not Connected Masking    32
  //
  // Status codes: Any masking:             Bit 10(1)
  //               Noisy masking:           Bit 11(0), Bit 12(0)
  //               No Connection masking:   Bit 11(1), Bit 12(0)
  //               Bridged Left masking:    Bit 11(0), Bit 12(1)
  //               Bridged Right masking:   Bit 11(1), Bit 12(1)
  //
  // Adapted from code of the class TRDclusterizer
  //

  Short_t signal = getData(row, col, time);

  // Only set the Pad Status if the signal is > 0
  if (signal > 0) {
    switch (status) {
      case TRDCalPadStatus::kMasked:
        SETBIT(signal, 10);
        CLRBIT(signal, 11);
        CLRBIT(signal, 12);
        break;
      case TRDCalPadStatus::kNotConnected:
        SETBIT(signal, 10);
        SETBIT(signal, 11);
        CLRBIT(signal, 12);
        break;
      case TRDCalPadStatus::kPadBridgedLeft:
        SETBIT(signal, 10);
        CLRBIT(signal, 11);
        SETBIT(signal, 12);
        break;
      case TRDCalPadStatus::kPadBridgedRight:
        SETBIT(signal, 10);
        SETBIT(signal, 11);
        SETBIT(signal, 12);
        break;
      default:
        CLRBIT(signal, 10);
        CLRBIT(signal, 11);
        CLRBIT(signal, 12);
    }
    setData(row, col, time, signal);
  }
}

//____________________________________________________________________________________
Bool_t TRDArrayADC::isPadCorrupted(Int_t row, Int_t col, Int_t time)
{
  //
  // Checks if the pad has any masking as corrupted (Bit 10 in signal set)
  //
  // Adapted from code of the class TRDclusterizer
  //

  Short_t signal = getData(row, col, time);
  return (signal > 0 && TESTBIT(signal, 10)) ? kTRUE : kFALSE;
}

//____________________________________________________________________________________
void TRDArrayADC::compress()
{
  //
  // Compress the array
  //

  if (mNAdim != mNrow * mNumberOfChannels * mNtime) {
    LOG(info) << "The ADC array is already compressed";
    return;
  }

  Int_t counter = 0;
  Int_t newDim = 0;
  Int_t j;
  Int_t l;
  Int_t r = 0;
  Int_t s = 0;
  Int_t k = 0;

  Int_t* longm = new Int_t[mNAdim];
  Int_t* longz = new Int_t[mNAdim];

  if (longz && longm && mADC) {

    memset(longz, 0, sizeof(Int_t) * mNAdim);
    memset(longm, 0, sizeof(Int_t) * mNAdim);

    for (Int_t i = 0; i < mNAdim; i++) {
      j = 0;
      if (mADC[i] == -1) {
        for (k = i; k < mNAdim; k++) {
          if ((mADC[k] == -1) && (j < 16000)) {
            j = j + 1;
            longm[r] = j;
          } else {
            break;
          }
        }
        r = r + 1;
      }
      l = 16001;
      if (mADC[i] == 0) {
        for (k = i; k < mNAdim; k++) {
          if ((mADC[k] == 0) && (l < 32767)) {
            l = l + 1;
            longz[s] = l;
          } else {
            break;
          }
        }
        s = s + 1;
      }
      if (mADC[i] > 0) {
        i = i + 1;
      }
      i = i + j + (l - 16001 - 1);
    }

    //Calculate the size of the compressed array
    for (Int_t i = 0; i < mNAdim; i++) {
      if (longm[i] != 0) {
        counter = counter + longm[i] - 1;
      }
      if (longz[i] != 0) {
        counter = counter + (longz[i] - 16001) - 1;
      }
    }

    Int_t counterTwo = 0;
    newDim = mNAdim - counter; //Dimension of the compressed array
    Short_t* buffer = new Short_t[newDim];

    if (buffer) {

      //Fill the buffer of the compressed array
      Int_t g = 0;
      Int_t h = 0;
      for (Int_t i = 0; i < newDim; i++) {
        if (counterTwo < mNAdim) {
          if (mADC[counterTwo] > 0) {
            buffer[i] = mADC[counterTwo];
          }
          if (mADC[counterTwo] == -1) {
            buffer[i] = -(longm[g]);
            counterTwo = counterTwo + longm[g] - 1;
            g++;
          }
          if (mADC[counterTwo] == 0) {
            buffer[i] = -(longz[h]);
            counterTwo = counterTwo + (longz[h] - 16001) - 1;
            h++;
          }
          counterTwo++;
        }
      }

      //Copy the buffer
      delete[] mADC;
      mADC = nullptr;
      mADC = new Short_t[newDim];
      mNAdim = newDim;
      for (Int_t i = 0; i < newDim; i++) {
        mADC[i] = buffer[i];
      }

      //Delete auxiliary arrays
      delete[] buffer;
      buffer = nullptr;
    }
  }

  if (longz) {
    delete[] longz;
    longz = nullptr;
  }
  if (longm) {
    delete[] longm;
    longm = nullptr;
  }
}

//____________________________________________________________________________________
void TRDArrayADC::expand()
{
  //
  // Expand the array
  //

  if (mADC) {

    //Check if the array has not been already expanded
    Int_t verif = 0;
    for (Int_t i = 0; i < mNAdim; i++) {
      if (mADC[i] < -1) {
        verif++;
      }
    }

    if (verif == 0) {
      LOG(info) << "Nothing to expand";
      return;
    }

    Int_t dimexp = 0;
    Int_t* longz = new Int_t[mNAdim];
    Int_t* longm = new Int_t[mNAdim];

    if (longz && longm) {

      //Initialize arrays
      memset(longz, 0, sizeof(Int_t) * mNAdim);
      memset(longm, 0, sizeof(Int_t) * mNAdim);
      Int_t r2 = 0;
      Int_t r3 = 0;
      for (Int_t i = 0; i < mNAdim; i++) {
        if ((mADC[i] < 0) && (mADC[i] >= -16000)) {
          longm[r2] = -mADC[i];
          r2++;
        }
        if (mADC[i] < -16000) {
          longz[r3] = -mADC[i] - 16001;
          r3++;
        }
      }

      //Calculate the new dimensions of the array
      for (Int_t i = 0; i < mNAdim; i++) {
        if (longm[i] != 0) {
          dimexp = dimexp + longm[i] - 1;
        }
        if (longz[i] != 0) {
          dimexp = dimexp + longz[i] - 1;
        }
      }
      dimexp = dimexp + mNAdim;

      //Write in the buffer the new array
      Int_t contaexp = 0;
      Int_t h = 0;
      Int_t l = 0;
      Short_t* bufferE = new Short_t[dimexp];
      if (bufferE) {
        for (Int_t i = 0; i < dimexp; i++) {
          if (mADC[contaexp] > 0) {
            bufferE[i] = mADC[contaexp];
          }
          if ((mADC[contaexp] < 0) && (mADC[contaexp] >= -16000)) {
            for (Int_t j = 0; j < longm[h]; j++) {
              bufferE[i + j] = -1;
            }
            i = i + longm[h] - 1;
            h++;
          }
          if (mADC[contaexp] < -16000) {
            for (Int_t j = 0; j < longz[l]; j++) {
              bufferE[i + j] = 0;
            }
            i = i + longz[l] - 1;
            l++;
          }
          contaexp++;
        }
        //Copy the buffer
        delete[] mADC;
        mADC = new Short_t[dimexp];
        mNAdim = dimexp;
        for (Int_t i = 0; i < dimexp; i++) {
          mADC[i] = bufferE[i];
        }

        delete[] bufferE;
      }

      //Delete auxiliary arrays
      delete[] longm;
      delete[] longz;
    }
  }
}
//____________________________________________________________________________________
void TRDArrayADC::deleteNegatives()
{

  //
  //This method modifies the digits array, changing the negative values (-1)
  //Produced during digitization into zero.
  //

  for (Int_t a = 0; a < mNAdim; a++) {
    if (mADC[a] == -1) {
      mADC[a] = 0;
    }
  }
}
//________________________________________________________________________________
void TRDArrayADC::reset()
{
  //
  // Reset the array, the old contents are deleted
  // The array keeps the same dimensions as before
  //

  memset(mADC, 0, sizeof(Short_t) * mNAdim);
}
//________________________________________________________________________________
void TRDArrayADC::conditionalReset(TRDSignalIndex* idx)
{
  //
  // Reset the array, the old contents are deleted
  // The array keeps the same dimensions as before
  //

  if (idx->getNoOfIndexes() > 25)
    memset(mADC, 0, sizeof(Short_t) * mNAdim);
  else {
    Int_t row, col;
    while (idx->nextRCIndex(row, col)) {
      Int_t colnumb = mgLutPadNumbering[col];
      memset(&mADC[(row * mNumberOfChannels + colnumb) * mNtime], 0, mNtime);
    }
  }
}

//________________________________________________________________________________
void TRDArrayADC::createLut()
{
  //
  // Initializes the Look Up Table to relate
  // pad numbering and mcm channel numbering
  //

  if (mgLutPadNumbering)
    return;

  mgLutPadNumbering = new Short_t[TRDFeeParam::getNcol()];
  memset(mgLutPadNumbering, 0, sizeof(Short_t) * TRDFeeParam::getNcol());

  for (Int_t mcm = 0; mcm < 8; mcm++) {
    Int_t lowerlimit = 0 + mcm * 18;
    Int_t upperlimit = 18 + mcm * 18;
    Int_t shiftposition = 1 + 3 * mcm;
    for (Int_t index = lowerlimit; index < upperlimit; index++) {
      mgLutPadNumbering[index] = index + shiftposition;
    }
  }
}
