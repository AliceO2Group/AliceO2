// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

////////////////////////////////////////////////////////
//                                                    //
// Container class for ADC values                     //
// ported from aliroot                                //
//                                                    //
////////////////////////////////////////////////////////

#include <TMath.h>
#include <Rtypes.h>
#include "TRDBase/TRDCommonParam.h"
#include <fairlogger/Logger.h>
#include "TRDBase/TRDArrayADC.h"
#include "TRDBase/TRDCalPadStatus.h"
#include "TRDBase/TRDFeeParam.h"
#include "TRDBase/TRDSignalIndex.h"

using namespace o2::trd;

short* TRDArrayADC::mgLutPadNumbering = nullptr;

//____________________________________________________________________________________
TRDArrayADC::TRDArrayADC()
{
  //
  // TRDArrayADC default constructor
  //

  createLut();
}

//____________________________________________________________________________________
TRDArrayADC::TRDArrayADC(int nrow, int ncol, int ntime)
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

  mADC = new short[mNAdim];
  memcpy(mADC, b.mADC, mNAdim * sizeof(short));
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
  mADC = new short[mNAdim];
  memcpy(mADC, b.mADC, mNAdim * sizeof(short));

  return *this;
}

//____________________________________________________________________________________
void TRDArrayADC::allocate(int nrow, int ncol, int ntime)
{
  //
  // Allocate memory for an TRDArrayADC array with dimensions
  // Row*NumberOfNecessaryMCMs*ADCchannelsInMCM*Time
  //

  mNrow = nrow;
  mNcol = ncol;
  mNtime = ntime;
  int adcchannelspermcm = TRDFeeParam::getNadcMcm();
  int padspermcm = TRDFeeParam::getNcolMcm();
  int numberofmcms = mNcol / padspermcm;
  mNumberOfChannels = numberofmcms * adcchannelspermcm;
  mNAdim = nrow * mNumberOfChannels * ntime;

  if (mADC) {
    delete[] mADC;
  }

  mADC = new short[mNAdim];
  //std::fill(mADC.begin(), myVector.end(), 0);
  memset(mADC, 0, sizeof(short) * mNAdim);
}

////////////////////////////////////////////////////////////////////
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//________________________________________________________________________________
short* TRDArrayADC::getDataAddress(int nrow, int ncol, int ntime) const
{
  //
  // get the address of the given pad
  //

  int corrcolumn = mgLutPadNumbering[ncol];

  return &mADC[(nrow * mNumberOfChannels + corrcolumn) * mNtime + ntime];
}
//________________________________________________________________________________
short TRDArrayADC::getData(int nrow, int ncol, int ntime) const
{
  //
  // get the data using the pad numbering.
  // To access data using the mcm scheme use instead
  // the method getDataByAdcCol
  //

  int corrcolumn = mgLutPadNumbering[ncol];

  return mADC[(nrow * mNumberOfChannels + corrcolumn) * mNtime + ntime];
}
//________________________________________________________________________________
void TRDArrayADC::setData(int nrow, int ncol, int ntime, short value)
{
  //
  // set the data using the pad numbering.
  // To write data using the mcm scheme use instead
  // the method setDataByAdcCol
  //

  int colnumb = mgLutPadNumbering[ncol];

  mADC[(nrow * mNumberOfChannels + colnumb) * mNtime + ntime] = value;
}

void TRDArrayADC::getData(int r, int c, int t, int n, short* vals) const
{
  int colNum = mgLutPadNumbering[c];
  for (int ic = n, idx = (r * mNumberOfChannels + colNum) * mNtime + t; ic--; idx += mNtime)
    vals[ic] = mADC[idx];
}

//____________________________________________________________________________________
short TRDArrayADC::getDataBits(int row, int col, int time) const
{
  //
  // get the ADC value for a given position: row, col, time
  // Taking bit masking into account
  //
  // Adapted from code of the class TRDclusterizer
  //

  short tempval = getData(row, col, time);
  // Be aware of manipulations introduced by pad masking in the RawReader
  // Only output the manipulated Value
  CLRBIT(tempval, 10);
  CLRBIT(tempval, 11);
  CLRBIT(tempval, 12);
  return tempval;
}

//____________________________________________________________________________________
UChar_t TRDArrayADC::getPadStatus(int row, int col, int time) const
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
  short signal = getData(row, col, time);
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
void TRDArrayADC::setPadStatus(int row, int col, int time, UChar_t status)
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

  short signal = getData(row, col, time);

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
Bool_t TRDArrayADC::isPadCorrupted(int row, int col, int time)
{
  //
  // Checks if the pad has any masking as corrupted (Bit 10 in signal set)
  //
  // Adapted from code of the class TRDclusterizer
  //

  short signal = getData(row, col, time);
  return (signal > 0 && TESTBIT(signal, 10)) ? kTRUE : kFALSE;
}

//____________________________________________________________________________________
void TRDArrayADC::compress()
{
  //
  // Compress the array
  //

  if (mNAdim != mNrow * mNumberOfChannels * mNtime) {
    LOG (info) << "The ADC array is already compressed";
    return;
  }

  int counter = 0;
  int newDim = 0;
  int j;
  int l;
  int r = 0;
  int s = 0;
  int k = 0;

  int* longm = new int[mNAdim];
  int* longz = new int[mNAdim];

  if (longz && longm && mADC) {

    memset(longz, 0, sizeof(int) * mNAdim);
    memset(longm, 0, sizeof(int) * mNAdim);

    for (int i = 0; i < mNAdim; i++) {
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
    for (int i = 0; i < mNAdim; i++) {
      if (longm[i] != 0) {
        counter = counter + longm[i] - 1;
      }
      if (longz[i] != 0) {
        counter = counter + (longz[i] - 16001) - 1;
      }
    }

    int counterTwo = 0;
    newDim = mNAdim - counter; //Dimension of the compressed array
    short* buffer = new short[newDim];

    if (buffer) {

      //Fill the buffer of the compressed array
      int g = 0;
      int h = 0;
      for (int i = 0; i < newDim; i++) {
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
      mADC = new short[newDim];
      mNAdim = newDim;
      for (int i = 0; i < newDim; i++) {
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
void TRDArrayADC::Expand()
{
  //
  // Expand the array
  //

  if (mADC) {

    //Check if the array has not been already expanded
    int verif = 0;
    for (int i = 0; i < mNAdim; i++) {
      if (mADC[i] < -1) {
        verif++;
      }
    }

    if (verif == 0) {
      LOG(info) << "Nothing to expand";
      return;
    }

    int dimexp = 0;
    int* longz = new int[mNAdim];
    int* longm = new int[mNAdim];

    if (longz && longm) {

      //Initialize arrays
      memset(longz, 0, sizeof(int) * mNAdim);
      memset(longm, 0, sizeof(int) * mNAdim);
      int r2 = 0;
      int r3 = 0;
      for (int i = 0; i < mNAdim; i++) {
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
      for (int i = 0; i < mNAdim; i++) {
        if (longm[i] != 0) {
          dimexp = dimexp + longm[i] - 1;
        }
        if (longz[i] != 0) {
          dimexp = dimexp + longz[i] - 1;
        }
      }
      dimexp = dimexp + mNAdim;

      //Write in the buffer the new array
      int contaexp = 0;
      int h = 0;
      int l = 0;
      short* bufferE = new short[dimexp];
      if (bufferE) {
        for (int i = 0; i < dimexp; i++) {
          if (mADC[contaexp] > 0) {
            bufferE[i] = mADC[contaexp];
          }
          if ((mADC[contaexp] < 0) && (mADC[contaexp] >= -16000)) {
            for (int j = 0; j < longm[h]; j++) {
              bufferE[i + j] = -1;
            }
            i = i + longm[h] - 1;
            h++;
          }
          if (mADC[contaexp] < -16000) {
            for (int j = 0; j < longz[l]; j++) {
              bufferE[i + j] = 0;
            }
            i = i + longz[l] - 1;
            l++;
          }
          contaexp++;
        }
        //Copy the buffer
        delete[] mADC;
        mADC = new short[dimexp];
        mNAdim = dimexp;
        for (int i = 0; i < dimexp; i++) {
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
void TRDArrayADC::DeleteNegatives()
{

  //
  //This method modifies the digits array, changing the negative values (-1)
  //Produced during digitization into zero.
  //

  for (int a = 0; a < mNAdim; a++) {
    if (mADC[a] == -1) {
      mADC[a] = 0;
    }
  }
}
//________________________________________________________________________________
void TRDArrayADC::Reset()
{
  //
  // Reset the array, the old contents are deleted
  // The array keeps the same dimensions as before
  //

  memset(mADC, 0, sizeof(short) * mNAdim);
}
//________________________________________________________________________________
void TRDArrayADC::ConditionalReset(TRDSignalIndex* idx)
{
  //
  // Reset the array, the old contents are deleted
  // The array keeps the same dimensions as before
  //

  if (idx->getNoOfIndexes() > 25)
    memset(mADC, 0, sizeof(short) * mNAdim);
  else {
    int row, col;
    while (idx->NextRCIndex(row, col)) {
      int colnumb = mgLutPadNumbering[col];
      memset(&mADC[(row * mNumberOfChannels + colnumb) * mNtime], 0, mNtime);
    }
  }
}

//________________________________________________________________________________
void TRDArrayADC::CreateLut()
{
  //
  // Initializes the Look Up Table to relate
  // pad numbering and mcm channel numbering
  //

  if (mgLutPadNumbering)
    return;

  mgLutPadNumbering = new short[TRDFeeParam::getNcol()];
  memset(mgLutPadNumbering, 0, sizeof(short) * TRDFeeParam::getNcol());

  for (int mcm = 0; mcm < 8; mcm++) {
    int lowerlimit = 0 + mcm * 18;
    int upperlimit = 18 + mcm * 18;
    int shiftposition = 1 + 3 * mcm;
    for (int index = lowerlimit; index < upperlimit; index++) {
      mgLutPadNumbering[index] = index + shiftposition;
    }
  }
}
