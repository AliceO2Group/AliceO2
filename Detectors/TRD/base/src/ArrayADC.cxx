
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
#include <FairLogger.h>
#include "TRDBase/TRDCommonParam.h"
#include "TRDBase/ArrayADC.h"
#include "TRDBase/TRDCalPadStatus.h"
#include "TRDBase/TRDFeeParam.h"
#include "TRDBase/TRDSignalIndex.h"

using namespace o2::trd;

//____________________________________________________________________________________
ArrayADC::ArrayADC()
  : mNdet(0), mNrow(0), mNcol(0), mNumberOmChannels(0), mNtime(0), mNAdim(0), mADC(0)
{
  //
  // ArrayADC demault constructor
  //

  CreateLut();
}

//____________________________________________________________________________________
ArrayADC::ArrayADC(Int_t nrow, Int_t ncol, Int_t ntime)
  : mNdet(0), mNrow(0), mNcol(0), mNumberOmChannels(0), mNtime(0), mNAdim(0), mADC(0)
{
  //
  // ArrayADC constructor
  //

  CreateLut();
  Allocate(nrow, ncol, ntime);
}

//____________________________________________________________________________________
ArrayADC::ArrayADC(const ArrayADC& b)
  : mNdet(b.mNdet), mNrow(b.mNrow), mNcol(b.mNcol), mNumberOmChannels(b.mNumberOmChannels), mNtime(b.mNtime), mNAdim(b.mNAdim), mADC(b.mADC) //this will do the copy, due to begin a vector
{
  //
  // ArrayADC copy constructor
  //

  //copy done by vector constructor for fADC
}

//____________________________________________________________________________________
ArrayADC::~ArrayADC()
{
  //
  // ArrayADC destructor
  //
}

//____________________________________________________________________________________
ArrayADC& ArrayADC::operator=(const ArrayADC& b)
{
  //
  // Assignment operator
  //

  if (this == &b) {
    return *this;
  }
  mNdet = b.mNdet;
  mNrow = b.mNrow;
  mNcol = b.mNcol;
  mNumberOmChannels = b.mNumberOfChannels;
  mNtime = b.mNtime;
  mNAdim = b.mNAdim;
  mADC = b.mADC; //.resize(mNAdim); // resize incase b.mADC is bigger than this one;

  return *this;
}

//____________________________________________________________________________________
void ArrayADC::Allocate(Int_t nrow, Int_t ncol, Int_t ntime)
{
  //
  // Allocate memory for an ArrayADC array with dimensions
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

  mADC.clear();
}

////////////////////////////////////////////////////////////////////
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//________________________________________________________________________________
Short_t* ArrayADC::getDataAddress(Int_t nrow, Int_t ncol, Int_t ntime) const
{
  //
  // get the address of the given pad
  //
  // TODO is this function okay given a resize etc. of a vector can move its underlying storage ?
  Int_t corrcolumn = mgLutPadNumbering[ncol];
  Int_t offset = (nrow * mNumberOfChannels + corrcolumn) * mNtime + ntime;
  Short_t* mADCptr = (Short_t*)mADC.data();
  return &mADCptr[offset];
}
//________________________________________________________________________________
Short_t ArrayADC::getData(Int_t nrow, Int_t ncol, Int_t ntime) const
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
void ArrayADC::setData(Int_t nrow, Int_t ncol, Int_t ntime, Short_t value)
{
  //
  // set the data using the pad numbering.
  // To write data using the mcm scheme use instead
  // the method setDataByAdcCol
  //

  Int_t colnumb = mgLutPadNumbering[ncol];

  mADC[(nrow * mNumberOfChannels + colnumb) * mNtime + ntime] = value;
}

void ArrayADC::getData(Int_t r, Int_t c, Int_t t, Int_t n, Short_t* vals) const
{
  Int_t colNum = mgLutPadNumbering[c];
  for (Int_t ic = n, idx = (r * mNumberOfChannels + colNum) * mNtime + t; ic--; idx += mNtime)
    vals[ic] = mADC[idx];
}

//____________________________________________________________________________________
Short_t ArrayADC::getDataBits(Int_t row, Int_t col, Int_t time) const
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
UChar_t ArrayADC::getPadStatus(Int_t row, Int_t col, Int_t time) const
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
void ArrayADC::setPadStatus(Int_t row, Int_t col, Int_t time, UChar_t status)
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
Bool_t ArrayADC::IsPadCorrupted(Int_t row, Int_t col, Int_t time)
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
void ArrayADC::Compress()
{
  //
  // Compress the array
  //

  if (mNAdim != mNrow * mNumberOfChannels * mNtime) {
    LOG(INFO) << "The ADC array is already compressed";
    return;
  }

  Int_t counter = 0;
  Int_t newDim = 0;
  Int_t j;
  Int_t l;
  Int_t r = 0;
  Int_t s = 0;
  Int_t k = 0;

  std::vector<Int_t> longm;
  std::vector<Int_t> longz;

  longm.clear();
  longz.clear();

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
  std::vector<Short_t> buffer(newDim);

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
  mADC.resize(newDim);
  mNAdim = newDim;
  mADC = buffer;
}

//____________________________________________________________________________________
void ArrayADC::Expand()
{
  //
  // Expand the array
  //

  //Check if the array has not been already expanded
  Int_t verif = 0;
  for (Int_t i = 0; i < mNAdim; i++) {
    if (mADC[i] < -1) {
      verif++;
    }
  }

  if (verif == 0) {
    LOG(INFO) << "Nothing to expand";
    return;
  }

  Int_t dimexp = 0;
  std::vector<Int_t> longz(mNAdim);
  std::vector<Int_t> longm(mNAdim);

  //Initialize arrays
  longz.clear();
  longm.clear();
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
  std::vector<Short_t> buffer;

  for (Int_t i = 0; i < dimexp; i++) {
    if (mADC[contaexp] > 0) {
      buffer[i] = mADC[contaexp];
    }
    if ((mADC[contaexp] < 0) && (mADC[contaexp] >= -16000)) {
      for (Int_t j = 0; j < longm[h]; j++) {
        buffer[i + j] = -1;
      }
      i = i + longm[h] - 1;
      h++;
    }
    if (mADC[contaexp] < -16000) {
      for (Int_t j = 0; j < longz[l]; j++) {
        buffer[i + j] = 0;
      }
      i = i + longz[l] - 1;
      l++;
    }
    contaexp++;
  }
  //Copy the buffer
  mADC.resize(dimexp);
  mNAdim = dimexp;
  mADC = buffer;
}
//____________________________________________________________________________________
void ArrayADC::DeleteNegatives()
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
void ArrayADC::Reset()
{
  //
  // Reset the array, the old contents are deleted
  // The array keeps the same dimensions as before
  //

  std::fill(mADC.begin(), mADC.end(), 0);
}
//________________________________________________________________________________
void ArrayADC::ConditionalReset(TRDSignalIndex* idx)
{
  //
  // Reset the array, the old contents are deleted
  // The array keeps the same dimensions as before
  //

  if (idx->getNoOfIndexes() > 25)
    std::fill(mADC.begin(), mADC.begin() + mNAdim, 0);
  else {
    Int_t row, col;
    while (idx->NextRCIndex(row, col)) {
      Int_t colnumb = mgLutPadNumbering[col];
      auto mADCiterator = mADC.begin();
      auto offset = mADCiterator + mNtime * (row * mNumberOfChannels + colnumb);
      std::fill(offset, offset + mNtime, 0);
    }
  }
}

//________________________________________________________________________________
void ArrayADC::CreateLut()
{
  //
  // Initializes the Look Up Table to relate
  // pad numbering and mcm channel numbering
  //

  mgLutPadNumbering.resize(TRDFeeParam::getNcol());

  std::fill(mgLutPadNumbering.begin(), mgLutPadNumbering.begin() + TRDFeeParam::getNcol(), 0);

  for (Int_t mcm = 0; mcm < 8; mcm++) {
    Int_t lowerlimit = 0 + mcm * 18;
    Int_t upperlimit = 18 + mcm * 18;
    Int_t shiftposition = 1 + 3 * mcm;
    for (Int_t index = lowerlimit; index < upperlimit; index++) {
      mgLutPadNumbering[index] = index + shiftposition;
    }
  }
}
