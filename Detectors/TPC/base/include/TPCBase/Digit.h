// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digit.h
/// \brief Definition of the TPC Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DIGIT_H_
#define ALICEO2_TPC_DIGIT_H_

#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"
#include "CommonDataFormat/TimeStamp.h"
#ifndef __OPENCL__
#include <climits>
#endif

namespace o2
{
namespace tpc
{
/// \class Digit
/// This is the definition of the common Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the global pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.
using DigitBase = o2::dataformats::TimeStamp<int>;
class Digit : public DigitBase
{
 public:
  /// Default constructor
  GPUdDefault() Digit() CON_DEFAULT;

  /// Constructor, initializing values for position, charge, time and common mode
  /// \param cru CRU of the Digit
  /// \param charge Accumulated charge of Digit
  /// \param row Global pad row in which the Digit was created
  /// \param pad Pad in which the Digit was created
  GPUdi() Digit(int cru, float charge, int row, int pad, int time);

  /// Destructor
  GPUdDefault() ~Digit() CON_DEFAULT;

  /// Get the accumulated charged of the Digit in ADC counts.
  /// The conversion is such that the decimals are simply stripped
  /// \return charge of the Digit
  GPUdi() int getCharge() const { return static_cast<int>(mCharge); }

  /// Get the accumulated charged of the Digit as a float
  /// \return charge of the Digit as a float
  GPUdi() float getChargeFloat() const { return mCharge; }

  GPUdi() void setCharge(float q) { mCharge = q; }

  /// Get the CRU of the Digit
  /// \return CRU of the Digit
  GPUdi() int getCRU() const { return mCRU; }

  /// Get the global pad row of the Digit
  /// \return Global pad row of the Digit
  GPUdi() int getRow() const { return mRow; }

  /// Get the pad of the Digit
  /// \return pad of the Digit
  GPUdi() int getPad() const { return mPad; }

 protected:
  float mCharge = 0.f;      ///< ADC value of the Digit
  unsigned short mCRU = USHRT_MAX; ///< CRU of the Digit
  unsigned char mRow = UCHAR_MAX;  ///< Global pad row of the Digit
  unsigned char mPad = UCHAR_MAX;  ///< Pad of the Digit

  ClassDefNV(Digit, 1);
};

GPUdi() Digit::Digit(int cru, float charge, int row, int pad, int time)
  : DigitBase(time),
    mCharge(charge),
    mCRU(cru),
    mRow(row),
    mPad(pad)
{
}

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_DIGIT_H_
