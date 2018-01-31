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

#include "SimulationDataFormat/TimeStamp.h"
#include "TObject.h"

namespace o2 {
namespace TPC
{
/// \class Digit
/// This is the definition of the common Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.
using DigitBase = o2::dataformats::TimeStamp<int>;
class Digit : public DigitBase {
  public:

    /// Default constructor
    Digit();

    /// Constructor, initializing values for position, charge, time and common mode
    /// \param cru CRU of the Digit
    /// \param charge Accumulated charge of Digit
    /// \param row Row in which the Digit was created
    /// \param pad Pad in which the Digit was created
    Digit(int cru, float charge, int row, int pad, int time);

    /// Destructor
    ~Digit() = default;

    // Add charge to an existing digit
    /// \param charge Charge to be added to the digit
    void addCharge(float charge) { mCharge += charge; }

    /// Get the accumulated charged of the Digit in ADC counts.
    /// The conversion is such that the decimals are simply stripped
    /// \return charge of the Digit
    int getCharge() const { return static_cast<int>(mCharge); }

    /// Get the accumulated charged of the Digit as a float
    /// \return charge of the Digit as a float
    float getChargeFloat() const { return mCharge; }

    /// Get the CRU of the Digit
    /// \return CRU of the Digit
    int getCRU() const { return mCRU; }

    /// Get the pad row of the Digit
    /// \return pad row of the Digit
    int getRow() const { return mRow; }

    /// Get the pad of the Digit
    /// \return pad of the Digit
    int getPad() const { return mPad; }

  protected:

    float                   mCharge;          ///< ADC value of the Digit
    unsigned short          mCRU;             ///< CRU of the Digit
    unsigned char           mRow;             ///< Row of the Digit
    unsigned char           mPad;             ///< Pad of the Digit


    ClassDefNV(Digit, 1);
};

inline
Digit::Digit()
  : DigitBase(),
    mCharge(0.f),
    mCRU(-1),
    mRow(-1),
    mPad(-1)
{}

inline
Digit::Digit(int cru, float charge, int row, int pad, int time)
  : DigitBase(time),
    mCharge(charge),
    mCRU(cru),
    mRow(row),
    mPad(pad)
{}

}
}

#endif // ALICEO2_TPC_DIGIT_H_
