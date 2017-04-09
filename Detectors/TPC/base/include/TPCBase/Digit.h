/// \file Digit.h
/// \brief Definition of the Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_Digit_H_
#define ALICEO2_TPC_Digit_H_

namespace o2 {
namespace TPC {

/// \class Digit
/// This is the definition of the common Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.

class Digit {
  public:

    /// Default constructor
    Digit();

    /// Constructor, initializing values for position, charge, time and common mode
    /// \param cru CRU of the Digit
    /// \param charge Accumulated charge of Digit
    /// \param row Row in which the Digit was created
    /// \param pad Pad in which the Digit was created
    Digit(int cru, float charge, int row, int pad);

    /// Destructor
    virtual ~Digit()= default;

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

    /// Get the timeBin of the Digit
    /// \return timeBin of the Digit
    virtual int getTimeStamp() const = 0;


  protected:

    float                   mCharge;          ///< ADC value of the Digit
    unsigned short          mCRU;             ///< CRU of the Digit
    unsigned char           mRow;             ///< Row of the Digit
    unsigned char           mPad;             ///< Pad of the Digit

};

inline
Digit::Digit()
  : mCharge(0.f)
  , mCRU(-1)
  , mRow(-1)
  , mPad(-1)
{}

inline
Digit::Digit(int cru, float charge, int row, int pad)
  : mCharge(charge)
  , mCRU(cru)
  , mRow(row)
  , mPad(pad)
{}

}
}

#endif // ALICEO2_TPC_Digit_H_
