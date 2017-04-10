/// \file DigitData.h
/// \brief Definition of the Data Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitData_H_
#define ALICEO2_TPC_DigitData_H_

#include "TPCBase/Digit.h"

namespace o2 {
namespace TPC {

/// \class DigitData
/// This is the definition of the Data Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.


class DigitData : public Digit {
  public:

    /// Default constructor
    DigitData();

    /// Constructor, initializing values for position, charge, time and common mode
    /// \param MClabel std::vector containing the MC event and track ID encoded in a long int
    /// \param cru CRU of the DigitData
    /// \param charge Accumulated charge of DigitData
    /// \param row Row in which the DigitData was created
    /// \param pad Pad in which the DigitData was created
    /// \param time Time at which the DigitData was created
    /// \param commonMode Common mode signal on that ROC in the time bin of the DigitData. If not assigned, it is set to zero.
    DigitData(int cru, float charge, int row, int pad, int time);

    /// Destructor
    ~DigitData() override = default;

    /// Get the timeBin of the DigitData
    /// \return timeBin of the DigitData
    int getTimeStamp() const final { return mTimeBin; };

  private:
    unsigned int            mTimeBin;        ///< Time bin of that Digit

};

inline
DigitData::DigitData()
  : Digit(-1, -1.f, -1, -1)
  , mTimeBin(0)
{}

inline
DigitData::DigitData(int cru, float charge, int row, int pad, int time)
  : Digit(cru, charge, row, pad)
  , mTimeBin(time)
{}

}
}

#endif // ALICEO2_TPC_DigitData_H_
