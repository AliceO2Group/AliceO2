/// \file DigitMC.h
/// \brief Definition of the Monte Carlo Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitMC_H_
#define ALICEO2_TPC_DigitMC_H_

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>
#endif

#include "TPCBase/Digit.h"

#include "FairTimeStamp.h"

namespace boost {
namespace serialization {
class access; 
}
}

namespace o2 {
namespace TPC {

/// \class DigitMC
/// This is the definition of the Monte Carlo Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.
/// Additional information attached to it are the MC label of the contributing tracks


class DigitMC : public FairTimeStamp, public Digit {
  public:

    /// Default constructor
    DigitMC();

    /// Constructor, initializing values for position, charge, time and common mode
    /// \param MClabel std::vector containing the MC event and track ID encoded in a long int
    /// \param cru CRU of the DigitMC
    /// \param charge Accumulated charge of DigitMC
    /// \param row Row in which the DigitMC was created
    /// \param pad Pad in which the DigitMC was created
    /// \param time Time at which the DigitMC was created
    /// \param commonMode Common mode signal on that ROC in the time bin of the DigitMC. If not assigned, it is set to zero.
    DigitMC(int cru, float charge, int row, int pad, int time, float commonMode = 0.f);

    /// Destructor
    ~DigitMC() override = default;

    /// Get the timeBin of the DigitMC
    /// \return timeBin of the DigitMC
    int getTimeStamp() const final { return static_cast<int>(FairTimeStamp::GetTimeStamp()); };

    /// Get the common mode signal of the DigitMC
    /// \return common mode signal of the DigitMC
    float getCommonMode() const { return mCommonMode; }

  private:
    #ifndef __CINT__
    friend class boost::serialization::access;
    #endif
    
    float                   mCommonMode;      ///< Common mode value of the DigitMC
      
  ClassDefOverride(DigitMC, 3);
};

inline
DigitMC::DigitMC()
  : FairTimeStamp()
  , Digit(-1, -1.f, -1, -1)
  , mCommonMode(0.f)
{}

inline
DigitMC::DigitMC(int cru, float charge, int row, int pad, int time, float commonMode)
  : FairTimeStamp(time)
  , Digit(cru, charge, row, pad)
  , mCommonMode(commonMode)
{}

}
}

#endif // ALICEO2_TPC_DigitMC_H_
