// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitMC.h
/// \brief Definition of the Monte Carlo Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitMC_H_
#define ALICEO2_TPC_DigitMC_H_

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>
#endif

#include "TPCBase/Digit.h"
#include "TPCSimulation/DigitPad.h"

#include "FairTimeStamp.h"

namespace boost {
namespace serialization {
class access; 
}
}

namespace o2 {
namespace TPC {

#ifdef TPC_DIGIT_USEFAIRLINKS
  using DigitBase = FairTimeStamp;
#else
  // A minimal (temporary) TimeStamp class, introduced here for
  // reducing memory consumption to a minimum.
  // This can be used only when MCtruth is not done using FairLinks.
  class TimeStamp : public TObject {
  public:
    TimeStamp() {}
    TimeStamp(int time) {
      // we use the TObjectID for the time
      SetUniqueID(time);
    }
    int GetTimeStamp() const { return TObject::GetUniqueID(); }
    ClassDef(TimeStamp, 1);
  };
  using DigitBase = TimeStamp;
#endif

/// \class DigitMC
/// This is the definition of the Monte Carlo Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.
/// Additional information attached to it are the MC label of the contributing tracks

class DigitMC : public DigitBase, public Digit {
  public:

    /// Default constructor
    DigitMC();

#ifdef TPC_DIGIT_USEFAIRLINKS
    /// Constructor, initializing values for position, charge, time and common mode
    /// \param MClabel std::vector containing the MC event and track ID encoded in a long int
    /// \param cru CRU of the DigitMC
    /// \param charge Accumulated charge of DigitMC
    /// \param row Row in which the DigitMC was created
    /// \param pad Pad in which the DigitMC was created
    /// \param time Time at which the DigitMC was created
    /// \param commonMode Common mode signal on that ROC in the time bin of the DigitMC. If not assigned, it is set to zero.
    DigitMC(int cru, float charge, int row, int pad, int time, float commonMode = 0.f);
#else
    /// Constructor, initializing values for position, charge, time and common mode
    /// \param MClabel std::vector containing the MC event and track ID encoded in a long int
    /// \param cru CRU of the DigitMC
    /// \param charge Accumulated charge of DigitMC
    /// \param row Row in which the DigitMC was created
    /// \param pad Pad in which the DigitMC was created
    /// \param time Time at which the DigitMC was created
    /// \param commonMode Common mode signal on that ROC in the time bin of the DigitMC. If not assigned, it is set to zero.
    DigitMC(std::vector<long> const &MClabel, int cru, float charge, int row, int pad, int time, float commonMode = 0.f);
#endif

    /// Destructor
    ~DigitMC() override = default;

    /// Get the timeBin of the DigitMC
    /// \return timeBin of the DigitMC
    int getTimeStamp() const final { return static_cast<int>(DigitBase::GetTimeStamp()); };

    /// Get the common mode signal of the DigitMC
    /// \return common mode signal of the DigitMC
    float getCommonMode() const { return mCommonMode; }

  private:
    #ifndef __CINT__
    friend class boost::serialization::access;
    #endif
#ifndef TPC_DIGIT_USEFAIRLINKS
    std::vector<long>       mMClabel;         ///< MC truth information to (multiple) event ID and track ID encoded in a long
#endif
    float                   mCommonMode;      ///< Common mode value of the DigitMC
      
  ClassDefOverride(DigitMC, 3);
};

inline
DigitMC::DigitMC()
  : DigitBase()
  , Digit(-1, -1.f, -1, -1)
  , mCommonMode(0.f)
#ifndef TPC_DIGIT_USEFAIRLINKS
  , mMClabel()
#endif
  {}

#ifdef TPC_DIGIT_USEFAIRLINKS
inline
DigitMC::DigitMC(int cru, float charge, int row, int pad, int time, float commonMode)
  : DigitBase(time)
  , Digit(cru, charge, row, pad)
  , mCommonMode(commonMode)
{}
#else
inline
DigitMC::DigitMC(std::vector<long> const &MClabel, int cru, float charge, int row, int pad, int time, float commonMode)
  : DigitBase(time)
  , Digit(cru, charge, row, pad)
  , mMClabel(MClabel)
  , mCommonMode(commonMode)
{}
#endif

}
}

#endif // ALICEO2_TPC_DigitMC_H_
