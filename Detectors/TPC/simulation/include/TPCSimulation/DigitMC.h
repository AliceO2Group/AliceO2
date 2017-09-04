// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitMC.h
/// \brief Definition of the Monte Carlo Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitMC_H_
#define ALICEO2_TPC_DigitMC_H_

#include "TPCBase/Digit.h"
#include "TPCSimulation/DigitPad.h"

#include "FairTimeStamp.h"

namespace o2 {
namespace TPC {

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

/// \class DigitMC
/// This is the definition of the Monte Carlo Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.
/// Additional information attached to it are the MC label of the contributing tracks

class DigitMC : public DigitBase, public Digit {
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
    DigitMC(std::vector<long> const &MClabel, int cru, float charge, int row, int pad, int time);

    /// Destructor
    ~DigitMC() override = default;

    /// Get the timeBin of the DigitMC
    /// \return timeBin of the DigitMC
    int getTimeStamp() const final { return static_cast<int>(DigitBase::GetTimeStamp()); };

    /// Get the number of MC labels associated to the DigitMC
    /// \return Number of MC labels associated to the DigitMC
    size_t getNumberOfMClabels() const { return mMClabel.size(); }

    /// Get a specific MC Event ID
    /// \param iOccurrence Sorted by occurrence, i.e. for iOccurrence=0 the MC event ID of the most dominant track
    /// \return MC Event ID
    int getMCEvent(int iOccurrence) const { return static_cast<int>(mMClabel[iOccurrence]*1E-6); }

    /// Get a specific MC Track ID
    /// \param iOccurrence Sorted by occurrence, i.e. for iOccurrence=0 the MC ID of the most dominant track
    /// \return MC Track ID
    int getMCTrack(int iOccurrence) const { return static_cast<int>((mMClabel[iOccurrence])%int(1E6)); }

  private:
    std::vector<long>       mMClabel;         ///< MC truth information to (multiple) event ID and track ID encoded in a long
      
  ClassDefOverride(DigitMC, 3);
};

inline
DigitMC::DigitMC()
  : DigitBase(),
    Digit(-1, -1.f, -1, -1),
    mMClabel()
  {}

inline
DigitMC::DigitMC(std::vector<long> const &MClabel, int cru, float charge, int row, int pad, int time)
  : DigitBase(time),
    Digit(cru, charge, row, pad),
    mMClabel(MClabel)
{}

}
}

#endif // ALICEO2_TPC_DigitMC_H_
