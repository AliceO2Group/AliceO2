// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitPad.h
/// \brief Definition of the Pad container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_DigitPad_H_
#define ALICEO2_TPC_DigitPad_H_

#include <map>

#include "FairRootManager.h"
#include "TPCSimulation/CommonMode.h"
#include <TClonesArray.h>

namespace o2 {
namespace TPC {

/// \class DigitPad
/// This is the fifth and lowest class of the intermediate Digit Containers, in which all incoming electrons from the hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the individual pad containers and is contained within the Row Container.

class DigitPad{
  public:

    /// Constructor
    /// \param mPad Pad ID
    DigitPad(int mPad);

    /// Destructor
    ~DigitPad() = default;

    /// Resets the container
    void reset();

    /// Get the Pad ID
    /// \return Pad ID
    int getPad() const {return mPad;}

    /// Get the accumulated charge on that pad
    /// \return Accumulated charge
    float getChargePad() const {return mChargePad;}

    /// Add digit to the time bin container
    /// \param hitID MC Hit ID
    /// \param charge Charge of the digit
    void setDigit(size_t hitID, float charge);

    /// Fill output TClonesArray
    /// \param output Output container
    /// \param cru CRU ID
    /// \param timeBin Time bin
    /// \param row Row ID
    /// \param pad pad ID
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad, float commonMode = 0);

  private:
    /// The MC labels are sorted by occurrence such that the event/track combination with the largest number of occurrences is first
    /// This is then dumped into a std::vector and attached to the digits
    /// \todo Find out how many different event/track combinations are relevant
    /// \param std::vector containing the sorted MCLabels
    void processMClabels(std::vector<long> &sortedMCLabels) const;

    float                  mChargePad;   ///< Total accumulated charge on that pad for a given time bin
    unsigned char          mPad;         ///< Pad of the ADC value
    // according to event + trackID + sorted according to most probable
    std::map<long, int>    mMCID;        //! Map containing the MC labels (key) and the according number of occurrence (value)

    // TODO: optimize this treatment, for example by using a structure like this
    // struct MCIDValue {
    //   unsigned int eventId : 15; // 32k event Id possible
    //   unsigned int trackId: 17; // 128K tracks possible
    //   unsigned int occurences : 32; // 4G occurrences possible
    // }
    // std::vector<MCID> mMCID;
};

inline
DigitPad::DigitPad(int pad)
  : mChargePad(0.)
  , mPad(pad)
  , mMCID()
{}

inline 
void DigitPad::setDigit(size_t trackID, float charge)
{
  static FairRootManager *mgr = FairRootManager::Instance();
  const int eventID = mgr->GetEntryNr();
  /// the MC ID is encoded such that we can have 999,999 tracks
  /// numbers larger than 1000000 correspond to the event ID
  /// i.e. 12000010 corresponds to event 12 with track ID 10
  /// \todo Faster would be a bit shift
  ++mMCID[(eventID)*1000000 + trackID];
  mChargePad += charge;
}

inline
void DigitPad::reset()
{
  mChargePad = 0;
  mMCID.clear();
}
  
}
}

#endif // ALICEO2_TPC_DigitPad_H_
