// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitSector.h
/// \brief Definition of the Sector container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitSector_H_
#define ALICEO2_TPC_DigitSector_H_

#include "TPCSimulation/DigitTime.h"

#include <deque>

namespace o2 {
namespace TPC {

class DigitSector{
  public:

    /// Constructor
    DigitSector();

    /// Destructor
    ~DigitSector() = default;

    void init(const short sector, const TimeBin timeBinEvent);

    /// Resets the container
    void reset();

    short getSector() const { return mSector; }

    /// Get the size of the container
    /// \return Size of the time bin container
    size_t getSize() const {return mTimeBins.size();}

    /// Get the container
    /// \return container
    const std::vector<DigitTime>& getTimeBinContainer() const { return mTimeBins; }

    /// Add digit to the row container
    /// \param eventID MC Event ID
    /// \param hitID MC Hit ID
    /// \param timeBin Time bin of the digit
    /// \param row Pad row of digit
    /// \param pad Pad of digit
    /// \param charge Charge of the digit
    void setDigit(size_t eventID, size_t hitID, const CRU &cru, TimeBin timeBin, GlobalPadNumber globalPad, float charge);

    /// Fill output vector
    /// \param output Output container
    /// \param mcTruth MC Truth container
    /// \param debug Optional debug output container
    /// \param SectorID Sector ID
    /// \param eventTime time stamp of the event
    /// \param isContinuous Switch for continuous readout
    void fillOutputContainer(std::vector<Digit> *output, dataformats::MCTruthContainer<MCCompLabel> &mcTruth,
                             std::vector<DigitMCMetaData> *debug, Sector sector, TimeBin eventTime=0, bool isContinuous=true);

  private:
    short                  mSector;
    int                    mFirstTimeBin;     ///< First time bin to consider in that event
    int                    mEffectiveTimeBin; ///< Effective time bin of the digit sector
    int                    mNTimeBins;        ///< Maximal number of time bins in that Sector
    std::vector<DigitTime> mTimeBins;         ///< Time bin Container for the ADC value
};

inline
DigitSector::DigitSector()
  : mSector(-1),
    mFirstTimeBin(0),
    mEffectiveTimeBin(0),
    mNTimeBins(500),
    mTimeBins()
{}

inline
void DigitSector::init(const short sector, const TimeBin timeBinEvent)
{
  if(mSector == sector) return;
  else {
      mSector = sector;
      mFirstTimeBin = timeBinEvent;
      std::cout << mFirstTimeBin << "\n";
      reset();
    }
}

inline
void DigitSector::reset()
{
  for(auto &aTime : mTimeBins) {
      aTime.reset();
    }
}

}
}

#endif // ALICEO2_TPC_DigitSector_H_
