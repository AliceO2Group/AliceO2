// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitCRU.h
/// \brief Definition of the CRU container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitCRU_H_
#define ALICEO2_TPC_DigitCRU_H_

#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/CommonModeContainer.h"

#include <deque>

class TClonesArray;

namespace o2 {
namespace TPC {

/// \class DigitCRU
/// This is the second class of the intermediate Digit Containers, in which all incoming electrons from the hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the Time Bin containers and is contained within the Digit Container.

class DigitCRU{
  public:
    
    /// Constructor
    /// \param mCRU CRU ID
    DigitCRU(int mCRU, CommonModeContainer &commonModeCont);

    /// Destructor
    ~DigitCRU() = default;

    /// Resets the container
    void reset();

    /// Get the number of entries in the container
    /// \return Number of entries in the time bin container
    int getNentries() const;

    /// Get the size of the container
    /// \return Size of the time bin container
    size_t getSize() const {return mTimeBins.size();}

    /// Get the container
    /// \return container
    const std::deque<std::unique_ptr<DigitTime>>& getTimeBinContainer() const { return mTimeBins; }

    /// Get the CRU ID
    /// \return CRU ID
    int getCRUID() const {return mCRU;}

    /// Add digit to the row container
    /// \param hitID MC Hit ID
    /// \param timeBin Time bin of the digit
    /// \param row Pad row of digit
    /// \param pad Pad of digit
    /// \param charge Charge of the digit
    void setDigit(size_t hitID, int timeBin, int row, int pad, float charge);

    /// Fill output TClonesArray
    /// \param output Output container
    /// \param mcTruth MC Truth container
    /// \param debug Optional debug output container
    /// \param cruID CRU ID
    /// \param eventTime time stamp of the event
    /// \param isContinuous Switch for continuous readout
    void fillOutputContainer(TClonesArray *output, o2::dataformats::MCTruthContainer<o2::MCCompLabel> &mcTruth, TClonesArray *debug, int cru, int eventTime=0, bool isContinuous=true);

  private:
    int                    mFirstTimeBin;
    int                    mEffectiveTimeBin;
    int                    mNTimeBins;        ///< Maximal number of time bins in that CRU
    unsigned short         mCRU;              ///< CRU of the ADC value
    std::deque<std::unique_ptr<DigitTime>> mTimeBins;         ///< Time bin Container for the ADC value
    CommonModeContainer    &mCommonModeContainer; ///< Reference to the common mode container
};
    
inline
DigitCRU::DigitCRU(int CRU, CommonModeContainer &commonModeCont)
  : mFirstTimeBin(0),
    mEffectiveTimeBin(0),
    mNTimeBins(500),
    mCRU(CRU),
    mTimeBins(),
    mCommonModeContainer(commonModeCont)
{}
    
inline 
void DigitCRU::reset()
{
//  for(auto &aTime : mTimeBins) {
//    if(aTime == nullptr) continue;
//    aTime->reset();
//  }
  mTimeBins.clear();
}
    
inline 
int DigitCRU::getNentries() const
{
  int counter = 0;
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    ++counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitCRU_H_
