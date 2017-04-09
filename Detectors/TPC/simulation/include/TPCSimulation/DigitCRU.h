/// \file DigitCRU.h
/// \brief Definition of the CRU container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitCRU_H_
#define ALICEO2_TPC_DigitCRU_H_

#include "DigitTime.h"
#include "CommonMode.h"

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
    DigitCRU(int mCRU);

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

    /// Get the CRU ID
    /// \return CRU ID
    int getCRUID() const {return mCRU;}

    /// Add digit to the row container
    /// \param eventID MC ID of the event
    /// \param trackID MC ID of the track
    /// \param timeBin Time bin of the digit
    /// \param row Pad row of digit
    /// \param pad Pad of digit
    /// \param charge Charge of the digit
    void setDigit(int eventID, int trackID, int timeBin, int row, int pad, float charge);

    /// Fill output TClonesArray
    /// \param output Output container
    /// \param cruID CRU ID
    void fillOutputContainer(TClonesArray *output, int cru, int eventTime=0);

    /// Fill output TClonesArray
    /// \param output Output container
    /// \param cruID CRU ID
    void fillOutputContainer(TClonesArray *output, int cru, std::vector<CommonMode> &commonModeContainer);

    /// Process Common Mode Information
    /// \param output Output container
    /// \param cruID CRU ID
    void processCommonMode(std::vector<CommonMode> &, int cru);

  private:
    int                    mFirstTimeBin;
    int                    mTimeBinLastEvent;
    int                    mEffectiveTimeBin;
    int                    mNTimeBins;        ///< Maximal number of time bins in that CRU
    unsigned short         mCRU;              ///< CRU of the ADC value
    std::deque<std::unique_ptr<DigitTime>> mTimeBins;         ///< Time bin Container for the ADC value
};
    
inline
DigitCRU::DigitCRU(int CRU)
  : mFirstTimeBin(0)
  , mTimeBinLastEvent(0)
  , mEffectiveTimeBin(0)
  , mNTimeBins(500)
  , mCRU(CRU)
{}
    
inline 
void DigitCRU::reset()
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->reset();
  }
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
