/// \file DigitPad.h
/// \brief Definition of the Pad container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_DigitPad_H_
#define ALICEO2_TPC_DigitPad_H_

#include <map>

#define TPC_DIGIT_USEFAIRLINKS 1

#include "FairRootManager.h"
#ifdef TPC_DIGIT_USEFAIRLINKS
#include "FairMultiLinkedData.h"
#include "FairLink.h"
#endif
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

#ifdef TPC_DIGIT_USEFAIRLINKS
    /// Get the MC Links
    /// \return MC Links
    const FairMultiLinkedData& getMCLinks() const { return mMCLinks; }
#endif
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
    float                  mChargePad;   ///< Total accumulated charge on that pad for a given time bin
    unsigned char          mPad;         ///< Pad of the ADC value
#ifdef TPC_DIGIT_USEFAIRLINKS
    FairMultiLinkedData    mMCLinks;     ///< MC links
#endif
};

inline
DigitPad::DigitPad(int pad)
  : mChargePad(0.)
  , mPad(pad)
#ifdef TPC_DIGIT_USEFAIRLINKS
  , mMCLinks()
#endif
{}

inline 
void DigitPad::setDigit(size_t trackID, float charge)
{
  /// the MC ID is encoded such that we can have 999,999 tracks
  /// numbers larger than 1000000 correspond to the event ID
  /// i.e. 12000010 corresponds to event 12 with track ID 10
  /// \todo Faster would be a bit shift
#ifdef TPC_DIGIT_USEFAIRLINKS
  mMCLinks.AddLink(FairLink(-1, FairRootManager::Instance()->GetEntryNr(), "MCTrack", trackID));
#endif
  mChargePad += charge;
}

inline
void DigitPad::reset()
{
  mChargePad = 0;
#ifdef TPC_DIGIT_USEFAIRLINKS
  mMCLinks.Reset();
#endif
}
  
}
}

#endif // ALICEO2_TPC_DigitPad_H_
