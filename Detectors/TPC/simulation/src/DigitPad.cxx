/// \file DigitPad.cxx
/// \brief Implementation of the Pad container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitPad.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/PadSecPos.h"
#include "TPCBase/CRU.h"
#include "TPCSimulation/DigitMC.h"

#include <boost/range/adaptor/reversed.hpp>
#include <boost/bind.hpp>

using namespace o2::TPC;

void DigitPad::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad, float commonMode)
{
  /// The charge accumulated on that pad is converted into ADC counts, saturation of the SAMPA is applied and a Digit is created in written out
  const float totalADC = mChargePad - commonMode; // common mode is subtracted here in order to properly apply noise, pedestals and saturation of the SAMPA

  const float mADC = SAMPAProcessing::makeSignal(totalADC, PadSecPos(CRU(cru).sector(), PadPos(row, pad)));
  if(mADC > 0) {
    TClonesArray &clref = *output;
    const size_t digiPos = clref.GetEntriesFast();
    DigitMC *digit = new(clref[digiPos]) DigitMC(cru, mADC, row, pad, timeBin, commonMode);
#ifdef TPC_DIGIT_USEFAIRLINKS
    digit->SetLinks(getMCLinks());
#endif
  }
}

