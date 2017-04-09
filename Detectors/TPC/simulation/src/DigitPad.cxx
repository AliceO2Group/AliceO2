/// \file DigitPad.cxx
/// \brief Implementation of the Pad container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitPad.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/DigitMC.h"

#include <boost/range/adaptor/reversed.hpp>
#include <boost/bind.hpp>

using namespace o2::TPC;

void DigitPad::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad, float commonMode)
{
  /// The charge accumulated on that pad is converted into ADC counts, saturation of the SAMPA is applied and a Digit is created in written out
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();
  float totalADC = mChargePad;
  std::vector<long> MClabel;
  processMClabels(MClabel);
  const float mADC = sampa.getADCSaturation(totalADC-commonMode); // we substract the common mode here in order to properly apply the saturation of the FECs
  if(mADC > 0) {
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) DigitMC(MClabel, cru, mADC, row, pad, timeBin, commonMode);
  }
}

void DigitPad::processMClabels(std::vector<long> &sortedMCLabels)
{
  /// Dump the map into a vector of pairs
  std::vector<std::pair<long, int> > pairMClabels(mMCID.begin(), mMCID.end());
  /// Sort by the number of occurrences
  std::sort(pairMClabels.begin(), pairMClabels.end(), boost::bind(&std::pair<long, int>::second, _1) < boost::bind(&std::pair<long, int>::second, _2));
  // iterate backwards over the vector and hence write MC with largest number of occurrences as first into the sortedMClabels vector
  for(auto &aMCIDreversed : boost::adaptors::reverse(pairMClabels)) {
    sortedMCLabels.emplace_back(aMCIDreversed.first);
  }
}
