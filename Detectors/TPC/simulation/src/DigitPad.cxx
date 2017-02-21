/// \file DigitPad.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitPad.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/Digit.h"

#include <boost/range/adaptor/reversed.hpp>
#include <boost/bind.hpp>

using namespace AliceO2::TPC;

DigitPad::DigitPad(int pad)
  : mChargePad(0.)
  , mPad(pad)
  , mMCID()
{}

DigitPad::~DigitPad()
{
  mMCID.resize(0);
  mChargePad = 0;
}

void DigitPad::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad)
{  
  float totalADC = mChargePad;
  std::vector<long> MClabel;
  processMClabels(MClabel);
  const float mADC = SAMPAProcessing::getADCSaturation(totalADC);
  if(mADC > 0) {
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(MClabel, cru, mADC, row, pad, timeBin);
  }
}

void DigitPad::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad, float commonMode)
{
  float totalADC = mChargePad;
  std::vector<long> MClabel;
  processMClabels(MClabel);
  const float mADC = SAMPAProcessing::getADCSaturation(totalADC);
  if(mADC > 0) {
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(MClabel, cru, mADC, row, pad, timeBin, commonMode);
  }
}

void DigitPad::processMClabels(std::vector<long> &sortedMCLabels)
{
  // The MC labels encoded as described in the header are sorted by the number of occurence and a vector with the such sorted labels is returned
  
  std::map<long, int> frequencyMap;   //map containing the MC labels (key) and the according number of occurrence (value)
  // dump the MC labels in a map and increase the value with the number of occurrences.
  for (auto &aMCID : mMCID) {
    ++frequencyMap[aMCID];
  } 
  // Dump the map into a vector of pairs
  std::vector<std::pair<long, int> > pairMClabels(frequencyMap.begin(), frequencyMap.end());
  // Sort by the number of occurrences
  std::sort(pairMClabels.begin(), pairMClabels.end(), 
            boost::bind(&std::pair<long, int>::second, _1) < boost::bind(&std::pair<long, int>::second, _2));
  // iterate backwards over the vector and hence write MC with largest number of occurrences as first into the sortedMClabels vector
  for(auto &aMCIDreversed : boost::adaptors::reverse(pairMClabels)) {
    sortedMCLabels.emplace_back(aMCIDreversed.first);
  }
}