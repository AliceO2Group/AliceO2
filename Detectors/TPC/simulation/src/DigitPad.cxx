// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

#ifndef TPC_DIGIT_USEFAIRLINKS
    static std::vector<long> MClabels;
    MClabels.resize(0);
    DigitPad::processMClabels(MClabels);
#endif

    TClonesArray &clref = *output;
    const size_t digiPos = clref.GetEntriesFast();
    DigitMC *digit = new(clref[digiPos]) DigitMC(
#ifndef TPC_DIGIT_USEFAIRLINKS
						 MClabels,
#endif
						 cru, mADC, row, pad, timeBin, commonMode);
#ifdef TPC_DIGIT_USEFAIRLINKS
    digit->SetLinks(getMCLinks());
#endif
  }
}

#ifndef TPC_DIGIT_USEFAIRLINKS
void DigitPad::processMClabels(std::vector<long> &sortedMCLabels) const
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
#endif
