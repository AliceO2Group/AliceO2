// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "SimulationDataFormat/MCCompLabel.h"
#include "TPCSimulation/DigitMCMetaData.h"

#include <boost/range/adaptor/reversed.hpp>
#include <boost/bind.hpp>

using namespace o2::TPC;

void DigitPad::fillOutputContainer(TClonesArray *output, o2::dataformats::MCTruthContainer<long> &mcTruth, TClonesArray *debug, int cru, int timeBin, int row, int pad, float commonMode)
{
  /// The charge accumulated on that pad is converted into ADC counts, saturation of the SAMPA is applied and a Digit is created in written out
  const float totalADC = mChargePad - commonMode; // common mode is subtracted here in order to properly apply noise, pedestals and saturation of the SAMPA

  float noise = 0.f;
  float pedestal = 0.f;

  const float mADC = SAMPAProcessing::makeSignal(totalADC, PadSecPos(CRU(cru).sector(), PadPos(row, pad)), pedestal, noise);
  if(mADC > 0) {

    static std::vector<long> MClabels;
    MClabels.resize(0);
    DigitPad::processMClabels(MClabels);

    TClonesArray &clref = *output;
    const size_t digiPos = clref.GetEntriesFast();
    new(clref[digiPos]) DigitMC(MClabels, cru, mADC, row, pad, timeBin); /// create DigitMC

    for(int j=0; j<MClabels.size(); ++j) {
      // fill MCtruth output
      mcTruth.addElement(digiPos, MClabels[j]);
    }

    if(debug!=nullptr) {
      TClonesArray &clrefDebug = *debug;
      const size_t digiPosDebug = clrefDebug.GetEntriesFast();
      new(clrefDebug[digiPosDebug]) DigitMCMetaData(mChargePad, commonMode, pedestal, noise); /// create DigitMCMetaData
    }
  }
}

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
