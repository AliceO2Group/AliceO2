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
#include "TPCBase/Digit.h"
#include "TPCSimulation/DigitMCMetaData.h"

#include <boost/range/adaptor/reversed.hpp>
#include <boost/bind.hpp>
#include <cassert>

using namespace o2::TPC;

unsigned int DigitPad::sID = 0; // a global id counter
o2::dataformats::LabelContainer<std::pair<o2::MCCompLabel, int>, false> DigitPad::sLabels;

void DigitPad::fillOutputContainer(TClonesArray *output, o2::dataformats::MCTruthContainer<o2::MCCompLabel> &mcTruth, TClonesArray *debug, int cru, int timeBin, int row, int pad, float commonMode)
{
  /// The charge accumulated on that pad is converted into ADC counts, saturation of the SAMPA is applied and a Digit is created in written out
  const float totalADC = mChargePad - commonMode; // common mode is subtracted here in order to properly apply noise, pedestals and saturation of the SAMPA

  float noise = 0.f;
  float pedestal = 0.f;

  const float mADC = SAMPAProcessing::makeSignal(totalADC, PadSecPos(CRU(cru).sector(), PadPos(row, pad)), pedestal, noise);
  if(mADC > 0) {
    using P = std::pair<MCCompLabel, int>;
#ifndef EXTLABELS
    // Sort the MC labels according to their occurrence
    std::sort(mMClabel.begin(), mMClabel.end(), [](const P& a, const P& b) {return a.second > b.second;});
#else
    static std::vector<std::pair<o2::MCCompLabel, int>> labelvec;
    // involves some copying but does not need repeated allocations for labelvec
    sLabels.fillVectorOfLabels(mId, labelvec);
    std::sort(labelvec.begin(), labelvec.end(), [](const P& a, const P& b) {return a.second > b.second;});
#endif
    
    /// Write out the Digit
    TClonesArray &clref = *output;
    const size_t digiPos = clref.GetEntriesFast();
    new(clref[digiPos]) Digit(cru, mADC, row, pad, timeBin); /// create Digit
#ifndef EXTLABELS
    for(auto &mcLabel : mMClabel) {
//      std::cerr << "putting " << mcLabel.first << " in size " << labelvec.size() << "\n";
      mcTruth.addElement(digiPos, mcLabel.first); /// add MCTruth output
    }
#else
    for(auto &mcLabel : labelvec) {
    //      std::cerr << "putting " << mcLabel.first << " in size " << labelvec.size() << "\n";
      mcTruth.addElement(digiPos, mcLabel.first); /// add MCTruth output
    }
#endif

    if(debug!=nullptr) {
      TClonesArray &clrefDebug = *debug;
      const size_t digiPosDebug = clrefDebug.GetEntriesFast();
      new(clrefDebug[digiPosDebug]) DigitMCMetaData(mChargePad, commonMode, pedestal, noise); /// create DigitMCMetaData
    }
  }
}
