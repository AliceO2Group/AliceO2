// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitGlobalPad.cxx
/// \brief Implementation of the Pad container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitGlobalPad.h"
#include "TPCBase/Mapper.h"
#include "TPCSimulation/SAMPAProcessing.h"

#include <boost/bind.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include <vector>

using namespace o2::TPC;

void DigitGlobalPad::fillOutputContainer(std::vector<Digit>* output,
                                         dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                                         std::vector<DigitMCMetaData>* debug, const CRU& cru, TimeBin timeBin,
                                         GlobalPadNumber globalPad, float commonMode)
{
  const static Mapper& mapper = Mapper::instance();
  const PadPos pad = mapper.padPos(globalPad);

  /// The charge accumulated on that pad is converted into ADC counts, saturation of the SAMPA is applied and a Digit is
  /// created in written out
  const float totalADC = mChargePad - commonMode; // common mode is subtracted here in order to properly apply noise,
                                                  // pedestals and saturation of the SAMPA

  float noise, pedestal;
  const float mADC = SAMPAProcessing::makeSignal(totalADC, PadSecPos(cru.sector(), pad), pedestal, noise);

  /// only write out the data if there is actually charge on that pad
  if (mADC > 0 && mChargePad > 0) {

    /// Sort the MC labels according to their occurrence
    using P = std::pair<MCCompLabel, int>;
    std::sort(mMClabel.begin(), mMClabel.end(), [](const P& a, const P& b) { return a.second > b.second; });

    /// Write out the Digit
    const auto digiPos = output->size();
    output->emplace_back(cru, mADC, pad.getRow(), pad.getPad(), timeBin); /// create Digit and append to container

    for (auto& mcLabel : mMClabel) {
      mcTruth.addElement(digiPos, mcLabel.first); /// add MCTruth output
    }

    if (debug != nullptr) {
      debug->emplace_back(mChargePad, commonMode, pedestal, noise); /// create DigitMCMetaData
    }
  }
}
