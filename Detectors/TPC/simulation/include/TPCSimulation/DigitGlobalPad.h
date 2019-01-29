// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitGlobalPad.h
/// \brief Definition of the GlobalPad container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitGlobalPad_H_
#define ALICEO2_TPC_DigitGlobalPad_H_

#include <map>
#include <vector>

#include "TTree.h" // for TTree destructor

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Digit.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCSimulation/DigitMCMetaData.h"
#include "TPCSimulation/SAMPAProcessing.h"

namespace o2
{
namespace TPC
{

/// \class DigitGlobalPad
/// This is the fifth and lowest class of the intermediate Digit Containers, in which all incoming electrons from the
/// hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the individual GlobalPad containers and is contained within the Row Container.

class DigitGlobalPad
{
 public:
  /// Constructor
  DigitGlobalPad();

  /// Destructor
  ~DigitGlobalPad() = default;

  /// Resets the container
  void reset();

  /// Get the accumulated charge on that GlobalPad
  /// \return Accumulated charge
  float getChargePad() const { return mChargePad; }

  /// Add digit to the time bin container
  /// \param eventID MC Event ID
  /// \param trackID MC Track ID
  /// \param signal Charge of the digit in ADC counts
  void addDigit(const MCCompLabel& label, float signal);

  /// Fill output vector
  /// \param output Output container
  /// \param mcTruth MC Truth container
  /// \param debug Optional debug output container
  /// \param cru CRU ID
  /// \param timeBin Time bin
  /// \param row Row ID
  /// \param pad Pad ID
  /// \param commonMode Common mode value of that specific ROC
  template <DigitzationMode MODE>
  void fillOutputContainer(std::vector<Digit>* output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                           std::vector<DigitMCMetaData>* debug, const CRU& cru, TimeBin timeBin,
                           GlobalPadNumber globalPad, float commonMode = 0.f);

 private:
  /// Compare two MC labels regarding trackID, eventID and sourceID
  /// \param label1 MC label 1
  /// \param label2 MC label 2
  /// \return true, if trackID, eventID and sourceID are the same
  bool compareMClabels(const MCCompLabel& label1, const MCCompLabel& label2) const;

  float mChargePad;                                  ///< Total accumulated charge on that GlobalPad for a given time bin
  std::vector<std::pair<MCCompLabel, int>> mMClabel; ///< vector to accumulate the MC labels
};

inline DigitGlobalPad::DigitGlobalPad() : mChargePad(0.), mMClabel() {}

inline void DigitGlobalPad::addDigit(const MCCompLabel& label, float signal)
{
  bool isKnown = false;
  // MCCompLabel tempLabel(trackID, eventID);
  for (auto& mcLabel : mMClabel) {
    if (compareMClabels(label, mcLabel.first)) {
      ++mcLabel.second;
      isKnown = true;
    }
  }
  if (!isKnown) {
    mMClabel.emplace_back(label, 1);
  }
  mChargePad += signal;
}

inline void DigitGlobalPad::reset()
{
  mChargePad = 0;
  mMClabel.clear();
}

inline bool DigitGlobalPad::compareMClabels(const MCCompLabel& label1, const MCCompLabel& label2) const
{
  return (label1.getEventID() == label2.getEventID() && label1.getTrackID() == label2.getTrackID() &&
          label1.getSourceID() == label2.getSourceID());
}

template <DigitzationMode MODE>
inline void DigitGlobalPad::fillOutputContainer(std::vector<Digit>* output,
                                                dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                                                std::vector<DigitMCMetaData>* debug, const CRU& cru, TimeBin timeBin,
                                                GlobalPadNumber globalPad, float commonMode)
{
  const static Mapper& mapper = Mapper::instance();
  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  const PadPos pad = mapper.padPos(globalPad);

  /// The charge accumulated on that pad is converted into ADC counts, saturation of the SAMPA is applied and a Digit
  /// is created in written out
  const float totalADC = mChargePad - commonMode; // common mode is subtracted here in order to properly apply noise,
                                                  // pedestals and saturation of the SAMPA

  float noise, pedestal;
  const float mADC = sampaProcessing.makeSignal<MODE>(totalADC, cru.sector(), globalPad, pedestal, noise);

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
}
}

#endif // ALICEO2_TPC_DigitGlobalPad_H_
