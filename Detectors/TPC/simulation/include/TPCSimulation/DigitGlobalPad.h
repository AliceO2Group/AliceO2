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

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Digit.h"
#include "TPCSimulation/DigitMCMetaData.h"
#include "TTree.h" // for TTree destructor

#include <map>

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
  void addDigit(size_t eventID, size_t trackID, float signal);

  /// Fill output vector
  /// \param output Output container
  /// \param mcTruth MC Truth container
  /// \param debug Optional debug output container
  /// \param cru CRU ID
  /// \param timeBin Time bin
  /// \param row Row ID
  /// \param pad Pad ID
  /// \param commonMode Common mode value of that specific ROC
  void fillOutputContainer(std::vector<Digit>* output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                           std::vector<DigitMCMetaData>* debug, const CRU& cru, TimeBin timeBin,
                           GlobalPadNumber globalPad, float commonMode = 0.f);

 private:
  /// Compare two MC labels regarding trackID, eventID and sourceID
  /// \param label1 MC label 1
  /// \param label2 MC label 2
  /// \return true, if trackID, eventID and sourceID are the same
  bool compareMClabels(const MCCompLabel& label1, const MCCompLabel& label2) const;

  float mChargePad; ///< Total accumulated charge on that GlobalPad for a given time bin
  std::vector<std::pair<MCCompLabel, int>> mMClabel; ///< vector to accumulate the MC labels
};

inline DigitGlobalPad::DigitGlobalPad() : mChargePad(0.), mMClabel() {}

inline void DigitGlobalPad::addDigit(size_t eventID, size_t trackID, float signal)
{
  bool isKnown = false;
  MCCompLabel tempLabel(trackID, eventID);
  for (auto& mcLabel : mMClabel) {
    if (compareMClabels(tempLabel, mcLabel.first)) {
      ++mcLabel.second;
      isKnown = true;
    }
  }
  if (!isKnown) {
    mMClabel.emplace_back(tempLabel, 1);
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
}
}

#endif // ALICEO2_TPC_DigitGlobalPad_H_
