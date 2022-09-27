// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include <fmt/format.h>

#include "TTree.h" // for TTree destructor

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "CommonUtils/DebugStreamer.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Mapper.h"
#include "TPCSimulation/SAMPAProcessing.h"

namespace o2::tpc
{

/// \struct PrevDigitInfo
struct PrevDigitInfo {
  float signal{}; ///< pad signal
  float cumul{};  ///< cumulative signal of prev time bins
  int tot{};      ///< remaining time over threshold

  bool hasSignal() const { return signal != 0 || cumul != 0 || tot != 0; }

  ClassDefNV(PrevDigitInfo, 1);
};

/// \class DigitGlobalPad
/// This is the lowest class of the intermediate Digit Containers, in which all incoming electrons from the
/// hits are sorted into after amplification
/// The structure assures proper sorting of the Digits and proper propagation of the MC labels when later on
/// written out for further processing.

class DigitGlobalPad
{
 public:
  using Streamer = o2::utils::DebugStreamer;

  /// Constructor
  DigitGlobalPad() = default;

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
  void addDigit(const MCCompLabel& label, float signal,
                o2::dataformats::LabelContainer<std::pair<MCCompLabel, int>, false>&);

  /// Fold signal with previous pad signal add ion tail and ToT for sigmal saturation
  void foldSignal(PrevDigitInfo& prevDigit, const int sector, const int pad, const TimeBin time, Streamer* debugStream = nullptr);

  void setID(int id) { mID = id; }
  int getID() const { return mID; }

  /// Fill output vector
  /// \param output Output container
  /// \param mcTruth MC Truth container
  /// \param cru CRU ID
  /// \param timeBin Time bin
  /// \param globalPad Global pad ID
  /// \param commonMode Common mode value of that specific ROC
  template <DigitzationMode MODE>
  void fillOutputContainer(std::vector<Digit>& output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                           const CRU& cru, TimeBin timeBin,
                           GlobalPadNumber globalPad,
                           o2::dataformats::LabelContainer<std::pair<MCCompLabel, int>, false>& labelContainer,
                           float commonMode = 0.f, const PrevDigitInfo& prevDigit = PrevDigitInfo(), Streamer* debugStream = nullptr);

 private:
  /// Compare two MC labels regarding trackID, eventID and sourceID
  /// \param label1 MC label 1
  /// \param label2 MC label 2
  /// \return true, if trackID, eventID and sourceID are the same
  bool compareMClabels(const MCCompLabel& label1, const MCCompLabel& label2) const;

  float mChargePad = 0.; ///< Total accumulated charge on that GlobalPad for a given time bin
  int mID = -1;          ///< ID of this digit to refer into labels (-1 means not initialized)
};

inline void DigitGlobalPad::addDigit(const MCCompLabel& label, float signal,
                                     o2::dataformats::LabelContainer<std::pair<MCCompLabel, int>, false>& labels)
{
  bool isKnown = false;
  auto view = labels.getLabels(mID);
  for (auto& mcLabel : view) {
    if (compareMClabels(label, mcLabel.first)) {
      ++mcLabel.second;
      isKnown = true;
      break;
    }
  }

  //
  if (!isKnown) {
    std::pair<MCCompLabel, int> newlabel(label, 1);
    labels.addLabel(mID, newlabel);
  }
  mChargePad += signal;
}

inline void DigitGlobalPad::foldSignal(PrevDigitInfo& prevDigit, const int sector, const int pad, const TimeBin time, Streamer* debugStream)
{
  const auto& eleParam = ParameterElectronics::Instance();
  const float kAmp = 0.1276;  // TODO: replace with pad-by-pad value
  const float kTime = 0.0515; // TODO: replace with pad-by-pad value
  const float tailSlopeUnit = std::exp(-kTime);
  const PrevDigitInfo prevDigInf = prevDigit;

  // Saturation tail simulation
  if (eleParam.doSaturationTail) {
    if (prevDigit.signal > 1023.f) {
      prevDigit.tot += int(std::round(mChargePad * eleParam.adcToT));
    }
    if (prevDigit.tot > 0) {
      prevDigit.tot -= 1;
    }
  }

  // ion tail simulation
  if (eleParam.doIonTail) {
    mChargePad = mChargePad + kAmp * (1 - tailSlopeUnit) * prevDigit.cumul;
    prevDigit.cumul += prevDigInf.signal;
    prevDigit.cumul *= tailSlopeUnit;
    if (prevDigit.cumul < 0.1) {
      prevDigit.cumul = 0;
    }
  }
  prevDigit.signal = mChargePad;

  if (debugStream) {
    debugStream->setStreamer("debug_digits", "UPDATE");
    float kAmpTmp = kAmp;
    float kTimeTmp = kTime;
    float tailSlopeUnitTmp = tailSlopeUnit;
    PrevDigitInfo prevDigInfTmp = prevDigInf;

    debugStream->getStreamer() << debugStream->getUniqueTreeName("foldSignal").data()
                               << "kAmp=" << kAmpTmp
                               << "kTime=" << kTimeTmp
                               << "tailSlopeUnit=" << tailSlopeUnitTmp
                               << "prevDig=" << prevDigInfTmp
                               << "dig=" << prevDigit
                               << "\n";
  }

  // TODO: propagate labels for ion tail?
}

inline void DigitGlobalPad::reset()
{
  mChargePad = 0;
}

inline bool DigitGlobalPad::compareMClabels(const MCCompLabel& label1, const MCCompLabel& label2) const
{
  // we compare directly on the bare label (in which eventID, labelID etc. are encoded)
  // this avoids any logical operator on the label; optimization motivated from an Intel VTune analysis
  // (note that this is also faster than using the operator= of MCCompLabel)
  return label1.getRawValue() == label2.getRawValue();
}

template <DigitzationMode MODE>
inline void DigitGlobalPad::fillOutputContainer(std::vector<Digit>& output,
                                                dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                                                const CRU& cru, TimeBin timeBin,
                                                GlobalPadNumber globalPad,
                                                o2::dataformats::LabelContainer<std::pair<MCCompLabel, int>, false>& labels,
                                                float commonMode, const PrevDigitInfo& prevDigit, Streamer* debugStream)
{
  const Mapper& mapper = Mapper::instance();
  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  const PadPos pad = mapper.padPos(globalPad);
  static std::vector<std::pair<MCCompLabel, int>> labelCollector; // static workspace container for sorting

  /// The charge accumulated on that pad is converted into ADC counts, saturation of the SAMPA is applied and a Digit
  /// is created in written out

  float noise, pedestal;
  const float mADC = sampaProcessing.makeSignal<MODE>(mChargePad, cru.sector(), globalPad, commonMode, pedestal, noise, prevDigit.tot);

  if (debugStream) {
    int sector = cru.sector();
    float adc = mADC;
    PrevDigitInfo prevDigitTmp = prevDigit;
    debugStream->getStreamer() << debugStream->getUniqueTreeName("digit").data()
                               << "sector=" << sector
                               << "pad=" << globalPad
                               << "charge=" << mChargePad
                               << "adc=" << adc
                               << "prevDig=" << prevDigitTmp
                               << "cm=" << commonMode
                               << "\n";
  }

  /// only write out the data if there is actually charge on that pad
  if (mADC > 0) {
    auto labelview = labels.getLabels(mID);

    /// Write out the Digit
    const auto digiPos = output.size();
    output.emplace_back(cru, mADC, pad.getRow(), pad.getPad(), timeBin); /// create Digit and append to container

    labelCollector.clear();
    for (auto& mcLabel : labelview) {
      labelCollector.push_back(mcLabel);
    }
    if (labelview.size() > 1) {
      /// Sort the MC labels according to their occurrence
      using P = std::pair<MCCompLabel, int>;
      std::sort(labelCollector.begin(), labelCollector.end(), [](const P& a, const P& b) { return a.second > b.second; });
    }
    for (auto& mcLabel : labelCollector) {
      mcTruth.addElement(digiPos, mcLabel.first); /// add MCTruth output
    }
  }
}
} // namespace o2::tpc

#endif // ALICEO2_TPC_DigitGlobalPad_H_
