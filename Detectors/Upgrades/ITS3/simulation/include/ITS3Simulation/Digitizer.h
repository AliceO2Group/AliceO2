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

/// \file Digitizer.h
/// \brief Definition of the ITS digitizer
#ifndef ALICEO2_ITS3_DIGITIZER_H
#define ALICEO2_ITS3_DIGITIZER_H

#include <vector>
#include <deque>
#include <memory>

#include "Rtypes.h"
#include "TObject.h"

#include "ITSMFTSimulation/AlpideSimResponse.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SegmentationMosaix.h"
#include "ITS3Simulation/DigiParams.h"
#include "ITS3Simulation/ChipDigitsContainer.h"
#include "ITS3Simulation/ChipSimResponse.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2::its3
{

class Digitizer : public TObject
{
  using ExtraDig = std::vector<itsmft::PreDigitLabelRef>; ///< container for extra contributions to PreDigits

 public:
  ~Digitizer() = default;

  void setDigits(std::vector<o2::itsmft::Digit>* dig) { mDigits = dig; }
  void setMCLabels(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mclb) { mMCLabels = mclb; }
  void setROFRecords(std::vector<o2::itsmft::ROFRecord>* rec) { mROFRecords = rec; }

  o2::its3::DigiParams& getParams() { return mParams; }
  const o2::its3::DigiParams& getParams() const { return mParams; }

  void init();

  /// Steer conversion of hits to digits
  void process(const std::vector<itsmft::Hit>* hits, int evID, int srcID, int layer);
  void setEventTime(const o2::InteractionTimeRecord& irt, int layer);
  double getEndTimeOfROFMax() const
  {
    ///< return the time corresponding to end of the last reserved ROFrame : mROFrameMax
    return mParams.getROFrameLength() * (mROFrameMax + 1) + mParams.getTimeOffset();
  }

  void setContinuous(bool v) { mParams.setContinuous(v); }
  bool isContinuous() const { return mParams.isContinuous(); }
  void fillOutputContainer(uint32_t maxFrame = 0xffffffff, int layer = -1);

  // provide the common itsmft::GeometryTGeo to access matrices and segmentation
  void setGeometry(const o2::its::GeometryTGeo* gm) { mGeometry = gm; }

  uint32_t getEventROFrameMin() const { return mEventROFrameMin; }
  uint32_t getEventROFrameMax() const { return mEventROFrameMax; }
  void resetEventROFrames()
  {
    mEventROFrameMin = 0xffffffff;
    mEventROFrameMax = 0;
  }
  void resetROFrameBounds()
  {
    mROFrameMin = 0;
    mROFrameMax = 0;
    mNewROFrame = 0;
  }

  void setDeadChannelsMap(const o2::itsmft::NoiseMap* mp) { mDeadChanMap = mp; }

 private:
  void processHit(const o2::itsmft::Hit& hit, uint32_t& maxFr, int evID, int srcID, int layer);
  void registerDigits(o2::its3::ChipDigitsContainer& chip, uint32_t roFrame, float tInROF, int nROF,
                      uint16_t row, uint16_t col, int nEle, o2::MCCompLabel& lbl, int layer);

  ExtraDig* getExtraDigBuffer(uint32_t roFrame)
  {
    if (mROFrameMin > roFrame) {
      return nullptr; // nothing to do
    }
    int ind = roFrame - mROFrameMin;
    while (ind >= int(mExtraBuff.size())) {
      mExtraBuff.emplace_back(std::make_unique<ExtraDig>());
    }
    return mExtraBuff[ind].get();
  }

  static constexpr float sec2ns = 1e9;

  o2::its3::DigiParams mParams;            ///< digitization parameters
  o2::InteractionTimeRecord mEventTime;    ///< global event time and interaction record
  o2::InteractionRecord mIRFirstSampledTF; ///< IR of the 1st sampled IR, noise-only ROFs will be inserted till this IR only
  double mCollisionTimeWrtROF{};
  uint32_t mROFrameMin = 0; ///< lowest RO frame of current digits
  uint32_t mROFrameMax = 0; ///< highest RO frame of current digits
  uint32_t mNewROFrame = 0; ///< ROFrame corresponding to provided time
  bool mIsBeforeFirstRO = false;

  uint32_t mEventROFrameMin = 0xffffffff; ///< lowest RO frame for processed events (w/o automatic noise ROFs)
  uint32_t mEventROFrameMax = 0;          ///< highest RO frame forfor processed events (w/o automatic noise ROFs)

  static constexpr std::array<o2::its3::SegmentationMosaix, 3> mIBSegmentations{0, 1, 2};

  const o2::its3::ChipSimResponse* mSimRespIB = nullptr;     // simulated response for IB
  const o2::itsmft::AlpideSimResponse* mSimRespOB = nullptr; // simulated response for OB
  bool mSimRespIBOrientation{false};                         // wether the orientation in the IB response function is flipped
  float mSimRespIBShift{0.f};                                // adjusting the Y-shift in the IB response function to match sensor local coord.
  float mSimRespIBScaleX{1.f};                               // scale x-local coordinate to response function x-coordinate
  float mSimRespIBScaleZ{1.f};                               // scale z-local coordinate to response function z-coordinate
  float mSimRespOBShift{0.f};                                // adjusting the Y-shift in the OB response function to match sensor local coord.

  const o2::its::GeometryTGeo* mGeometry = nullptr; ///< ITS3 geometry

  std::vector<o2::its3::ChipDigitsContainer> mChips; ///< Array of chips digits containers
  std::deque<std::unique_ptr<ExtraDig>> mExtraBuff;  ///< burrer (per roFrame) for extra digits

  std::vector<o2::itsmft::Digit>* mDigits = nullptr;                       //! output digits
  std::vector<o2::itsmft::ROFRecord>* mROFRecords = nullptr;               //! output ROF records
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels

  const o2::itsmft::NoiseMap* mDeadChanMap = nullptr;

  ClassDef(Digitizer, 5);
};

} // namespace o2::its3

#endif /* ALICEO2_ITS3_DIGITIZER_H */
