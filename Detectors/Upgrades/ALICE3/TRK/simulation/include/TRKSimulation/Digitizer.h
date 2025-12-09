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
/// \brief Definition of the TRK digitizer
#ifndef ALICEO2_TRK_DIGITIZER_H
#define ALICEO2_TRK_DIGITIZER_H

#include <vector>
#include <deque>
#include <memory>

#include "Rtypes.h"  // for Digitizer::Class
#include "TObject.h" // for TObject

#include "TRKSimulation/ChipSimResponse.h"
#include "TRKSimulation/ChipDigitsContainer.h"

#include "TRKSimulation/DigiParams.h"
#include "TRKSimulation/Hit.h"
#include "TRKBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif

namespace o2::trk
{

class Digitizer
{
  using ExtraDig = std::vector<itsmft::PreDigitLabelRef>; ///< container for extra contributions to PreDigits

 public:
  void setDigits(std::vector<o2::itsmft::Digit>* dig) { mDigits = dig; }
  void setMCLabels(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mclb) { mMCLabels = mclb; }
  void setROFRecords(std::vector<o2::itsmft::ROFRecord>* rec) { mROFRecords = rec; }

  o2::trk::DigiParams& getParams() { return (o2::trk::DigiParams&)mParams; }
  const o2::trk::DigiParams& getParams() const { return mParams; }

  void init();

  const o2::trk::ChipSimResponse* getChipResponse(int chipID);

  /// Steer conversion of hits to digits
  void process(const std::vector<o2::trk::Hit>* hits, int evID, int srcID);
  void setEventTime(const o2::InteractionTimeRecord& irt);
  double getEndTimeOfROFMax() const
  {
    ///< return the time corresponding to end of the last reserved ROFrame : mROFrameMax
    return mParams.getROFrameLength() * (mROFrameMax + 1) + mParams.getTimeOffset();
  }

  void setContinuous(bool v) { mParams.setContinuous(v); }
  bool isContinuous() const { return mParams.isContinuous(); }
  void fillOutputContainer(uint32_t maxFrame = 0xffffffff);

  const o2::trk::DigiParams& getDigitParams() const { return mParams; }

  // provide the common trk::GeometryTGeo to access matrices and segmentation
  void setGeometry(const o2::trk::GeometryTGeo* gm) { mGeometry = gm; }

  uint32_t getEventROFrameMin() const { return mEventROFrameMin; }
  uint32_t getEventROFrameMax() const { return mEventROFrameMax; }
  void resetEventROFrames()
  {
    mEventROFrameMin = 0xffffffff;
    mEventROFrameMax = 0;
  }

  void setDeadChannelsMap(const o2::itsmft::NoiseMap* mp) { mDeadChanMap = mp; }

 private:
  void processHit(const o2::trk::Hit& hit, uint32_t& maxFr, int evID, int srcID);
  void registerDigits(o2::trk::ChipDigitsContainer& chip, uint32_t roFrame, float tInROF, int nROF,
                      uint16_t row, uint16_t col, int nEle, o2::MCCompLabel& lbl);

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

  /// Get the number of columns according to the subdetector
  /// \param subDetID 0 for VD, 1 for ML/OT
  /// \param layer 0 to 2 for VD, 0 to 7 for ML/OT
  /// \return Number of columns (In the entire layer(VD) or chip (ML/OT)
  int getNCols(int subDetID, int layer)
  {
    if (subDetID == 0) { // VD
      return constants::VD::petal::layer::nCols;
    } else if (subDetID == 1) { // ML/OT: the smallest element is a chip of 470 rows and 640 cols
      return constants::moduleMLOT::chip::nCols;
    }
    return 0;
  }

  /// Get the number of rows according to the subdetector
  /// \param subDetID 0 for VD, 1 for ML/OT
  /// \param layer 0 to 2 for VD, 0 to 7 for ML/OT
  /// \return Number of rows (In the entire layer(VD) or chip (ML/OT)
  int getNRows(int subDetID, int layer)
  {
    if (subDetID == 0) { // VD
      return constants::VD::petal::layer::nRows[layer];
    } else if (subDetID == 1) { // ML/OT
      return constants::moduleMLOT::chip::nRows;
    }
    return 0;
  }

  static constexpr float sec2ns = 1e9;

  o2::trk::DigiParams mParams;             ///< digitization parameters
  o2::InteractionTimeRecord mEventTime;    ///< global event time and interaction record
  o2::InteractionRecord mIRFirstSampledTF; ///< IR of the 1st sampled IR, noise-only ROFs will be inserted till this IR only
  double mCollisionTimeWrtROF{};
  uint32_t mROFrameMin = 0; ///< lowest RO frame of current digits
  uint32_t mROFrameMax = 0; ///< highest RO frame of current digits
  uint32_t mNewROFrame = 0; ///< ROFrame corresponding to provided time

  uint32_t mEventROFrameMin = 0xffffffff; ///< lowest RO frame for processed events (w/o automatic noise ROFs)
  uint32_t mEventROFrameMax = 0;          ///< highest RO frame forfor processed events (w/o automatic noise ROFs)

  int mNumberOfChips = 0;

  const o2::trk::ChipSimResponse* mChipSimResp = nullptr;     // simulated response
  const o2::trk::ChipSimResponse* mChipSimRespVD = nullptr;   // simulated response for VD chips
  const o2::trk::ChipSimResponse* mChipSimRespMLOT = nullptr; // simulated response for ML/OT chips

  bool mSimRespOrientation{false};   // wether the orientation in the response function is flipped
  float mSimRespVDShift{0.f};        // adjusting the Y-shift in the APTS response function to match sensor local coord.
  float mSimRespVDScaleX{1.f};       // scale x-local coordinate to response function x-coordinate
  float mSimRespVDScaleZ{1.f};       // scale z-local coordinate to response function z-coordinate
  float mSimRespMLOTShift{0.f};      // adjusting the Y-shift in the APTS response function to match sensor local coord.
  float mSimRespMLOTScaleX{1.f};     // scale x-local coordinate to response function x-coordinate
  float mSimRespMLOTScaleZ{1.f};     // scale z-local coordinate to response function z-coordinate
  float mSimRespVDScaleDepth{1.f};   // scale depth-local coordinate to response function depth-coordinate
  float mSimRespMLOTScaleDepth{1.f}; // scale depth-local coordinate to response function depth-coordinate

  const o2::trk::GeometryTGeo* mGeometry = nullptr; ///< TRK geometry

  std::vector<o2::trk::ChipDigitsContainer> mChips; ///< Array of chips digits containers
  std::deque<std::unique_ptr<ExtraDig>> mExtraBuff; ///< buffer (per roFrame) for extra digits

  std::vector<o2::itsmft::Digit>* mDigits = nullptr;                       //! output digits
  std::vector<o2::itsmft::ROFRecord>* mROFRecords = nullptr;               //! output ROF records
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels

  const o2::itsmft::NoiseMap* mDeadChanMap = nullptr;
  const o2::itsmft::NoiseMap* mNoiseMap = nullptr;
};
} // namespace o2::trk
