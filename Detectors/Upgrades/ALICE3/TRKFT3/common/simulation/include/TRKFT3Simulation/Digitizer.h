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
/// \brief Definition of the TRK/FT3 digitizer
#ifndef ALICEO2_TRKFT3_DIGITIZER_H
#define ALICEO2_TRKFT3_DIGITIZER_H

#include <vector>
#include <deque>
#include <memory>
#include <type_traits>

#include "Rtypes.h"  // for Digitizer::Class
#include "TObject.h" // for TObject

#include "TRKFT3Simulation/ChipSimResponse.h"
#include "TRKFT3Simulation/ChipDigitsContainer.h"

#include "TRKFT3Simulation/DigiParams.h"
#include "DataFormatsTRKFT3/Hit.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "FT3Base/GeometryTGeo.h"
#include "TRKBase/GeometryTGeo.h"
#include "DataFormatsTRKFT3/Digit.h"
#include "DataFormatsTRKFT3/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2::trkft3
{

template <int DetID>
class Digitizer
{
  static_assert(DetID == o2::detectors::DetID::TRK || DetID == o2::detectors::DetID::FT3, "only TRK and FT3 digitizers are supported");
  using GeometryTGeo = std::conditional_t<DetID == o2::detectors::DetID::TRK, o2::trk::GeometryTGeo, o2::ft3::GeometryTGeo>;
  using ExtraDig = std::vector<itsmft::PreDigitLabelRef>; ///< container for extra contributions to PreDigits

 public:
  void setDigits(std::vector<o2::trkft3::Digit>* dig) { mDigits = dig; }
  void setMCLabels(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mclb) { mMCLabels = mclb; }
  void setROFRecords(std::vector<o2::trkft3::ROFRecord>* rec) { mROFRecords = rec; }
  void setResponseName(const std::string& name) { mRespName = name; }

  o2::trkft3::DigiParams<DetID>& getParams() { return mParams; }
  const o2::trkft3::DigiParams<DetID>& getParams() const { return mParams; }

  void init();

  const o2::trkft3::ChipSimResponse* getChipResponse(int chipID);

  /// Steer conversion of hits to digits
  void process(const std::vector<o2::trkft3::Hit>* hits, int evID, int srcID, int layer);
  void setEventTime(const o2::InteractionTimeRecord& irt, int layer);

  void fillOutputContainer(uint32_t maxFrame, int layer);

  void resetROFrameBounds()
  {
    mROFrameMin = 0;
    mROFrameMax = 0;
    mNewROFrame = 0;
    mROFsWrtFirstRO = 0;
    mExtraBuff.clear();
  }

  const o2::trkft3::DigiParams<DetID>& getDigitParams() const { return mParams; }

  void setGeometry(const GeometryTGeo* gm)
  {
    LOG(info) << "trkft3::Digitizer set geom";
    mGeometry = gm;
  }

  uint32_t getEventROFrameMin() const { return mEventROFrameMin; }
  uint32_t getEventROFrameMax() const { return mEventROFrameMax; }
  void resetEventROFrames()
  {
    mEventROFrameMin = 0xffffffff;
    mEventROFrameMax = 0;
  }

  void setDeadChannelsMap(const o2::itsmft::NoiseMap* mp) { mDeadChanMap = mp; }

 private:
  void processHit(const o2::trkft3::Hit& hit, uint32_t& maxFr, int evID, int srcID, int rofLayer);
  void registerDigits(o2::trkft3::ChipDigitsContainer& chip, uint32_t roFrame, float tInROF, int nROF,
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

  /// Get the number of columns according to the subdetector
  /// \param subDetID 0 for VD, 1 for ML/OT
  /// \param layer 0 to 2 for VD, 0 to 7 for ML/OT
  /// \return Number of columns (In the entire layer(VD) or chip (ML/OT)
  int getNCols(int subDetID, int layer)
  {
    if (subDetID == 0) { // VD
      return o2::trk::constants::VD::petal::layer::nCols;
    } else if (subDetID == 1 || subDetID == 2) { // ML/OT: the smallest element is a chip of 470 rows and 640 cols
      return o2::trk::constants::moduleMLOT::chip::nCols;
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
      return o2::trk::constants::VD::petal::layer::nRows[layer];
    } else if (subDetID == 1 || subDetID == 2) { // ML/OT
      return o2::trk::constants::moduleMLOT::chip::nRows;
    }
    return 0;
  }

  int getROFLayer(int chipID) const
  {
    if constexpr (DetID == o2::detectors::DetID::TRK) {
      return mGeometry->getLayerTRK(chipID);
    } else {
      return mGeometry->getLayer(chipID);
    }
  }

  int getDisk(int chipID) const
  {
    if constexpr (DetID == o2::detectors::DetID::TRK) {
      return mGeometry->getDisk(chipID);
    } else {
      return -1;
    }
  }

  static constexpr float sec2ns = 1e9;

  o2::trkft3::DigiParams<DetID> mParams;   ///< digitization parameters
  o2::InteractionTimeRecord mEventTime;    ///< global event time and interaction record
  o2::InteractionRecord mIRFirstSampledTF; ///< IR of the 1st sampled IR, noise-only ROFs will be inserted till this IR only
  double mCollisionTimeWrtROF{};
  uint32_t mROFrameMin = 0; ///< lowest RO frame of current digits
  uint32_t mROFrameMax = 0; ///< highest RO frame of current digits
  uint32_t mNewROFrame = 0; ///< ROFrame corresponding to provided time

  int mROFsWrtFirstRO = 0;

  uint32_t mEventROFrameMin = 0xffffffff; ///< lowest RO frame for processed events (w/o automatic noise ROFs)
  uint32_t mEventROFrameMax = 0;          ///< highest RO frame forfor processed events (w/o automatic noise ROFs)

  int mNumberOfChips = 0;

  const o2::trkft3::ChipSimResponse* mChipSimResp = nullptr;     // simulated response
  const o2::trkft3::ChipSimResponse* mChipSimRespVD = nullptr;   // simulated response for VD chips
  const o2::trkft3::ChipSimResponse* mChipSimRespMLOT = nullptr; // simulated response for ML/OT chips

  std::string mRespName; /// APTS or ALICE3, depending on the response to be used

  bool mSimRespOrientation{false};   // wether the orientation in the response function is flipped
  float mSimRespVDShift{0.f};        // adjusting the Y-shift in the APTS response function to match sensor local coord.
  float mSimRespVDScaleX{1.f};       // scale x-local coordinate to response function x-coordinate
  float mSimRespVDScaleZ{1.f};       // scale z-local coordinate to response function z-coordinate
  float mSimRespMLOTShift{0.f};      // adjusting the Y-shift in the APTS response function to match sensor local coord.
  float mSimRespMLOTScaleX{1.f};     // scale x-local coordinate to response function x-coordinate
  float mSimRespMLOTScaleZ{1.f};     // scale z-local coordinate to response function z-coordinate
  float mSimRespVDScaleDepth{1.f};   // scale depth-local coordinate to response function depth-coordinate
  float mSimRespMLOTScaleDepth{1.f}; // scale depth-local coordinate to response function depth-coordinate

  const GeometryTGeo* mGeometry = nullptr; ///< TRK or FT3 geometry

  std::vector<o2::trkft3::ChipDigitsContainer> mChips; ///< Array of chips digits containers
  std::deque<std::unique_ptr<ExtraDig>> mExtraBuff;    ///< buffer (per roFrame) for extra digits

  std::vector<o2::trkft3::Digit>* mDigits = nullptr;                       //! output digits
  std::vector<o2::trkft3::ROFRecord>* mROFRecords = nullptr;               //! output ROF records
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels

  const o2::itsmft::NoiseMap* mDeadChanMap = nullptr;
  const o2::itsmft::NoiseMap* mNoiseMap = nullptr;
};
} // namespace o2::trkft3

extern template class o2::trkft3::Digitizer<o2::detectors::DetID::TRK>;
extern template class o2::trkft3::Digitizer<o2::detectors::DetID::FT3>;

#endif
