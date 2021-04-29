// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Rtypes.h"  // for Digitizer::Class
#include "TObject.h" // for TObject

#include "ITSMFTSimulation/ChipDigitsContainer.h"
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include "ITSMFTSimulation/DigiParams.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITS3Base/GeometryTGeo.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace its3
{
class Digitizer : public TObject
{
  using ExtraDig = std::vector<itsmft::PreDigitLabelRef>; ///< container for extra contributions to PreDigits

 public:
  Digitizer() = default;
  ~Digitizer() override = default;
  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  void setDigits(std::vector<o2::itsmft::Digit>* dig) { mDigits = dig; }
  void setMCLabels(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mclb) { mMCLabels = mclb; }
  void setROFRecords(std::vector<o2::itsmft::ROFRecord>* rec) { mROFRecords = rec; }

  o2::itsmft::DigiParams& getParams() { return (o2::itsmft::DigiParams&)mParams; }
  const o2::itsmft::DigiParams& getParams() const { return mParams; }

  void init();

  /// Steer conversion of hits to digits
  void process(const std::vector<itsmft::Hit>* hits, int evID, int srcID);
  void setEventTime(const o2::InteractionTimeRecord& irt);
  double getEndTimeOfROFMax() const
  {
    ///< return the time corresponding to end of the last reserved ROFrame : mROFrameMax
    return mParams.getROFrameLength() * (mROFrameMax + 1) + mParams.getTimeOffset();
  }

  void setContinuous(bool v) { mParams.setContinuous(v); }
  bool isContinuous() const { return mParams.isContinuous(); }
  void fillOutputContainer(uint32_t maxFrame = 0xffffffff);

  void setDigiParams(const o2::itsmft::DigiParams& par) { mParams = par; }
  const o2::itsmft::DigiParams& getDigitParams() const { return mParams; }

  // provide the common itsmft::GeometryTGeo to access matrices and segmentation
  void setGeometry(const o2::its3::GeometryTGeo* gm) { mGeometry = gm; }

  uint32_t getEventROFrameMin() const { return mEventROFrameMin; }
  uint32_t getEventROFrameMax() const { return mEventROFrameMax; }
  void resetEventROFrames()
  {
    mEventROFrameMin = 0xffffffff;
    mEventROFrameMax = 0;
  }

 private:
  void processHit(const o2::itsmft::Hit& hit, uint32_t& maxFr, int evID, int srcID);
  void registerDigits(o2::itsmft::ChipDigitsContainer& chip, uint32_t roFrame, float tInROF, int nROF,
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

  std::vector<SegmentationSuperAlpide> mSuperSegmentations;
  static constexpr float sec2ns = 1e9;

  o2::itsmft::DigiParams mParams;          ///< digitization parameters
  o2::InteractionTimeRecord mEventTime;    ///< global event time and interaction record
  o2::InteractionRecord mIRFirstSampledTF; ///< IR of the 1st sampled IR, noise-only ROFs will be inserted till this IR only
  double mCollisionTimeWrtROF;
  uint32_t mROFrameMin = 0; ///< lowest RO frame of current digits
  uint32_t mROFrameMax = 0; ///< highest RO frame of current digits
  uint32_t mNewROFrame = 0; ///< ROFrame corresponding to provided time

  uint32_t mEventROFrameMin = 0xffffffff; ///< lowest RO frame for processed events (w/o automatic noise ROFs)
  uint32_t mEventROFrameMax = 0;          ///< highest RO frame forfor processed events (w/o automatic noise ROFs)

  std::unique_ptr<o2::itsmft::AlpideSimResponse> mAlpSimResp; // simulated response

  const o2::its3::GeometryTGeo* mGeometry = nullptr; ///< ITS OR MFT upgrade geometry

  std::vector<o2::itsmft::ChipDigitsContainer> mChips; ///< Array of chips digits containers
  std::deque<std::unique_ptr<ExtraDig>> mExtraBuff;    ///< burrer (per roFrame) for extra digits

  std::vector<o2::itsmft::Digit>* mDigits = nullptr;                       //! output digits
  std::vector<o2::itsmft::ROFRecord>* mROFRecords = nullptr;               //! output ROF records
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels

  ClassDefOverride(Digitizer, 2);
};
} // namespace its3
} // namespace o2

#endif /* ALICEO2_ITS3_DIGITIZER_H */
