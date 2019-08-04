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
#ifndef ALICEO2_ITSMFT_DIGITIZER_H
#define ALICEO2_ITSMFT_DIGITIZER_H

#include <vector>
#include <deque>
#include <memory>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject

#include "ITSMFTSimulation/ChipDigitsContainer.h"
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include "ITSMFTSimulation/DigiParams.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITSMFTBase/GeometryTGeo.h"
#include "ITSMFTBase/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace itsmft
{
class Digitizer : public TObject
{
  using ExtraDig = std::vector<PreDigitLabelRef>; ///< container for extra contributions to PreDigits

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
  void process(const std::vector<Hit>* hits, int evID, int srcID);
  void setEventTime(double t);
  double getEventTime() const { return mEventTime; }
  double getEndTimeOfROFMax() const
  {
    ///< return the time corresponding to end of the last reserved ROFrame : mROFrameMax
    return mParams.getROFrameLength() * (mROFrameMax + 1) + mParams.getTimeOffset();
  }

  void setContinuous(bool v) { mParams.setContinuous(v); }
  bool isContinuous() const { return mParams.isContinuous(); }
  void fillOutputContainer(UInt_t maxFrame = 0xffffffff);

  void setDigiParams(const o2::itsmft::DigiParams& par) { mParams = par; }
  const o2::itsmft::DigiParams& getDigitParams() const { return mParams; }

  // provide the common itsmft::GeometryTGeo to access matrices and segmentation
  void setGeometry(const o2::itsmft::GeometryTGeo* gm) { mGeometry = gm; }

  UInt_t getEventROFrameMin() const { return mEventROFrameMin; }
  UInt_t getEventROFrameMax() const { return mEventROFrameMax; }
  void resetEventROFrames()
  {
    mEventROFrameMin = 0xffffffff;
    mEventROFrameMax = 0;
  }

 private:
  void processHit(const o2::itsmft::Hit& hit, UInt_t& maxFr, int evID, int srcID);
  void registerDigits(ChipDigitsContainer& chip, UInt_t roFrame, float tInROF, int nROF,
                      UShort_t row, UShort_t col, int nEle, o2::MCCompLabel& lbl);

  ExtraDig* getExtraDigBuffer(UInt_t roFrame)
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

  o2::itsmft::DigiParams mParams; ///< digitization parameters
  double mEventTime = 0;          ///< global event time
  bool mContinuous = false;       ///< flag for continuous simulation
  UInt_t mROFrameMin = 0;         ///< lowest RO frame of current digits
  UInt_t mROFrameMax = 0;         ///< highest RO frame of current digits
  UInt_t mNewROFrame = 0;         ///< ROFrame corresponding to provided time

  UInt_t mEventROFrameMin = 0xffffffff; ///< lowest RO frame for processed events (w/o automatic noise ROFs)
  UInt_t mEventROFrameMax = 0;          ///< highest RO frame forfor processed events (w/o automatic noise ROFs)

  std::unique_ptr<o2::itsmft::AlpideSimResponse> mAlpSimResp; // simulated response

  const o2::itsmft::GeometryTGeo* mGeometry = nullptr; ///< ITS OR MFT upgrade geometry

  std::vector<o2::itsmft::ChipDigitsContainer> mChips; ///< Array of chips digits containers
  std::deque<std::unique_ptr<ExtraDig>> mExtraBuff;    ///< burrer (per roFrame) for extra digits

  std::vector<o2::itsmft::Digit>* mDigits = nullptr;                       //! output digits
  std::vector<o2::itsmft::ROFRecord>* mROFRecords = nullptr;               //! output ROF records
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels

  ClassDefOverride(Digitizer, 2);
};
} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_DIGITIZER_H */
