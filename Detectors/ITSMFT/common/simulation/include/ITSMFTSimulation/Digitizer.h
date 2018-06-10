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
#ifndef ALICEO2_ITS_DIGITIZER_H
#define ALICEO2_ITS_DIGITIZER_H

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
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace ITSMFT
{
class Digitizer : public TObject
{
  using ExtraDig = std::vector<PreDigitLabelRef>; ///< container for extra contributions to PreDigits

 public:
  Digitizer() = default;
  ~Digitizer() override = default;
  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  void setHits(const std::vector<Hit>* hits) { mHits = hits; }
  void setDigits(std::vector<o2::ITSMFT::Digit>* dig) { mDigits = dig; }
  void setMCLabels(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mclb) { mMCLabels = mclb; }

  void init();

  /// Steer conversion of hits to digits
  void process();
  void processHit(const o2::ITSMFT::Hit& hit, UInt_t& maxFr);
  void setEventTime(double t);
  double getEventTime() const { return mEventTime; }

  void setContinuous(bool v) { mParams.setContinuous(v); }
  bool isContinuous() const { return mParams.isContinuous(); }
  void fillOutputContainer(UInt_t maxFrame = 0xffffffff);

  void setDigiParams(const o2::ITSMFT::DigiParams& par) { mParams = par; }
  const o2::ITSMFT::DigiParams& getDigitParams() const { return mParams; }

  void setCoeffToNanoSecond(double cf) { mCoeffToNanoSecond = cf; }
  double getCoeffToNanoSecond() const { return mCoeffToNanoSecond; }

  int getCurrSrcID() const { return mCurrSrcID; }
  int getCurrEvID() const { return mCurrEvID; }

  void setCurrSrcID(int v);
  void setCurrEvID(int v);

  // provide the common ITSMFT::GeometryTGeo to access matrices and segmentation
  void setGeometry(const o2::ITSMFT::GeometryTGeo* gm) { mGeometry = gm; }

 private:
  void registerDigits(ChipDigitsContainer& chip, UInt_t roFrame, float tInROF, int nROF,
                      UShort_t row, UShort_t col, int nEle, o2::MCCompLabel& lbl);

  ExtraDig* getExtraDigBuffer(UInt_t roFrame)
  {
    assert(roFrame >= mROFrameMin);
    int ind = roFrame - mROFrameMin;
    while (ind >= int(mExtraBuff.size())) {
      mExtraBuff.emplace_back(std::make_unique<ExtraDig>());
    }
    return mExtraBuff[ind].get();
  }

  static constexpr float sec2ns = 1e9;

  o2::ITSMFT::DigiParams mParams;  ///< digitization parameters
  double mEventTime = 0;           ///< global event time
  double mCoeffToNanoSecond = 1.0; ///< coefficient to convert event time (Fair) to ns
  bool mContinuous = false;        ///< flag for continuous simulation
  UInt_t mROFrameMin = 0;          ///< lowest RO frame of current digits
  UInt_t mROFrameMax = 0;          ///< highest RO frame of current digits
  UInt_t mNewROFrame = 0;          ///< ROFrame corresponding to provided time
  int mCurrSrcID = 0;              ///< current MC source from the manager
  int mCurrEvID = 0;               ///< current event ID from the manager

  std::unique_ptr<o2::ITSMFT::AlpideSimResponse> mAlpSimResp; // simulated response

  const o2::ITSMFT::GeometryTGeo* mGeometry = nullptr; ///< ITS OR MFT upgrade geometry

  std::vector<o2::ITSMFT::ChipDigitsContainer> mChips; ///< Array of chips digits containers
  std::deque<std::unique_ptr<ExtraDig>> mExtraBuff;    ///< burrer (per roFrame) for extra digits

  const std::vector<Hit>* mHits = nullptr;                                 //! input hits
  std::vector<o2::ITSMFT::Digit>* mDigits = nullptr;                       //! output digits
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels

  ClassDefOverride(Digitizer, 2);
};
}
}

#endif /* ALICEO2_ITS_DIGITIZER_H */
