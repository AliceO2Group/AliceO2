// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_DIGITIZER_H
#define ALICEO2_PHOS_DIGITIZER_H

#include "PHOSBase/Digit.h"
#include "PHOSBase/Geometry.h"
#include "PHOSBase/Hit.h"
#include "PHOSSimulation/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace phos
{
class Digitizer : public TObject
{
 public:
  Digitizer() = default;
  ~Digitizer() override = default;
  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  void init();
  void finish();

  /// Steer conversion of hits to digits
  void process(const std::vector<Hit>& hits, std::vector<Digit>& digits);

  void setEventTime(double t);
  double getEventTime() const { return mEventTime; }

  void setContinuous(bool v) { mContinuous = v; }
  bool isContinuous() const { return mContinuous; }

  //  void setCoeffToNanoSecond(double cf) { mCoeffToNanoSecond = cf; }
  //  double getCoeffToNanoSecond() const { return mCoeffToNanoSecond; }

  void setCurrSrcID(int v);
  int getCurrSrcID() const { return mCurrSrcID; }

  void setCurrEvID(int v);
  int getCurrEvID() const { return mCurrEvID; }

  void setCoeffToNanoSecond(double c) { mCoeffToNanoSecond = c; }

 protected:
  Double_t NonLinearity(Double_t e);
  Double_t DigitizeEnergy(Double_t e);
  Double_t Decalibrate(Double_t e);
  Double_t TimeResolution(Double_t time, Double_t e);
  Double_t SimulateNoiseEnergy();
  Double_t SimulateNoiseTime();

 private:
  const Geometry* mGeometry = nullptr; //!  PHOS geometry
  double mEventTime = 0;               ///< global event time
  double mCoeffToNanoSecond = 1.0;     ///< coefficient to convert event time (Fair) to ns
  bool mContinuous = false;            ///< flag for continuous simulation
  UInt_t mROFrameMin = 0;              ///< lowest RO frame of current digits
  UInt_t mROFrameMax = 0;              ///< highest RO frame of current digits
  int mCurrSrcID = 0;                  ///< current MC source from the manager
  int mCurrEvID = 0;                   ///< current event ID from the manager

  o2::dataformats::MCTruthContainer<o2::phos::MCLabel> mMCTruthContainer; ///<  storage for MC truth information

  ClassDefOverride(Digitizer, 2);
};
} // namespace phos
} // namespace o2

#endif /* ALICEO2_PHOS_DIGITIZER_H */
