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
  Double_t SimulateNoiseEnergy(void);
  Double_t SimulateNoiseTime(void);

 private:
  const Geometry* mGeometry = nullptr; //!  PHOS geometry
  double mEventTime = 0;               ///< global event time
  double mCoeffToNanoSecond = 1.0;     ///< coefficient to convert event time (Fair) to ns
  bool mContinuous = false;            ///< flag for continuous simulation
  UInt_t mROFrameMin = 0;              ///< lowest RO frame of current digits
  UInt_t mROFrameMax = 0;              ///< highest RO frame of current digits
  int mCurrSrcID = 0;                  ///< current MC source from the manager
  int mCurrEvID = 0;                   ///< current event ID from the manager
  bool mApplyNonLinearity = true;      ///< if Non-linearity will be applied
  bool mApplyDigitization = true;      ///< if energy digitization should be applied
  bool mApplyDecalibration = false;    ///< if de-calibration should be applied
  bool mApplyTimeResolution = true;    ///< if Hit time should be smeared
  double mZSthreshold = 0.005;         ///< Zero Suppression threshold
  double maNL = 0.04;                  ///< Parameter a for Non-Linearity
  double mbNL = 0.2;                   ///< Parameter b for Non-Linearity
  double mcNL = 1.;                    ///< Parameter c for Non-Linearity
  double mADCWidth = 0.005;            ///< Widht of ADC channel used for energy digitization
  double mTimeResolutionA = 2.;        ///< Time resolution parameter A (in ns)
  double mTimeResolutionB = 2.;        ///< Time resolution parameter B (in ns/GeV)
  double mTimeResThreshold = 0.5;      ///< threshold for time resolution calculation (in GeV)
  double mAPDNoise = 0.005;            ///< Electronics (and APD) noise (in GeV)
  double kMinNoiseTime = -200.;        ///< minimum time in noise channels (in ns)
  double kMaxNoiseTime = 2000.;        ///< minimum time in noise channels (in ns)

  //  std::unordered_map<Int_t, std::deque<Digit>> mDigits; ///< used to sort digits by tower

  ClassDefOverride(Digitizer, 1);
};
} // namespace phos
} // namespace o2

#endif /* ALICEO2_PHOS_DIGITIZER_H */
