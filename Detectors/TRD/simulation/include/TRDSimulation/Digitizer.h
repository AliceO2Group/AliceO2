// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DIGITIZER_H_
#define ALICEO2_TRD_DIGITIZER_H_

#include "TRDSimulation/Detector.h"
#include "TRDBase/Digit.h"
#include "TRDBase/TRDCommonParam.h"

namespace o2
{
namespace trd
{

class TRDGeometry;
class TRDSimParam;
class TRDPadPlane;
class TRDArraySignal;
class PadResponse;

class Digitizer
{
 public:
  Digitizer();
  ~Digitizer() = default;
  void process(std::vector<HitType> const&, DigitContainer_t&);
  void setEventTime(double timeNS) { mTime = timeNS; }
  void setEventID(int entryID) { mEventID = entryID; }
  void setSrcID(int sourceID) { mSrcID = sourceID; }

 private:
  TRDGeometry* mGeo = nullptr;            // access to TRDGeometry
  PadResponse* mPRF = nullptr;            // access to PadResponse
  TRDSimParam* mSimParam = nullptr;       // access to TRDSimParam instance
  TRDCommonParam* mCommonParam = nullptr; // access to TRDCommonParam instance

  double mTime = 0.;
  int mEventID = 0;
  int mSrcID = 0;

  bool mSDigits{ false };             // true: convert signals to summable digits, false by defaults
  std::vector<HitType> mHitContainer; // The container of hits in a given detector

  void getHitContainerPerDetector(const std::vector<HitType>&, std::array<std::vector<HitType>, kNdet>&);
  // Digitization chaing methods
  bool convertHits(const int, const std::vector<HitType>&, SignalContainer_t&); // True if hit-to-signal conversion is successful
  bool convertSignalsToDigits(const int, SignalContainer_t&);                   // True if signal-to-digit conversion is successful
  bool convertSignalsToSDigits(const int, SignalContainer_t&);                  // True if signal-to-sdigit conversion is successful
  bool convertSignalsToADC(const int, SignalContainer_t&);                      // True if signal-to-ADC conversion is successful
  bool diffusion(float, double, double, double&, double&, double&);             // True if diffusion is applied successfully
};
} // namespace trd
} // namespace o2
#endif
