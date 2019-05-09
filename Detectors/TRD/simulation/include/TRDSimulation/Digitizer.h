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

#include "TRDBase/Digit.h"
#include "TRDSimulation/Detector.h"

namespace o2
{
namespace trd
{

class Digitizer
{
 public:
  //
  Digitizer();
  ~Digitizer();
  void process(std::vector<o2::trd::HitType> const&,
               std::vector<o2::trd::Digit>&);
  void setEventTime(double timeNS) { mTime = timeNS; }
  void setEventID(int entryID) { mEventID = entryID; }
  void setSrcID(int sourceID) { mSrcID = sourceID; }

 private:
  TRDGeometry* mGeom = nullptr;

  double mTime = 0.;
  int mEventID = 0;
  int mSrcID = 0;

  bool mSDigits; // true: convert signals to summable digits, false by default

  std::vector<o2::trd::HitType> mHitContainer; // The container of hits in a given detector

  bool getHitContainer(const int, const std::vector<o2::trd::HitType>&, std::vector<o2::trd::HitType>&); // True if there are hits in the detector
  // Digitization chaing methods
  bool convertHits(const int, const std::vector<o2::trd::HitType>&, int&); // True if hit-to-signal conversion is successful
  bool convertSignalsToDigits(const int, int&);                            // True if signal-to-digit conversion is successful
  bool convertSignalsToSDigits(const int, int&);                           // True if singal-to-sdigit conversion is successful
  bool convertSignalsToADC(const int, int&);                               // True if signal-to-ADC conversion is successful
  bool diffusion(float, double, double, double&, double&, double&);        // True if diffusion is applied successfully
};
} // namespace trd
} // namespace o2
#endif
