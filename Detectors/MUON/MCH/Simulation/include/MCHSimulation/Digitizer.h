// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file Digitizer.h
 * C++  MCH Digitizer.
 * @author Michael Winn, Laurent Aphecetche
 */

#ifndef O2_MCH_SIMULATION_MCHDIGITIZER_H_
#define O2_MCH_SIMULATION_MCHDIGITIZER_H_

#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Geometry.h"
#include "MCHSimulation/Hit.h"
#include "MCHSimulation/Response.h"
#include "MCHMappingInterface/Segmentation.h"

#include "TGeoManager.h"

namespace o2
{
namespace mch
{

class Digitizer
{
 public:
  Digitizer(Int_t mode = 0);

  ~Digitizer() = default;

  void init();

  void setEventTime(double timeNS) { mEventTime = timeNS; }
  void setEventID(int eventID) { mEventID = eventID; }
  void setSrcID(int sID) { mSrcID = sID; }

  //process hits: fill digit vector with digits
  void process(const std::vector<Hit> hits, std::vector<Digit>& digits);

  void fillOutputContainer(std::vector<Digit>& digits);
  void flushOutputContainer(std::vector<Digit>& digits);

  void setContinuous(bool val) { mContinuous = val; }
  bool isContinuous() const { return mContinuous; }

 private:
  double mEventTime;
  int mReadoutWindowCurrent{ 0 };
  int mEventID = 0;
  int mSrcID = 0;

  bool mContinuous = false;

  //number of detector elements
  const static int mNdE = 156;
  // digit per pad
  std::vector<Digit> mDigits;

  std::map<int, int> mdetID;

  std::vector<mapping::Segmentation> mSeg;

  //member with parameters and signal generation
  Response mMuonresponse;

  int processHit(const Hit& hit, double event_time);
};

} // namespace mch
} // namespace o2
#endif
