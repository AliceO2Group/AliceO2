// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_
#define DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_

#include "HMPIDBase/Digit.h"
#include "HMPIDSimulation/Detector.h" // for the hit
#include <vector>

namespace o2
{
namespace hmpid
{

class HMPIDDigitizer
{
 public:
  void setEventTime(double timeNS) { mTime = timeNS; }
  void setEventID(int eventID) { mEventID = eventID; }
  void setSrcID(int sID) { mSrcID = sID; }

  // this will process hits and fill the digit vector with digits which are finalized
  void process(std::vector<o2::hmpid::HitType> const&, std::vector<o2::hmpid::Digit>& digit);

 private:
  double mTime = 0.;
  int mEventID = 0;
  int mSrcID = 0;

  // internal buffers for digits
  std::vector<o2::hmpid::Digit> mSummable;
  std::vector<o2::hmpid::Digit> mFinal;

  // other stuff needed for digitizaton

  ClassDefNV(HMPIDDigitizer, 1);
};
} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_ */
