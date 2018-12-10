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
  double mTime = 0.;
  int mEventID = 0;
  int mSrcID = 0;
};
} // namespace trd
} // namespace o2
#endif
