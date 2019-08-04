// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Declaration of a transient MC label class for TRD

#ifndef ALICEO2_TRD_MCLABEL_H_
#define ALICEO2_TRD_MCLABEL_H_

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace trd
{

class MCLabel : public o2::MCCompLabel
{
 private:
  bool mIsDigit{false};

 public:
  MCLabel() = default;
  MCLabel(int trackID, int eventID, int srcID, int isDigit)
    : o2::MCCompLabel(trackID, eventID, srcID, false), mIsDigit(isDigit) {}
  int isDigit() const { return mIsDigit; }

  ClassDefNV(MCLabel, 1);
};

} // namespace trd
} // namespace o2

#endif
