// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITVIEWOBJECTGENERATOR_H
#define O2_FITVIEWOBJECTGENERATOR_H

#include "CCDB/CcdbObjectInfo.h"
#include "FT0Calibration/FT0CalibrationObject.h"
#include "FT0Calibration/FT0ChannelDataTimeSlotContainer.h"

namespace o2::calibration::fit
{

class FITViewObjectGenerator
{

  using VisObjectsType = std::vector<std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TObject>>>;

 public:
  template <typename ObjectToVisualize>
  static VisObjectsType generateViewObjects(const ObjectToVisualize&);

};






}

#endif //O2_FITVIEWOBJECTGENERATOR_H
