// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FT0Calibration/FT0ChannelTimeTimeSlotContainer.h"

using namespace o2::ft0;

FT0ChannelTimeCalibrationObject FT0TimeChannelOffsetCalibrationObjectAlgorithm::generateCalibrationObject(const FT0ChannelTimeTimeSlotContainer& container)
{
  FT0ChannelTimeCalibrationObject calibrationObject;

  for (unsigned int iCh = 0; iCh < o2::ft0::Nchannels_FT0; ++iCh) {
    calibrationObject.mTimeOffsets[iCh] = container.getMeanGaussianFitValue(iCh);
  }

  return calibrationObject;
}
