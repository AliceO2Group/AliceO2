// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDAngularResiduals.h
/// \brief This data structure stores the angular residuals between the TRD tracks and their tracklets

/// \author Sven Hoppner
// \E-Mail sven.hoppner@cern.ch

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct AngularResiduals {
  //float mImpactAngle;                 // Impact Angle of track
  unsigned short mTrackletCounter; //counter of the tracklets
  float mAngleDiffSum;             //Sum of Angle Difference of tracklet to track for each bin
  //short mDetectorId;                  // Detector index

  GPUd() AngularResiduals() : mTrackletCounter(0), mAngleDiffSum(0)
  {
  }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE
