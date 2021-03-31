// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_CALIBVDRIFT_H_
#define ALICEO2_TRD_CALIBVDRIFT_H_

/// \file   CalibVDrift.h
/// \author Ole Schmidt, ole.schmidt@cern.ch

#include "GPUO2Interface.h"
#include "GPUTRDSpacePointInternal.h" // FIXME: replace with actual data type

namespace o2
{
namespace trd
{

/// \brief VDrift calibration class
///
/// This class is used to determine chamber-wise vDrift values
///
/// origin: TRD
/// \author Ole Schmidt, ole.schmidt@cern.ch

class CalibVDrift
{
 public:
  /// default constructor
  CalibVDrift() = default;

  /// default destructor
  ~CalibVDrift() = default;

  /// set input angular deviations
  void setAngDevInp(const std::vector<o2::gpu::GPUTRDSpacePointInternal> input) { mAngulerDeviationProf = input; }

  /// main processing function
  void process();

 private:
  //FIXME: replace GPUTRDSpacePointInternal with the type Sven creates for collecting the angular deviations
  std::vector<o2::gpu::GPUTRDSpacePointInternal> mAngulerDeviationProf; ///< input TRD track to tracklet angular deviations
};

} // namespace trd

} // namespace o2
#endif
