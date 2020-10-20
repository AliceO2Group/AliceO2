// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MEANVERTEX_CALIB_DEVICE_H
#define O2_MEANVERTEX_CALIB_DEVICE_H

/// @file   MeanVertexCalibratorSpec.h
/// @brief  Device to calibrate MeanVertex

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DetectorsCalibration/MeanVertexCalibrator.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class MeanVertexCalibDevice : public Task
{
 public:
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void sendOutput(DataAllocator& output);

  std::unique_ptr<o2::calibration::MeanVertexCalibrator> mCalibrator;
};

} // namespace calibration

namespace framework
{
DataProcessorSpec getMeanVertexCalibDeviceSpec();

} // namespace framework
} // namespace o2

#endif
