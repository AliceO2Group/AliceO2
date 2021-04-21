// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_GPU_DPL_DISPLAY_H
#define O2_GPU_DPL_DISPLAY_H

#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/Task.h"

namespace o2::gpu
{
class O2GPUDPLDisplaySpec : public o2::framework::Task
{
 public:
  O2GPUDPLDisplaySpec(bool useMC, o2::dataformats::GlobalTrackID::mask_t trkMask, o2::dataformats::GlobalTrackID::mask_t clMask) : mUseMC(useMC), mTrkMask(trkMask), mClMask(clMask) {}
  ~O2GPUDPLDisplaySpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  bool mUseMC = false;
  o2::dataformats::GlobalTrackID::mask_t mTrkMask;
  o2::dataformats::GlobalTrackID::mask_t mClMask;
};

} // namespace o2::gpu

#endif
