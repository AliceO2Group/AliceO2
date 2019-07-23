// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0ReconstructorSpec.h

#ifndef O2_FT0RECONSTRUCTORDPL_H
#define O2_FT0RECONSTRUCTORDPL_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "FT0Reconstruction/CollisionTimeRecoTask.h"
#include "DataFormatsFT0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

class FT0ReconstructorDPL : public Task
{
 public:
  FT0ReconstructorDPL(bool useMC) : mUseMC(useMC) {}
  ~FT0ReconstructorDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true;
  std::vector<o2::ft0::RecPoints> mRecPoints;
  o2::ft0::CollisionTimeRecoTask mReco;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFT0;
};

/// create a processor spec
framework::DataProcessorSpec getFT0ReconstructorSpec(bool useMC = true);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0RECONSTRUCTORDPL_H */
