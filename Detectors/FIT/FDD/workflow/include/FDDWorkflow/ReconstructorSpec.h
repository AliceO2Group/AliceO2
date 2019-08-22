// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ReconstructorSpec.h

#ifndef O2_FDD_RECONSTRUCTORDPL_H
#define O2_FDD_RECONSTRUCTORDPL_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "FDDReconstruction/Reconstructor.h"
#include "DataFormatsFDD/RecPoint.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
{

class FDDReconstructorDPL : public Task
{
 public:
  FDDReconstructorDPL(bool useMC) : mUseMC(useMC) {}
  ~FDDReconstructorDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true;
  std::vector<o2::fdd::RecPoint> mRecPoints;
  o2::fdd::Reconstructor mReco;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFDD;
};

/// create a processor spec
framework::DataProcessorSpec getFDDReconstructorSpec(bool useMC = true);

} // namespace fdd
} // namespace o2

#endif /* O2_FDD_RECONSTRUCTORDPL_H */
