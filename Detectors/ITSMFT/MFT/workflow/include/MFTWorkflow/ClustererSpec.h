// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClustererSpec.h

#ifndef O2_MFT_CLUSTERERDPL_H_
#define O2_MFT_CLUSTERERDPL_H_

#include <fstream>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "ITSMFTReconstruction/Clusterer.h"

namespace o2
{
namespace mft
{

class ClustererDPL : public o2::framework::Task
{
 public:
  ClustererDPL() = default;
  ~ClustererDPL() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  int mState = 0;
  std::unique_ptr<std::ifstream> mFile = nullptr;
  std::unique_ptr<o2::itsmft::Clusterer> mClusterer = nullptr;
};

/// create a processor spec
/// run MFT cluster finder
o2::framework::DataProcessorSpec getClustererSpec();

} // namespace MFT
} // namespace o2

#endif /* O2_MFT_CLUSTERERDPL */
