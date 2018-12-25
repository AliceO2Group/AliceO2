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

#ifndef O2_ITS_CLUSTERERDPL
#define O2_ITS_CLUSTERERDPL

#include <fstream>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "ITSMFTReconstruction/Clusterer.h"

using namespace o2::framework;

namespace o2
{
namespace ITS
{

class ClustererDPL : public Task
{
 public:
  ClustererDPL() = default;
  ~ClustererDPL() = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  std::unique_ptr<std::ifstream> mFile = nullptr;
  std::unique_ptr<o2::ITSMFT::Clusterer> mClusterer = nullptr;
};

/// create a processor spec
/// run ITS cluster finder
framework::DataProcessorSpec getClustererSpec();

} // namespace ITS
} // namespace o2

#endif /* O2_ITS_CLUSTERERDPL */
