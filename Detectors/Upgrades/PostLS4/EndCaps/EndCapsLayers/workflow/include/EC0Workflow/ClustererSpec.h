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

#ifndef O2_EC0_CLUSTERERDPL
#define O2_EC0_CLUSTERERDPL

#include <fstream>

#include "EndCapsReconstruction/Clusterer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{

namespace endcaps
{
class Clusterer;
}

namespace ecl
{

class ClustererDPL : public Task
{
 public:
  ClustererDPL(bool useMC) : mUseMC(useMC) {}
  ~ClustererDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  bool mUseMC = true;
  bool mFullClusters = true; // RSTODO: TO BE ELIMINATED but MFT is not ready yet
  bool mPatterns = true;
  int mNThreads = 1;
  std::unique_ptr<std::ifstream> mFile = nullptr;
  std::unique_ptr<o2::endcaps::Clusterer> mClusterer = nullptr;
};

/// create a processor spec
/// run ITS cluster finder
framework::DataProcessorSpec getClustererSpec(bool useMC);

} // namespace ecl
} // namespace o2

#endif /* O2_EC0_CLUSTERERDPL */
