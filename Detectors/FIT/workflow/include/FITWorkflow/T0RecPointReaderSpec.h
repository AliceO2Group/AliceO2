// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   T0RecPointReaderSpec.h

#ifndef O2_T0_RECPOINTREADER
#define O2_T0_RECPOINTREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFITT0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace t0
{

class RecPointReader : public Task
{
 public:
  RecPointReader(bool useMC = true);
  ~RecPointReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true; // use MC truth
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginT0;

  std::vector<o2::t0::RecPoints>* mRecPoints = nullptr;

  std::string mInputFileName = "o2reco_t0.root";
  std::string mRecPointTreeName = "o2sim";
  std::string mRecPointBranchName = "T0Cluster";
};

/// create a processor spec
/// read simulated ITS digits from a root file
framework::DataProcessorSpec getT0RecPointReaderSpec(bool useMC);

} // namespace t0
} // namespace o2

#endif /* O2_T0_RECPOINTREADER */
