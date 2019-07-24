// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0RecPointReaderSpec.h

#ifndef O2_FT0_RECPOINTREADER
#define O2_FT0_RECPOINTREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFT0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
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
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFT0;

  std::vector<o2::ft0::RecPoints>* mRecPoints = nullptr;

  std::string mInputFileName = "o2reco_ft0.root";
  std::string mRecPointTreeName = "o2sim";
  std::string mRecPointBranchName = "FT0Cluster";
};

/// create a processor spec
/// read simulated ITS digits from a root file
framework::DataProcessorSpec getFT0RecPointReaderSpec(bool useMC);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0_RECPOINTREADER */
