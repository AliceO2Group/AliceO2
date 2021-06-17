// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CalibClusReaderSpec.h

#ifndef O2_TOF_CALIBCLUSREADER
#define O2_TOF_CALIBCLUSREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTOF/CosmicInfo.h"
#include "DataFormatsTOF/CalibInfoCluster.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class CalibClusReader : public Task
{
 public:
  CalibClusReader(bool isCosmics) : mIsCosmics(isCosmics) {}
  ~CalibClusReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  std::unique_ptr<TFile> mFile = nullptr;
  std::unique_ptr<TTree> mTree = nullptr;
  int mState = 0;
  int mCurrentEntry = 0;
  bool mIsCosmics = false;
  std::vector<CalibInfoCluster> mClusInfos, *mPclusInfos = &mClusInfos;
  std::vector<CosmicInfo> mCosmicInfo, *mPcosmicInfo = &mCosmicInfo;
};

/// create a processor spec
/// read cluster calib infos from a root file
DataProcessorSpec getCalibClusReaderSpec(bool isCosmics);

} // namespace tof
} // namespace o2

#endif /* O2_TOF_CALIBCLUSREADER */
