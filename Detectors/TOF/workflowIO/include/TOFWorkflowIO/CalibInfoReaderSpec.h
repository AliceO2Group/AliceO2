// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CalibInfoReaderSpec.h

#ifndef O2_TOF_CALIBINFOREADER
#define O2_TOF_CALIBINFOREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Diagnostic.h"
#include <string>

class TTree;

using namespace o2::framework;

namespace o2
{
namespace tof
{

class CalibInfoReader : public Task
{
 public:
  CalibInfoReader(int instance, int ninstances, const char* filename, bool toftpc = false) : mInstance(instance), mNinstances(ninstances), mFileName(filename), mTOFTPC(toftpc) {}
  ~CalibInfoReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  int mInstance;
  int mNinstances;
  std::string mFileName{};
  FILE* mFile = nullptr;
  TTree* mTree = nullptr;
  bool mTOFTPC = false;
  int mCurrentEntry = 0;
  int mGlobalEntry = 0;
  std::vector<o2::dataformats::CalibInfoTOF> mVect, *mPvect = &mVect;
  o2::tof::Diagnostic mDia, *mPdia = &mDia;
  std::vector<std::pair<int, unsigned long>> mIndices;
};

/// create a processor spec
/// read simulated TOF digits from a root file
framework::DataProcessorSpec getCalibInfoReaderSpec(int instance, int ninstances, const char* filename, bool toftpc = false);

} // namespace tof
} // namespace o2

#endif /* O2_TOF_CALIBINFOREADER */
