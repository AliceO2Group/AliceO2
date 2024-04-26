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

/// @file   TRDCalibReaderSpec.h

#ifndef O2_TRD_CALIBREADER
#define O2_TRD_CALIBREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTRD/AngularResidHistos.h"
#include "DataFormatsTRD/GainCalibHistos.h"
#include "DataFormatsTRD/PHData.h"
#include <vector>

namespace o2
{
namespace trd
{

class TRDCalibReader : public o2::framework::Task
{
 public:
  TRDCalibReader() = default;
  ~TRDCalibReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void connectTree();
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mInFileName{"trdangreshistos.root"};
  std::string mInTreeName{"calibdata"};
  o2::trd::AngularResidHistos mAngResids, *mAngResidPtr = &mAngResids;
  std::vector<o2::trd::PHData> mPHData, *mPHDataPtr = &mPHData;
  std::vector<int> mGainData, *mGainDataPtr = &mGainData;
};

/// create a processor spec
/// read TRD calibration data from a root file
framework::DataProcessorSpec getTRDCalibReaderSpec();

} // namespace trd
} // namespace o2

#endif /* O2_TRD_CALIBREADER */
