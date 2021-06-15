// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  std::vector<o2::trd::AngularResidHistos> mAngResids, *mAngResidPtr = &mAngResids;
};

/// create a processor spec
/// read TRD calibration data from a root file
framework::DataProcessorSpec getTRDCalibReaderSpec();

} // namespace trd
} // namespace o2

#endif /* O2_TRD_CALIBREADER */
