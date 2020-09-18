// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CTFReaderSpec.h

#ifndef O2_CTFREADER_SPEC
#define O2_CTFREADER_SPEC

#include "TFile.h"
#include "TTree.h"
#include <TStopwatch.h>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace ctf
{

using DetID = o2::detectors::DetID;

class CTFReaderSpec : public o2::framework::Task
{
 public:
  CTFReaderSpec(DetID::mask_t dm, const std::string& inp);
  ~CTFReaderSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  DetID::mask_t mDets;             // detectors
  std::vector<std::string> mInput; // input files
  uint32_t mTFCounter = 0;
  size_t mNextToProcess = 0;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getCTFReaderSpec(DetID::mask_t dets, const std::string& inp);

} // namespace ctf
} // namespace o2

#endif /* O2_CTFREADER_SPEC */
