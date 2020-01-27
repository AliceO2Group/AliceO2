// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawReaderSpec.h

#ifndef O2_TOF_RAWREADER
#define O2_TOF_RAWREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TOFReconstruction/Decoder.h"
#include "TOFBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class RawReader : public Task
{
 public:
  RawReader() = default;
  ~RawReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  std::string mFilename;
  std::vector<std::vector<o2::tof::Digit>> mDigits;
};

/// create a processor spec
/// read simulated TOF raws from a root file
framework::DataProcessorSpec getRawReaderSpec();

} // namespace tof
} // namespace o2

#endif /* O2_TOF_RAWREADER */
