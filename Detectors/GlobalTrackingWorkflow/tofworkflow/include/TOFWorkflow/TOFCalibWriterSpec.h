// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFCalibWriterSpec.h

#ifndef TOFWORKFLOW_TOFCALIBWRITER_H_
#define TOFWORKFLOW_TOFCALIBWRITER_H_

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>

using namespace o2::framework;

namespace o2
{
namespace tof
{

class TOFCalibWriter : public Task
{
 public:
  TOFCalibWriter() = default;
  ~TOFCalibWriter() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  std::string mOutFileName; // read from workflow
  std::string mOutTreeName; // read from workflow
};

/// create a processor spec
/// write TOF calbi info in a root file
o2::framework::DataProcessorSpec getTOFCalibWriterSpec();

} // namespace tof
} // namespace o2

#endif /* TOFWORKFLOW_TOFCALIBWRITER_H_ */
