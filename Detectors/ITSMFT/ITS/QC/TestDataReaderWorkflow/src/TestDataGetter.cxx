// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>

#include "Framework/ControlService.h"
#include "ITSQCDataReaderWorkflow/TestDataGetter.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

void TestDataGetter::init(InitContext& ic)
{
  LOG(DEBUG) << "Now Working on the GETTER BROS";
}

void TestDataGetter::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "START Getter";
  auto digits = pc.inputs().get<const std::vector<o2::itsmft::Digit>>("digits");
  LOG(DEBUG) << "Digit Size Getting For This TimeFrame (Event) = " << digits.size();

  int Run = pc.inputs().get<int>("Run");
  LOG(DEBUG) << "New " << Run;

  /*
			int ResetDecision = pc.inputs().get<int>("in");
			LOG(DEBUG) << "Reset Histogram Decision = " << ResetDecision;
		
			o2::itsmft::Digit digit = pc.inputs().get<o2::itsmft::Digit>("digits");
			LOG(DEBUG) << "Chip ID Getting " << digit.getChipIndex() << " Row = " << digit.getRow() << "   Column = " << digit.getColumn();
			*/

  //pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
}

DataProcessorSpec getTestDataGetterSpec()
{
  return DataProcessorSpec{
    "its-rawpixel-getter",
    Inputs{
      InputSpec{"digits", "ITS", "DIGITS", 0, Lifetime::Timeframe},
      InputSpec{"in", "ITS", "TEST", 0, Lifetime::Timeframe},
      //		InputSpec{ "Run", "TST", "TEST2", 0, Lifetime::Timeframe },
    },
    Outputs{},
    AlgorithmSpec{adaptFromTask<TestDataGetter>()},
  };
}
} // namespace its
} // namespace o2
