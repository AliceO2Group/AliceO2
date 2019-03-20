// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClustererSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "/data/zhaozhong/alice/O2/Detectors/ITSMFT/ITS/workflow/include/ITSWorkflow/DummySpec.h"

//#include "DetectorsBase/GeometryManager.h"
//#include "ITSBase/GeometryTGeo.h"

using namespace o2::framework;

namespace o2
{
namespace ITS
{

void DummyDPL::init(InitContext& ic)
{
 // o2::Base::GeometryManager::loadGeometry(); // for generating full clusters
 // o2::ITS::GeometryTGeo* geom = o2::ITS::GeometryTGeo::Instance();
}

void DummyDPL::run(ProcessingContext& pc)
{

  auto digits = pc.inputs().get<const std::vector<o2::ITSMFT::Digit>>("digits");
  auto labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
  auto rofs = pc.inputs().get<const std::vector<o2::ITSMFT::ROFRecord>>("ROframes");
  auto mc2rofs = pc.inputs().get<const std::vector<o2::ITSMFT::MC2ROFRecord>>("MC2ROframes");

  LOG(INFO) << "ITSClusterer pulled " << digits.size() << " digits, "
            << labels->getIndexedSize() << " MC label objects, in "
            << rofs.size() << " RO frames and "
            << mc2rofs.size() << " MC events";
	//reader.setDigits(&digits);
	LOG(INFO) << "DONE SET DIGITS " << digits.size() << " digits, ";

 }

DataProcessorSpec getDummySpec()
{
  return DataProcessorSpec{
    "its-dummy",
    Inputs{
      InputSpec{ "digits", "ITS", "DIGITS", 0, Lifetime::Timeframe },
      InputSpec{ "labels", "ITS", "DIGITSMCTR", 0, Lifetime::Timeframe },
      InputSpec{ "ROframes", "ITS", "ITSDigitROF", 0, Lifetime::Timeframe },
      InputSpec{ "MC2ROframes", "ITS", "ITSDigitMC2ROF", 0, Lifetime::Timeframe } },
    Outputs{
      OutputSpec{ "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe },
      OutputSpec{ "ITS", "CLUSTERS", 0, Lifetime::Timeframe },
      OutputSpec{ "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe },
      OutputSpec{ "ITS", "ITSClusterROF", 0, Lifetime::Timeframe },
      OutputSpec{ "ITS", "ITSClusterMC2ROF", 0, Lifetime::Timeframe } },
    AlgorithmSpec{ adaptFromTask<DummyDPL>() },
    Options{
      { "its-dictionary-file", VariantType::String, "complete_dictionary.bin", { "Name of the cluster-topology dictionary file" } } }
  };
}

} // namespace ITS
} // namespace o2
