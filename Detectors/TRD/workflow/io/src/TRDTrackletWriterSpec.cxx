// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "TRDWorkflowIO/TRDTrackletWriterSpec.h"
#include <SimulationDataFormat/MCTruthContainer.h>
#include <SimulationDataFormat/MCCompLabel.h>
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"

#include <fstream>
#include <iostream>

using namespace o2::framework;

namespace o2
{
namespace trd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getTRDTrackletWriterSpec(bool useMC)
{
  //  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  /* int producejson=1;
  if(producejson){
      //write tracklets in json format for Samesh javascript ui
      //probably only going to be for deubgging.
      //filename format is : E#.sector#.stack#.json E=Eventnumber(nominal) ... will use timestamp
      LOG(info) << " now to produce json";
      ofstream output("E15.0.1.json");
      output << " 10 " << endl;
  }*/
  //LOG(info) << "before writing out the tracklet size is " << Tracklet->size();
  return MakeRootTreeWriterSpec("TRDTrkltWrt",
                                "trdtracklets.root",
                                "o2sim",
                                BranchDefinition<std::vector<o2::trd::Tracklet64>>{InputSpec{"tracklets", "TRD", "TRACKLETS"}, "Tracklet"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{InputSpec{"trklabels", "TRD", "TRKLABELS"}, "TRKLabels", (useMC ? 1 : 0), "TRKLABELS"},
                                BranchDefinition<std::vector<o2::trd::TriggerRecord>>{InputSpec{"tracklettrigs", "TRD", "TRKTRGRD"}, "TrackTrg"})();
};

} // end namespace trd
} // end namespace o2
