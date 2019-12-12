// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDTRAPSIMULATORTRACKLETWRITER_H
#define O2_TRDTRAPSIMULATORTRACKLETWRITER_H

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "TRDBase/Digit.h"
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TRDBase/MCLabel.h"
#include "TRDBase/Tracklet.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getTRDTrackletWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("TRDTrackletWriter",
                                "trdtracklets.root",
                                "o2sim",
                                1,
                              //  BranchDefinition<std::vector<o2::trd::TrackletMCM>>{InputSpec{"tracklets", "TRD", "TRACKLETS"}, "TRDTracklet"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::trd::MCLabel>>{InputSpec{"labels", "TRD", "LABELS"}, "TRDMCLabels"})();
  //TODO maybe dont pass the labels through, come back and check this 
}

} // end namespace trd
} // end namespace o2

#endif // O2_TRDTRAPSIMULATORTRACKLETWRITER_H
