// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDWorkflowIO/TRDCalibratedTrackletWriterSpec.h"
#include "DataFormatsTRD/CalibratedTracklet.h"

#include "DPLUtils/MakeRootTreeWriterSpec.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getTRDCalibratedTrackletWriterSpec()
{
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;

  return MakeRootTreeWriterSpec("calibrated-tracklet-writer",
                                "trdcalibratedtracklets.root",
                                "ctracklets",
                                BranchDefinition<std::vector<CalibratedTracklet>>{InputSpec{"ctracklets", "TRD", "CTRACKLETS"}, "CTracklets"})();
  // BranchDefinition<std::vector<o2::trd::TriggerRecord>>{InputSpec{"tracklettrigs", "TRD", "TRKTRGRD"}, "TrackTrg"})();
};

} // end namespace trd
} // end namespace o2
