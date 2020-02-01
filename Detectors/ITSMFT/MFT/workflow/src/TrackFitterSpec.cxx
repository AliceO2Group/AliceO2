// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitterSpec.cxx
/// \brief Implementation of a data processor to read, refit and send tracks with attached clusters
///
/// \author Philippe Pillot, Subatech

#include "MFTWorkflow/TrackFitterSpec.h"

#include <stdexcept>
#include <list>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MFTTracking/TrackParam.h"
#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/TrackFitter.h"

namespace o2
{
namespace mft
{

using namespace std;
using namespace o2::framework;
using Track = o2::mft::TrackCA;

class TrackFitterTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools
    LOG(INFO) << "initializing track fitter";
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the tracks with attached clusters of the current event,
    /// refit them and send the new version
  }

 private:
  //_________________________________________________________________________________________________
  void copyHeader(const char*& bufferPtr, int& sizeLeft, char*& bufferPtrOut) const
  {
    /// copy header informations from the input buffer to the output message
    /// move the buffer ptr and decrease the size left
    /// throw an exception in case of error
  }

  //_________________________________________________________________________________________________
  void readTrack(const char*& bufferPtr, int& sizeLeft, Track& track, std::list<Cluster>& clusters) const
  {
    /// read the track informations from the buffer
    /// move the buffer ptr and decrease the size left
    /// throw an exception in case of error
  }

  //_________________________________________________________________________________________________
  void writeTrack(Track& track, char*& bufferPtrOut) const
  {
    /// write the track informations to the buffer and move the buffer ptr
  }
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFitterSpec()
{
}

} // namespace mft
} // namespace o2
