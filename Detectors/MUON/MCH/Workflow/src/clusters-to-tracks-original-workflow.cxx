// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file clusters-to-tracks-original-workflow.cxx
/// \brief Implementation of a DPL device to run the original track finder algorithm
///
/// \author Philippe Pillot, Subatech

#include "Framework/runDataProcessing.h"

#include "TrackFinderOriginalSpec.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  return WorkflowSpec{o2::mch::getTrackFinderOriginalSpec()};
}
