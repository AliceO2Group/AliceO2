// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author Gabriele Gaetano Fronzé, gfronze@cern.ch

#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRef.h"
// FIXME: this should not be needed as the framework should be able to
//        decode TClonesArray by itself.
#include "Framework/TMessageSerializer.h"
#include "DPLOutputTest.h"
#include "Framework/Logger.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return o2::workflows::DPLOutputTest();
}
