// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/MetricsService.h"
// FIXME: this should not be needed as the framework should be able to
//        decode TClonesArray by itself.
#include "Framework/TMessageSerializer.h"
#include "o2_sim_its_ALP3.h"
#include "FairMQLogger.h"
#include <TClonesArray.h>
#include <TH1F.h>

using namespace o2::framework;
using namespace o2::workflows;

using DataHeader = o2::header::DataHeader;
using DataOrigin = o2::header::DataOrigin;
using DataDescription = o2::header::DataDescription;

// This is how you can define your processing in a declarative way
void defineDataProcessing(WorkflowSpec &specs) {
  WorkflowSpec workflow{
    sim_its_ALP3(),
  };

  specs.swap(workflow);
}
