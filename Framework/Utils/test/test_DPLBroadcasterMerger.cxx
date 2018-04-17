//
// Created by Gabriele Gaetano Fronzé on 15/04/2018.
//

#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRef.h"
// FIXME: this should not be needed as the framework should be able to
//        decode TClonesArray by itself.
#include "Framework/TMessageSerializer.h"
#include "DPLBroadcasterMerger.h"
#include "FairMQLogger.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
void defineDataProcessing(WorkflowSpec &specs) {
  WorkflowSpec workflow{
    o2::workflows::DPLBroadcasterMergerWorkflow(),
  };

  specs.swap(workflow);
}
