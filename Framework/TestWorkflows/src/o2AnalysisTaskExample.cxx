// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2::framework;

// This is a stateful task, where we send the state downstream.
class ATask : public AnalysisTask
{
 public:
  explicit ATask(int state)
    : mSomeState{ state } {}

  void init(InitContext& ic) final
  {
    mSomeState += 1;
  }
  void run(ProcessingContext& pc) final
  {
  }

  void processTracks(std::shared_ptr<arrow::Table> tracks)
  {
  }

 private:
  int mSomeState;
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("mySimpleTrackAnalysis", 0)
  };
}
