// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <Python.h>

using namespace o2;
using namespace o2::framework;

// This is a stateful task, where we send the state downstream.
struct PythonTask {
  void init(InitContext& ic)
  {
     Py_Initialize(); 
     PyRun_SimpleString("print('hello world')\n");
     Py_Finalize();
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<PythonTask>(cfgc, TaskName{"myPythonIntegrationAnalysis"})};
}
