// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "build_geometry.C"
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <FairBoxGenerator.h>
#include <FairPrimaryGenerator.h>
#include <TStopwatch.h>
#include <memory>
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include <SimConfig/SimConfig.h>
#include <Generators/GeneratorFromFile.h>
#endif

void o2sim()
{
  auto& confref = o2::conf::SimConfig::Instance();
  auto genconfig = confref.getGenerator();

  auto run = new FairRunSim();
  run->SetOutputFile("o2sim.root");            // Output file
  run->SetName(confref.getMCEngine().c_str()); // Transport engine

  // construct geometry / including magnetic field
  build_geometry(run);

  // setup generator
  auto primGen = new FairPrimaryGenerator();

  if (genconfig.compare("boxgen") == 0) {
    // a simple "box" generator
    std::cout << "Init box generator\n";
    auto boxGen = new FairBoxGenerator(211, 10); /*protons*/
    boxGen->SetEtaRange(-0.9, 0.9);
    boxGen->SetPRange(0.1, 5);
    boxGen->SetPhiRange(0., 360.);
    boxGen->SetDebug(kTRUE);
    primGen->AddGenerator(boxGen);
  } else if (genconfig.compare("extkin") == 0) {
    // external kinematics
    // needs precense of a kinematics file "Kinematics.root"
    // TODO: make this configurable and check for presence
    auto extGen =  new o2::eventgen::GeneratorFromFile(confref.getExtKinematicsFileName().c_str());
    extGen->SetStartEvent(confref.getStartEvent());
    primGen->AddGenerator(extGen);
    std::cout << "using external kinematics\n";
  }
  else {
    LOG(FATAL) << "Invalid generator" << FairLogger::endl;
  }
  run->SetGenerator(primGen);

  // Timer
  TStopwatch timer;
  timer.Start();

  // run init
  run->Init();
  gGeoManager->Export("O2geometry.root");

  // runtime database
  bool kParameterMerged = true;
  auto rtdb = run->GetRuntimeDb();
  auto parOut = new FairParRootFileIo(kParameterMerged);
  parOut->open("o2sim_par.root");
  rtdb->setOutput(parOut);
  rtdb->saveOutput();
  rtdb->print();

  run->Run(confref.getNEvents());

  // needed ... otherwise nothing flushed?
  delete run;

  // Finish
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  // extract max memory usage
  FairSystemInfo sysinfo;

  std::cout << "\n\n";
  std::cout << "Macro finished succesfully.\n";
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s\n";
  std::cout << "Memory used " << sysinfo.GetMaxMemory() << " MB\n";
}
