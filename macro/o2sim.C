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
#include <Generators/Pythia8Generator.h>
#include "TVirtualMC.h"
#include "DataFormatsParameters/GRPObject.h"

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
  } else if (genconfig.compare("pythia8") == 0) {
    // pythia8 pp
    // configures pythia for min.bias pp collisions at 14 TeV
    // TODO: make this configurable
    auto py8Gen = new Pythia8Generator();
    py8Gen->SetParameters("Beams:idA 2212"); // p
    py8Gen->SetParameters("Beams:idB 2212"); // p
    py8Gen->SetParameters("Beams:eCM 14000."); // [GeV]
    py8Gen->SetParameters("SoftQCD:inelastic on"); // all inelastic processes
    primGen->AddGenerator(py8Gen);
  }else if (genconfig.compare("pythia8hi") == 0) {
    // pythia8 heavy-ion
    // exploits pythia8 heavy-ion machinery (available from v8.230)
    // configures pythia for min.bias Pb-Pb collisions at 5.52 TeV
    // TODO: make this configurable
    auto py8Gen = new Pythia8Generator();
    py8Gen->SetParameters("Beams:idA 1000822080"); // Pb ion
    py8Gen->SetParameters("Beams:idB 1000822080"); // Pb ion
    py8Gen->SetParameters("Beams:eCM 5520.0"); // [GeV]
    py8Gen->SetParameters("HeavyIon:SigFitNGen 0"); // valid for Pb-Pb 5520 only
    py8Gen->SetParameters("HeavyIon:SigFitDefPar 14.82,1.82,0.25,0.0,0.0,0.0,0.0,0.0"); // valid for Pb-Pb 5520 only
    py8Gen->SetParameters(("HeavyIon:bWidth " + std::to_string(confref.getBMax())).c_str()); // impact parameter from 0-x [fm]
    primGen->AddGenerator(py8Gen);
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

  std::time_t runStart = std::time(nullptr);
    
  // runtime database
  bool kParameterMerged = true;
  auto rtdb = run->GetRuntimeDb();
  auto parOut = new FairParRootFileIo(kParameterMerged);
  parOut->open("o2sim_par.root");
  rtdb->setOutput(parOut);
  rtdb->saveOutput();
  rtdb->print();

  run->Run(confref.getNEvents());

  {
    // store GRPobject
    o2::parameters::GRPObject grp;
    grp.setRun(run->GetRunId());
    grp.setTimeStart( runStart );
    grp.setTimeEnd( std::time(nullptr) );
    TObjArray* modArr = run->GetListOfModules();
    TIter next(modArr);
    FairModule* module = nullptr;
    while ( (module=(FairModule*)next()) ) {
      if (module->GetModId()<o2::Base::DetID::First) {
	continue; // passive
      }
      if (module->GetModId()>o2::Base::DetID::Last) {
	continue; // passive
      }
      grp.addDetReadOut( o2::Base::DetID(module->GetModId()) );    
    }
    grp.print();
    printf("VMC: %p\n",TVirtualMC::GetMC());
    auto field = dynamic_cast<o2::field::MagneticField*>(run->GetField());
    if (field) {
      o2::units::Current_t currDip = field->getCurrentDipole();
      o2::units::Current_t currL3  = field->getCurrentSolenoid();
      grp.setL3Current( currL3 );
      grp.setDipoleCurrent( currDip );
    }
    // todo: save beam information in the grp

    // save
    TFile grpF("o2sim_grp.root","recreate");
    grpF.WriteObjectAny(&grp,grp.Class(),"GRP");    
  }
  
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
