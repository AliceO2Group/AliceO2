/// \file run_sim_phos.C
/// \brief Simple macro to run single particle simulation in PHOS
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>

#include "Rtypes.h"
#include "TGeoManager.h"
#include "TMath.h"
#include "TStopwatch.h"
#include "TString.h"
#include "TSystem.h"

#include "FairBoxGenerator.h"
#include "FairParRootFileIo.h"
#include "FairPrimaryGenerator.h"
#include "FairRunSim.h"
#include "FairRuntimeDb.h"
#include "FairSystemInfo.h"

#include "Field/MagneticField.h"
#include "TGeoGlobalMagField.h"

#include "DetectorsPassive/Cave.h"
#include "PHOSSimulation/Detector.h"
#endif

void run_sim_phos(Int_t nEvents = 10, TString mcEngine = "TGeant3")
{

  FairLogger::GetLogger()->SetLogScreenLevel("DEBUG");
  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH", geom_dir.Data());

  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR", tut_configdir.Data());

  // Output file name
  char fileout[100];
  sprintf(fileout, "AliceO2_%s.phos.mc_%i_event.root", mcEngine.Data(), nEvents);
  TString outFile = fileout;

  // Parameter file name
  char filepar[100];
  sprintf(filepar, "AliceO2_%s.phos.params_%i.root", mcEngine.Data(), nEvents);
  TString parFile = filepar;

  // Debug option
  gDebug = 0;

  // Timer
  TStopwatch timer;
  timer.Start();

  // Create simulation run
  FairRunSim* run = new FairRunSim();

  run->SetName(mcEngine);      // Transport engine
  run->SetOutputFile(outFile); // Output file
  FairRuntimeDb* rtdb = run->GetRuntimeDb();

  // Create media
  run->SetMaterials("media.geo"); // Materials

  // Create geometry
  o2::passive::Cave* cave = new o2::passive::Cave("CAVE");
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  o2::field::MagneticField* magField =
    new o2::field::MagneticField("Maps", "Maps", -1., -1., o2::field::MagFieldParam::k5kG);
  run->SetField(magField);

  // ===| Add PHOS |============================================================
  o2::phos::Detector* phos = new o2::phos::Detector(kTRUE);
  run->AddModule(phos);

  // Create PrimaryGenerator
  FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
  FairBoxGenerator* boxGen = new FairBoxGenerator(22, 10); /*photons*/

  // boxGen->SetThetaRange(0.0, 90.0);
  boxGen->SetEtaRange(-0.3, 0.3);
  boxGen->SetPRange(1., 10.);
  boxGen->SetPhiRange(240., 330.);
  boxGen->SetDebug(kTRUE);

  primGen->AddGenerator(boxGen);

  run->SetGenerator(primGen);

  // store track trajectories
  // run->SetStoreTraj(kTRUE);

  // Initialize simulation run
  run->Init();

  // Runtime database
  Bool_t kParameterMerged = kTRUE;
  FairParRootFileIo* parOut = new FairParRootFileIo(kParameterMerged);
  parOut->open(parFile.Data());
  rtdb->setOutput(parOut);
  rtdb->saveOutput();
  rtdb->print();

  // Start run
  run->Run(nEvents);
  delete run;
  run->CreateGeometryFile("geofile_full.root");

  // Finish
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  // extract max memory usage
  FairSystemInfo sysinfo;

  std::cout << std::endl
            << std::endl;
  std::cout << "Macro finished succesfully." << std::endl;
  std::cout << "Output file is " << outFile << std::endl;
  std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << std::endl
            << std::endl;
  std::cout << "Memory used " << sysinfo.GetMaxMemory() << "\n";
}
