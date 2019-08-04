#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TSystem.h>
#include <TMath.h>
#include <TString.h>
#include <TStopwatch.h>

#include "FairRunSim.h"
#include "FairRuntimeDb.h"
#include "FairPrimaryGenerator.h"
#include "FairBoxGenerator.h"
#include "FairParRootFileIo.h"
#include "TGeoManager.h"

#include "DetectorsPassive/Cave.h"
#include "Field/MagneticField.h"
#include "EMCALSimulation/Detector.h"
#endif

extern TSystem* gSystem;

double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
{
  // compute turbo angle from radii and sensor width
  return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
}

void run_sim_emcal(Int_t nEvents = 1, TString mcEngine = "TGeant3")
{
  FairLogger::GetLogger()->SetLogScreenLevel("DEBUG2");
  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH", geom_dir.Data());

  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR", tut_configdir.Data());

  // Output file name
  char fileout[100];
  sprintf(fileout, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TString outFile = fileout;

  // Parameter file name
  char filepar[100];
  sprintf(filepar, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TString parFile = filepar;

  // In general, the following parts need not be touched

  // Debug option
  gDebug = 0;

  // Timer
  TStopwatch timer;
  timer.Start();

  // CDB manager
  //   o2::ccdb::Manager *cdbManager = o2::ccdb::Manager::Instance();
  //   cdbManager->setDefaultStorage("local://$ALICEO2/tpc/dirty/o2cdb");
  //   cdbManager->setRun(0);

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

  /*FairConstField field;
   field.SetField(0., 0., 5.); //in kG
   field.SetFieldRegion(-5000.,5000.,-5000.,5000.,-5000.,5000.); //in c
  */
  o2::field::MagneticField field("field", "field +5kG");
  run->SetField(&field);

  o2::emcal::Detector* emcal = new o2::emcal::Detector(kTRUE);
  run->AddModule(emcal);

  // Create PrimaryGenerator
  FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
  FairBoxGenerator* boxGen = new FairBoxGenerator(11, 100); // electrons

  //boxGen->SetThetaRange(0.0, 90.0);
  boxGen->SetEtaRange(-0.9, 0.9);
  boxGen->SetPtRange(5, 5.01);
  boxGen->SetPhiRange(0., 360.);
  boxGen->SetDebug(kFALSE);

  primGen->AddGenerator(boxGen);

  run->SetGenerator(primGen);
  run->SetStoreTraj(kTRUE);

  // store track trajectories
  //run->SetStoreTraj(kTRUE);

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

  // Finish
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  // Write geometry
  gGeoManager->Export("geometry.root", "geometry");

  cout << "Output file is " << outFile << endl;
  cout << "Parameter file is " << parFile << endl;
  cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl
       << endl;
  cout << endl
       << endl;
  cout << "Macro finished succesfully." << endl;
  cout << endl
       << endl;
}
