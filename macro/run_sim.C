#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <Rtypes.h>
#include <TSystem.h>
#include <TMath.h>
#include <TString.h>
#include <TStopwatch.h>
#include <TGeoManager.h>

#include "FairRunSim.h"
#include "FairRuntimeDb.h"
#include "FairPrimaryGenerator.h"
#include "FairBoxGenerator.h"
#include "FairParRootFileIo.h"

#include "DetectorsPassive/Cave.h"
#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSSimulation/Detector.h"
#include "ITSSimulation/Detector.h"
#include "TPCSimulation/Detector.h"
#endif

double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
{
  // compute turbo angle from radii and sensor width
  return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
}

void run_sim(Int_t nEvents = 2, TString mcEngine = "TGeant3")
{
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

  // gSystem->Load("libAliceO2Base");
  // gSystem->Load("libAliceO2its");

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

  //  FairDetector*  tpc = new O2tpc("TPCV2");
  //  tpc->SetGeometry();
  //  run->AddModule(tpc);

  //  TGeoGlobalMagField::Instance()->SetField(new o2::field::MagneticField("Maps","Maps", -1., -1., o2::field::MagneticField::k5kG));
  o2::field::MagneticField field("field", "field +5kG");
  run->SetField(&field);

  // ===| Add ITS |============================================================
  o2::its::Detector* its = new o2::its::Detector(kTRUE);
  run->AddModule(its);

  // ===| Add TPC |============================================================
  o2::tpc::Detector* tpc = new o2::tpc::Detector(kTRUE);
  //tpc->SetGeoFileName("TPCGeometry.root");
  run->AddModule(tpc);

  // Create PrimaryGenerator
  FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
  FairBoxGenerator* boxGen = new FairBoxGenerator(2212, 1); /*protons*/

  //boxGen->SetThetaRange(0.0, 90.0);
  boxGen->SetEtaRange(-0.9, 0.9);
  boxGen->SetPRange(100, 100.01);
  boxGen->SetPhiRange(0., 360.);
  boxGen->SetDebug(kTRUE);

  primGen->AddGenerator(boxGen);

  run->SetGenerator(primGen);

  // store track trajectories
  //  run->SetStoreTraj(kTRUE);

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
  //  run->CreateGeometryFile("geofile_full.root");

  // Finish
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();
  std::cout << std::endl
            << std::endl;
  std::cout << "Macro finished succesfully." << std::endl;
  std::cout << "Output file is " << outFile << std::endl;
  std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << std::endl
            << std::endl;

  delete run;

  return;
}
