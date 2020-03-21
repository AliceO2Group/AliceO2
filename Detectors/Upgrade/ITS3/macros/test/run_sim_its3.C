#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <Rtypes.h>
#include <TSystem.h>
#include <TMath.h>
#include <TString.h>
#include <TStopwatch.h>
#include <TGeoManager.h>
#include <array>
#include <vector>

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

void run_sim_its3(Int_t nEvents = 10, TString mcEngine = "TGeant3")
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
  //  o2::its::Detector* its = new o2::its::Detector(kTRUE);
  // *** ITS3 detector definition begins here ***
  // const int kNLrInner = 4;

  const int kBuildLevel = 0;
  const int kSensTypeID = 0; // dummy id for Alpide sensor

  const float ChipThicknessIB = 50.e-4;

  enum { kRmn,
         kZlen,
         kNPar };

  // const double tdr5dat[kNLrInner][kNPar] = {
  //   {2.34, 30.00}, // for each inner layer: rMin,zLen
  //   {3.20, 30.15},
  //   {3.99, 30.15},
  //   {4.21, 30.00}
  // };

  std::vector<std::array<double, 2>> tdr5data;
  tdr5data.emplace_back(std::array<double, 2>{2.34, 27.00});
  tdr5data.emplace_back(std::array<double, 2>{3.15, 27.15});
  tdr5data.emplace_back(std::array<double, 2>{3.93, 27.15});
  tdr5data.emplace_back(std::array<double, 2>{19.4, 80.f});
  tdr5data.emplace_back(std::array<double, 2>{24.7, 80.f});
  tdr5data.emplace_back(std::array<double, 2>{35.3, 150.f});
  tdr5data.emplace_back(std::array<double, 2>{40.5, 150.00});
  tdr5data.emplace_back(std::array<double, 2>{70.66, 150.00});
  tdr5data.emplace_back(std::array<double, 2>{100.00, 150.00});

  static constexpr float SensorLayerThickness = 30.e-4;

  // o2::its::Detector* its = new o2::its::Detector(kTRUE, kNLrInner);
  o2::its::Detector* its = new o2::its::Detector(kTRUE, tdr5data.size());
  //  its->setStaveModelIB(o2::its::Detector::kIBModel4);
  its->setStaveModelOB(o2::its::Detector::kOBModel2);
  its->createOuterBarrel(kFALSE);

  auto idLayer{0};
  for (auto& layerData : tdr5data) {
    double rLr = layerData[kRmn];
    double zlen = layerData[kZlen];
    its->defineInnerLayerITS3(idLayer, rLr, zlen, SensorLayerThickness, kSensTypeID, kBuildLevel);
    ++idLayer;
  }
  // *** ITS3 detector definition ends here ***
  run->AddModule(its);

  // ===| Add TPC |============================================================
  o2::tpc::Detector* tpc = new o2::tpc::Detector(kTRUE);
  //tpc->SetGeoFileName("TPCGeometry.root");
  //  run->AddModule(tpc);

  // Create PrimaryGenerator
  FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
  FairBoxGenerator* boxGen = new FairBoxGenerator(2212, 20); /*protons*/

  //boxGen->SetThetaRange(0.0, 90.0);
  boxGen->SetEtaRange(-0.9, 0.9);
  // boxGen->SetPRange(0.5, 0.51);
  boxGen->SetPtRange(0.2, 0.6);
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
