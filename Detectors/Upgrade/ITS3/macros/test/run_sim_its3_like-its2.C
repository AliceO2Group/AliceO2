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

void run_sim_its3(Int_t nEvents = 1, TString mcEngine = "TGeant3")
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
  const int kNLrInner = 3;

  const int kBuildLevel = 0;
  const int kSensTypeID = 0; // dummy id for Alpide sensor

  const float ChipThicknessIB = 50.e-4;

  enum { kRmn,
         kRmd,
         kRmx,
         kNModPerStave,
         kPhi0,
         kNStave,
         kNPar };

  const double tdr5dat[kNLrInner][kNPar] = {
    {2.24, 2.34, 2.67, 9., 16.42, 12}, // for each inner layer: rMin,rMid,rMax,NChip/Stave, phi0, nStaves
    {3.01, 3.15, 3.46, 9., 12.18, 16},
    {3.78, 3.93, 4.21, 9., 9.55, 20},
  };

  static constexpr int NCols = 1024;
  static constexpr int NRows = 512;
  static constexpr float PitchCol = 29.24e-4;
  static constexpr float PitchRow = 26.88e-4;
  static constexpr float PassiveEdgeReadOut = 0.12f;
  static constexpr float PassiveEdgeTop = 37.44e-4;
  static constexpr float ActiveMatrixSizeRows = PitchRow * NRows;
  static constexpr float SensorLayerThickness = 30.e-4;
  static constexpr float SensorSizeRows = ActiveMatrixSizeRows + PassiveEdgeTop + PassiveEdgeReadOut;

  o2::its::Detector* its = new o2::its::Detector(kTRUE, kNLrInner);

  its->setStaveModelIB(o2::its::Detector::kIBModel4);

  for (int idLr = 0; idLr < kNLrInner; idLr++) {
    double rLr = tdr5dat[idLr][kRmd];
    double phi0 = tdr5dat[idLr][kPhi0];

    int nStaveLr = TMath::Nint(tdr5dat[idLr][kNStave]);
    int nModPerStaveLr = TMath::Nint(tdr5dat[idLr][kNModPerStave]);
    int nChipsPerStaveLr = nModPerStaveLr;

    double turbo = radii2Turbo(tdr5dat[idLr][kRmn], rLr, tdr5dat[idLr][kRmx], SensorSizeRows);
    its->defineLayerTurbo(idLr, phi0, rLr, nStaveLr, nChipsPerStaveLr, SensorSizeRows, turbo,
                          ChipThicknessIB, SensorLayerThickness, kSensTypeID, kBuildLevel);
  }

  // *** ITS3 detector definition ends here ***
  run->AddModule(its);

  // ===| Add TPC |============================================================
  o2::tpc::Detector* tpc = new o2::tpc::Detector(kTRUE);
  //tpc->SetGeoFileName("TPCGeometry.root");
  //  run->AddModule(tpc);

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
