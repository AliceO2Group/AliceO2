#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <TSystem.h>
  #include <TMath.h>
  #include <TString.h>
  #include <TStopwatch.h>

  #include "FairRunSim.h"
  #include "FairRuntimeDb.h"
  #include "FairPrimaryGenerator.h"
  #include "FairBoxGenerator.h"
  #include "FairParRootFileIo.h"
  #include "FairConstField.h"

  #include "DetectorsPassive/Cave.h"
  #include "Field/MagneticField.h"
  #include "ITSBase/GeometryTGeo.h"
  #include "ITSMFTBase/SegmentationPixel.h"
  #include "ITSSimulation/Detector.h"
#endif

extern TSystem *gSystem;

double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
{
  // compute turbo angle from radii and sensor width
  return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
}

void run_sim_its(Int_t nEvents = 10, TString mcEngine = "TGeant3")
{
  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH",geom_dir.Data());


  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR",tut_configdir.Data());

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
  //   o2::CDB::Manager *cdbManager = o2::CDB::Manager::Instance();
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
  o2::Passive::Cave* cave = new o2::Passive::Cave("CAVE");
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  /*FairConstField field;
   field.SetField(0., 0., 5.); //in kG
   field.SetFieldRegion(-5000.,5000.,-5000.,5000.,-5000.,5000.); //in c
  */
  o2::field::MagneticField field("field","field +5kG");
  run->SetField(&field);

  o2::ITS::Detector* its = new o2::ITS::Detector("ITS", kTRUE, 7);
  run->AddModule(its);

  // build ITS upgrade detector
  // sensitive area 13x15mm (X,Z) with 20x20 micron pitch, 2mm dead zone on readout side and 50
  // micron guardring
  const double kSensThick = 18e-4;
  const double kPitchX = 20e-4;
  const double kPitchZ = 20e-4;
  const int kNRow = 650;
  const int kNCol = 1500;
  const double kSiThickIB = 150e-4;
  const double kSiThickOB = 150e-4;
  //  const double kSensThick = 120e-4;   // -> sensor Si thickness

  const double kReadOutEdge = 0.2; // width of the readout edge (passive bottom)
  const double kGuardRing = 50e-4; // width of passive area on left/right/top of the sensor

  const int kNLr = 7;
  const int kNLrInner = 3;
  const int kBuildLevel = 0;
  enum { kRmn, kRmd, kRmx, kNModPerStave, kPhi0, kNStave, kNPar };
  // Radii are from last TDR (ALICE-TDR-017.pdf Tab. 1.1, rMid is mean value)
  const double tdr5dat[kNLr][kNPar] = {
    { 2.24, 2.34, 2.67, 9., 16.37,
      12 }, // for each inner layer: rMin,rMid,rMax,NChip/Stave, phi0, nStaves
    { 3.01, 3.15, 3.46, 9., 12.03, 16 },
    { 3.78, 3.93, 4.21, 9., 10.02, 20 },
    { -1, 19.6, -1, 4., 0., 24 }, // for others: -, rMid, -, NMod/HStave, phi0, nStaves // 24 was 49
    { -1, 24.55, -1, 4., 0., 30 }, // 30 was 61
    { -1, 34.39, -1, 7., 0., 42 }, // 42 was 88
    { -1, 39.34, -1, 7., 0., 48 }  // 48 was 100
  };
  const int nChipsPerModule = 7; // For OB: how many chips in a row

  // Delete the segmentations from previous runs
  gSystem->Exec(" rm itsSegmentations.root ");

  // create segmentations:
  o2::ITSMFT::SegmentationPixel* seg0 = new o2::ITSMFT::SegmentationPixel(
    0,           // segID (0:9)
    1,           // chips per module
    kNCol,       // ncols (total for module)
    kNRow,       // nrows
    kPitchX,     // default row pitch in cm
    kPitchZ,     // default col pitch in cm
    kSensThick,  // sensor thickness in cm
    -1,          // no special left col between chips
    -1,          // no special right col between chips
    kGuardRing,  // left
    kGuardRing,  // right
    kGuardRing,  // top
    kReadOutEdge // bottom
    );           // see SegmentationPixel.h for extra options
  seg0->Store(o2::ITS::GeometryTGeo::getITSsegmentationFileName());
  seg0->Print();

  double dzLr, rLr, phi0, turbo;
  int nStaveLr, nModPerStaveLr, idLr;

  its->setStaveModelIB(o2::ITS::Detector::kIBModel22);
  its->setStaveModelOB(o2::ITS::Detector::kOBModel1);

  const int kNWrapVol = 3;
  const double wrpRMin[kNWrapVol] = { 2.1, 15.0, 32.0 };
  const double wrpRMax[kNWrapVol] = { 7.0, 27.0 + 2.5, 43.0 + 1.5 };
  const double wrpZSpan[kNWrapVol] = { 28.0, 86.0, 150.0 };

  its->setNumberOfWrapperVolumes(kNWrapVol); // define wrapper volumes for layers

  for (int iw = 0; iw < kNWrapVol; iw++) {
    its->defineWrapperVolume(iw, wrpRMin[iw], wrpRMax[iw], wrpZSpan[iw]);
  }

  for (int idLr = 0; idLr < kNLr; idLr++) {
    rLr = tdr5dat[idLr][kRmd];
    phi0 = tdr5dat[idLr][kPhi0];

    nStaveLr = TMath::Nint(tdr5dat[idLr][kNStave]);
    nModPerStaveLr = TMath::Nint(tdr5dat[idLr][kNModPerStave]);
    int nChipsPerStaveLr = nModPerStaveLr;
    if (idLr >= kNLrInner) {
      nChipsPerStaveLr *= nChipsPerModule;
      its->defineLayer(idLr, phi0, rLr, nChipsPerStaveLr * seg0->Dz(), nStaveLr, nModPerStaveLr,
                       kSiThickOB, seg0->Dy(), seg0->getChipTypeID(), kBuildLevel);
      //      printf("Add Lr%d: R=%6.2f DZ:%6.2f Staves:%3d NMod/Stave:%3d\n",
      //	     idLr,rLr,nChipsPerStaveLr*seg0->Dz(),nStaveLr,nModPerStaveLr);
    } else {
      turbo = -radii2Turbo(tdr5dat[idLr][kRmn], rLr, tdr5dat[idLr][kRmx], seg0->Dx());
      its->defineLayerTurbo(idLr, phi0, rLr, nChipsPerStaveLr * seg0->Dz(), nStaveLr,
                            nChipsPerStaveLr, seg0->Dx(), turbo, kSiThickIB, seg0->Dy(),
                            seg0->getChipTypeID(), kBuildLevel);
      //      printf("Add Lr%d: R=%6.2f DZ:%6.2f Turbo:%+6.2f Staves:%3d NMod/Stave:%3d\n",
      //	     idLr,rLr,nChipsPerStaveLr*seg0->Dz(),turbo,nStaveLr,nModPerStaveLr);
    }
  }

  // Create PrimaryGenerator
  FairPrimaryGenerator* primGen = new FairPrimaryGenerator();
  FairBoxGenerator* boxGen = new FairBoxGenerator(211, 100); //pions

  //boxGen->SetThetaRange(0.0, 90.0);
  boxGen->SetEtaRange(-0.9,0.9);
  boxGen->SetPtRange(1, 1.01);
  boxGen->SetPhiRange(0., 360.);
  boxGen->SetDebug(kFALSE);

  primGen->AddGenerator(boxGen);

  run->SetGenerator(primGen);

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

  cout << "Output file is " << outFile << endl;
  cout << "Parameter file is " << parFile << endl;
  cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl << endl;
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
  cout << endl << endl;
}
