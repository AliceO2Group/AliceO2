double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
{
  // compute turbo angle from radii and sensor width
  return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
}

void run_sim(Int_t nEvents = 10, TString mcEngine = "TGeant3")
{
  // Output file name
  const char fileout[100];
  sprintf(fileout, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TString outFile = fileout;

  // Parameter file name
  const char filepar[100];
  sprintf(filepar, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TString parFile = filepar;

  // In general, the following parts need not be touched

  // Debug option
  gDebug = 0;

  // Timer
  TStopwatch timer;
  timer.Start();

  gSystem->Load("libAliceO2Base");
  gSystem->Load("libAliceO2its");

  // Create simulation run
  FairRunSim* run = new FairRunSim();
  run->SetName(mcEngine);      // Transport engine
  run->SetOutputFile(outFile); // Output file
  FairRuntimeDb* rtdb = run->GetRuntimeDb();

  // Create media
  run->SetMaterials("media.geo"); // Materials

  // Create geometry
  FairModule* cave = new AliCave("CAVE");
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  //  FairDetector*  tpc = new O2tpc("TPCV2");
  //  tpc->SetGeometry();
  //  run->AddModule(tpc);

  TGeoGlobalMagField::Instance()->SetField(new AliceO2::Field::MagneticField("Maps","Maps", -1., -1., AliceO2::Field::MagneticField::k5kG));

  AliceO2::Base::Detector* its = new AliceO2::ITS::Detector("ITS", kTRUE, 7);
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
  AliceO2::ITS::UpgradeSegmentationPixel* seg0 = new AliceO2::ITS::UpgradeSegmentationPixel(
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
    );           // see UpgradeSegmentationPixel.h for extra options
  seg0->Store(AliceO2::ITS::UpgradeGeometryTGeo::getITSsegmentationFileName());
  seg0->Print();

  double dzLr, rLr, phi0, turbo;
  int nStaveLr, nModPerStaveLr, idLr;

  its->setStaveModelIB(AliceO2::ITS::Detector::kIBModel22);
  its->setStaveModelOB(AliceO2::ITS::Detector::kOBModel1);

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
  FairBoxGenerator* boxGen = new FairBoxGenerator(2212, 1); /*protons*/

  boxGen->SetThetaRange(0.0, 90.0);
  boxGen->SetPRange(100, 100.01);
  boxGen->SetPhiRange(0., 360.);
  boxGen->SetDebug(kTRUE);

  primGen->AddGenerator(boxGen);

  run->SetGenerator(primGen);

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
  run->CreateGeometryFile("geofile_full.root");

  // Finish
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
  cout << "Output file is " << outFile << endl;
  cout << "Parameter file is " << parFile << endl;
  cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl << endl;
}
