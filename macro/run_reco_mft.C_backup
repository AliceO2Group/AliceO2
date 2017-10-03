void run_reco_mft(Int_t nEvents = 1, Int_t nMuons = 100, TString mcEngine="TGeant3" )
{

  FairLogger *logger = FairLogger::GetLogger();
  //logger->SetLogFileName("MyLog.log");
  logger->SetLogToScreen(kTRUE);
  //logger->SetLogToFile(kTRUE);
  //logger->SetLogVerbosityLevel("HIGH");
  //logger->SetLogFileLevel("DEBUG4");
  logger->SetLogScreenLevel("INFO");

  // Verbosity level (0=quiet, 1=event level, 2=track level, 3=debug)
  Int_t iVerbose = 0; // just forget about it, for the moment

  // Input file name
  char filein[100];
  sprintf(filein, "AliceO2_%s.mc_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString inFile = filein;

  // Output file name
  char fileout[100];
  sprintf(fileout, "AliceO2_%s.reco_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString outFile = fileout;

  // Parameter file name
  char filepar[100];
  sprintf(filepar, "AliceO2_%s.params_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString parFile = filepar;

  // -----   Timer   --------------------------------------------------------
  TStopwatch timer;
  
  // -----   Reconstruction run   -------------------------------------------
  FairRunAna *fRun= new FairRunAna();
  fRun->SetInputFile(inFile);
  fRun->SetOutputFile(outFile);

  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput = new FairParRootFileIo();
  parInput->open(parFile.Data());
  rtdb->setFirstInput(parInput);
  rtdb->setOutput(parInput);
  rtdb->saveOutput();
  rtdb->print();

  o2::MFT::FindHits* fhi = new o2::MFT::FindHits();
  fRun->AddTask(fhi);

  o2::MFT::FindTracks* ftr = new o2::MFT::FindTracks();
  fRun->AddTask(ftr);

  fRun->Init();

  o2::field::MagneticField* fld = (o2::field::MagneticField*)fRun->GetField();
  if (!fld) {
    std::cout << "Failed to get field instance from FairRunAna" << std::endl;
    return;
  }
  printf("Field solenoid = %f [kG] \n",fld->solenoidField());
  printf("Field Bx,By,Bz in (0,0,0) = %f %f %f [kG] \n",
	 fld->GetBx(0.,0.,0.),
	 fld->GetBy(0.,0.,0.),
	 fld->GetBz(0.,0.,0.));

  timer.Start();

  fRun->Run();

  timer.Stop();

  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

}

