#if !defined(__CINT__) || defined(__MAKECINT__)

#include <sstream>

#include <TStopwatch.h>

#include "FairLogger.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"

#include "MFTReconstruction/ClusterizerTask.h"

#endif

void run_clus_mft(Int_t nEvents = 1, Int_t nMuons = 100, TString mcEngine="TGeant3")
{

  FairLogger *logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  // Input file name
  char filein[100];
  sprintf(filein, "AliceO2_%s.digi_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString inFile = filein;

  // Output file name
  char fileout[100];
  sprintf(fileout, "AliceO2_%s.clus_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString outFile = fileout;

  // Parameter file name
  char filepar[100];
  sprintf(filepar, "AliceO2_%s.params_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TString parFile = filepar;

  // Setup FairRoot analysis manager
  FairRunAna * fRun = new FairRunAna();
  FairFileSource *fFileSource = new FairFileSource(inFile);
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(outFile);
  
  // Setup Runtime DB
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(parFile);
  rtdb->setFirstInput(parInput1);

  // Setup clusterizer
  o2::MFT::ClusterizerTask *clus = new o2::MFT::ClusterizerTask;
  fRun->AddTask(clus);
  
  fRun->Init();
  
  fRun->Run();

}

