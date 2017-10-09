#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <sstream>

#include <TStopwatch.h>

#include "FairLogger.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"

#include "MFTReconstruction/ClustererTask.h"

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

  // Setup timer
  TStopwatch timer;

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
  o2::MFT::ClustererTask *clus = new o2::MFT::ClustererTask;
  fRun->AddTask(clus);
  
  fRun->Init();
  
  fRun->Run();

  std::cout << std::endl << std::endl;
  
  // Extract the maximal used memory an add is as Dart measurement
  // This line is filtered by CTest and the value send to CDash
  FairSystemInfo sysInfo;
  Float_t maxMemory=sysInfo.GetMaxMemory();
  std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage=ctime/rtime;
  cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
  cout << cpuUsage;
  cout << "</DartMeasurement>" << endl;
  cout << endl << endl;
  std::cout << "Macro finished succesfully" << std::endl;
  
  std::cout << endl << std::endl;
  std::cout << "Output file is "    << fileout << std::endl;
  //std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime
	    << "s" << endl << endl;

}

