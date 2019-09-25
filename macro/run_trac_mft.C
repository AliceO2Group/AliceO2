#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <sstream>

#include <TStopwatch.h>

#include "FairLogger.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include "Field/MagneticField.h"

#include "MFTReconstruction/TrackerTask.h"

#endif

static std::stringstream mcfile;

void run_trac_mft(float rate = 0., std::string outputfile = "o2track_mft.root", std::string inputfile = "o2clus.root",
                  std::string paramfile = "o2sim_par.root");

void run_trac_mft(Int_t nEvents = 10, Int_t nMuons = 100, TString mcEngine = "TGeant3", Float_t rate = 0.)
{

  std::stringstream inputfile, outputfile, paramfile;
  inputfile << "AliceO2_" << mcEngine << ".clus_" << nEvents << "ev_" << nMuons << "mu.root";
  paramfile << "AliceO2_" << mcEngine << ".params_" << nEvents << "ev_" << nMuons << "mu.root";
  outputfile << "AliceO2_" << mcEngine << ".trac_" << nEvents << "ev_" << nMuons << "mu.root";
  mcfile << "AliceO2_" << mcEngine << ".mc_" << nEvents << "ev_" << nMuons << "mu.root";
  run_trac_mft(rate, outputfile.str(), inputfile.str(), paramfile.str());
}

void run_trac_mft(float rate, std::string outputfile, std::string inputfile, std::string paramfile)
{

  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  // Setup timer
  TStopwatch timer;

  // Setup FairRoot analysis manager
  FairRunAna* fRun = new FairRunAna();
  FairFileSource* fFileSource = new FairFileSource(inputfile.data());
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(outputfile.data());

  // Setup Runtime DB
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.data());
  rtdb->setFirstInput(parInput1);

  // Setup tracker
  Int_t n = 1;            // Number of threads
  Bool_t mcTruth = kTRUE; // kFALSE if no comparison with MC is needed
  o2::mft::TrackerTask* trac = new o2::mft::TrackerTask(n, mcTruth);
  trac->setContinuousMode(rate > 0.);

  fRun->AddTask(trac);

  fRun->Init();

  timer.Start();

  fRun->Run();

  timer.Stop();

  std::cout << std::endl
            << std::endl;

  // Extract the maximal used memory an add is as Dart measurement
  // This line is filtered by CTest and the value send to CDash
  FairSystemInfo sysInfo;
  Float_t maxMemory = sysInfo.GetMaxMemory();
  std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage = ctime / rtime;
  cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
  cout << cpuUsage;
  cout << "</DartMeasurement>" << endl;
  cout << endl
       << endl;
  std::cout << "Macro finished succesfully" << std::endl;

  std::cout << endl
            << std::endl;
  std::cout << "Output file is " << outputfile.data() << std::endl;
  std::cout << "Parameter file is " << paramfile.data() << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl
            << endl;
}
