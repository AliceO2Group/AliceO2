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

#include "ITSReconstruction/TrivialVertexer.h"
#include "ITSReconstruction/CookedTrackerTask.h"
#endif

void run_trac_its(float rate = 0., std::string outputfile = "o2trac_its.root", std::string inputfile = "o2clus_its.root", std::string mcfile = "o2sim.root", std::string paramfile = "o2sim_par.root")
{
  // Initialize logger
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
  o2::ITS::CookedTrackerTask* trac = new o2::ITS::CookedTrackerTask(n, mcTruth);
  trac->setContinuousMode(rate > 0.);

  o2::ITS::TrivialVertexer& vertexer = trac->getVertexer();
  //vertexer.openInputFile(mcfile.str().c_str());
  vertexer.openInputFile(mcfile.data());

  fRun->AddTask(trac);

  fRun->Init();

  o2::field::MagneticField* fld = (o2::field::MagneticField*)fRun->GetField();
  if (!fld) {
    std::cout << "Failed to get field instance from FairRunAna" << std::endl;
    return;
  }
  trac->setBz(fld->solenoidField()); // in kG

  timer.Start();
  fRun->Run();

  std::cout << std::endl << std::endl;

  // Extract the maximal used memory an add is as Dart measurement
  // This line is filtered by CTest and the value send to CDash
  FairSystemInfo sysInfo;
  Float_t maxMemory = sysInfo.GetMaxMemory();
  std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage = ctime / rtime;
  cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
  cout << cpuUsage;
  cout << "</DartMeasurement>" << endl;
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;

  std::cout << endl << std::endl;
  std::cout << "Output file is " << outputfile.data() << std::endl;
  // std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl << endl;
}

