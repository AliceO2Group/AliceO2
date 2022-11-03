#include <fairlogger/Logger.h>
#include "TStopwatch.h"
#include "EMCALReconstruction/ClusterizerParameters.h"
#include "EMCALReconstruction/ClusterizerTask.h"

void run_clus_emcal(std::string outputfile = "EMCALClusters.root", std::string inputfile = "Data.root")
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  TStopwatch timer;
  //o2::base::GeometryManager::loadGeometry(); // needed provisionary, only to write full clusters

  // Setup clusterizer
  o2::emcal::ClusterizerParameters parameters(10000, 0, 10000, true, 0.03, 0.1, 0.05);
  o2::emcal::ClusterizerTask<o2::emcal::Digit>* clus = new o2::emcal::ClusterizerTask<o2::emcal::Digit>(&parameters);

  clus->process(inputfile, outputfile);

  timer.Stop();
  timer.Print();
}
