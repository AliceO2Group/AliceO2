void run_clus_emcal(std::string outputfile = "EMCALClusters.root", std::string inputfile)
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  TStopwatch timer;
  //o2::base::GeometryManager::loadGeometry(); // needed provisionary, only to write full clusters

  // Setup clusterizer
  o2::emcal::ClusterizerParameters parameters(10000, 0, 10000, true, 0.03, 0.1, 0.05);
  o2::emcal::ClusterizerTask* clus = new o2::emcal::ClusterizerTask(&parameters);

  clus->process(inputfile, outputfile);

  timer.Stop();
  timer.Print();
}
