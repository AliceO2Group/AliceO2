void load_all_libs()
{
   gSystem->Load("libDetectorsBase");
   gSystem->Load("libDetectorsPassive");
   gSystem->Load("libExampleModule1");
   gSystem->Load("libExampleModule2");
   gSystem->Load("libFLP2EPNex_distributed");
   gSystem->Load("libField");
   gSystem->Load("libGenerators");
   gSystem->Load("libHeaders");
   gSystem->Load("libHitAnalysis");
   gSystem->Load("libITSBase");
   gSystem->Load("libITSReconstruction");
   gSystem->Load("libITSSimulation");
   gSystem->Load("libMFTBase");
   gSystem->Load("libMFTReconstruction");
   gSystem->Load("libMFTSimulation");
   gSystem->Load("libITSMFTBase");
   gSystem->Load("libITSMFTReconstruction");
   gSystem->Load("libITSMFTSimulation");
   gSystem->Load("libMathUtils");
   gSystem->Load("libO2Device");
   gSystem->Load("libQCMerger");
   gSystem->Load("libQCMetricsExtractor");
   gSystem->Load("libQCProducer");
   gSystem->Load("libQCViewer");
   gSystem->Load("libSimulationDataFormat");
   gSystem->Load("libTPCBase");
   gSystem->Load("libTPCSimulation");
   gSystem->Load("libaliceHLTwrapper");
   gSystem->Load("libflp2epn");
   cout << endl << endl;
   cout << "Macro finished succesfully." << endl;
 }
