void load_all_libs()
{
  gSystem->Load("libaliceHLTwrapper");
  gSystem->Load("libCCDB");
  gSystem->Load("libDetectorsBase");
  gSystem->Load("libDetectorsPassive");
  gSystem->Load("libExampleModule1");
  gSystem->Load("libExampleModule2");
  gSystem->Load("libField");
  gSystem->Load("libFLP2EPNex_distributed");
  gSystem->Load("libflp2epn");
  gSystem->Load("libGenerators");
  gSystem->Load("libHitAnalysis");
  gSystem->Load("libITSBase");
  gSystem->Load("libITSSimulation");
  gSystem->Load("libMathUtils");
  gSystem->Load("libQCMerger");
  gSystem->Load("libQCProducer");
  gSystem->Load("libQCViewer");
  gSystem->Load("libSimulationDataFormat");
  gSystem->Load("libTPCBase");
  gSystem->Load("libTPCSimulation");
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
}
