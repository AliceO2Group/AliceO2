void load_all_libs()
{
  gSystem->Load("libALICEHLT");
  gSystem->Load("libAliceO2Base");
  gSystem->Load("libAliceO2Cdb");
  gSystem->Load("libFLP2EPNex");
  gSystem->Load("libFLP2EPNex_distributed");
  gSystem->Load("libField");
  gSystem->Load("libMathUtils");
  gSystem->Load("libO2SimulationDataFormat");
  gSystem->Load("libEvtGenerator");
  gSystem->Load("libPassive");
  gSystem->Load("libRoundtripTest");
  gSystem->Load("libITSBase");
  gSystem->Load("libSimulation");
  gSystem->Load("libTPCBase");
  gSystem->Load("libTPCSimulation");
  gSystem->Load("libo2qaLibrary");
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
}
