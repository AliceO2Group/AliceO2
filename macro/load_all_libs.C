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
  gSystem->Load("libits");
  gSystem->Load("libitsBase");
  gSystem->Load("libSimulation");
  gSystem->Load("libtpcBase");
  gSystem->Load("libtpcSimulation");
  gSystem->Load("libo2qaLibrary");
  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
}
