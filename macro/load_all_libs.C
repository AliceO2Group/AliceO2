void load_all_libs()
{
  gSystem->Load("libALICEHLT");
  gSystem->Load("libAliceO2Base");
  gSystem->Load("libAliceO2Cdb");
  gSystem->Load("libFLP2EPNex");
  gSystem->Load("libFLP2EPNex_distributed");
  gSystem->Load("libFLP2EPNex_dynamic");
  gSystem->Load("libField");
  gSystem->Load("libMathUtils");
  gSystem->Load("libO2Data");
  gSystem->Load("libO2Gen");
  gSystem->Load("libPassive");
  gSystem->Load("libRoundtripTest");
  gSystem->Load("libits");
  gSystem->Load("libtpc");

  cout << endl << endl;
  cout << "Macro finished succesfully." << endl;
}
