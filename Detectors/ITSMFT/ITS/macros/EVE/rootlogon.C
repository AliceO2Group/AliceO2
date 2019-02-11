{
  std::cout << "\n *** This is ITS visualisation ! ***\n\n";
  gSystem->Load("libEventVisualisationView");
  gROOT->LoadMacro("./DisplayEvents.C+");
  init();
  gEve->GetBrowser()->GetTabRight()->SetTab(1);
}
