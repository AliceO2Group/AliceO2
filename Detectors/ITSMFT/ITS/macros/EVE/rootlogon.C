{
  std::cout << "\n *** This is ITS visualisation ! ***\n\n";
  gROOT->LoadMacro("./DisplayEvents.C+");
  init();
  gEve->GetBrowser()->GetTabRight()->SetTab(1);
}
