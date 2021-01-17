#if defined(__CLING__) && !defined(__ROOTCLING__)
void rootlogon()
{
  gSystem->Load("libMCHSimulation");
  gSystem->SetIncludePath(gSystem->Getenv("ROOT_INCLUDE_PATH"));
}
#endif
