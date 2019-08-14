#if defined(__CLING__) && !defined(__ROOTCLING__)
void rootlogon()
{
  gROOT->Macro("load_all_libs.C");
}
#endif
