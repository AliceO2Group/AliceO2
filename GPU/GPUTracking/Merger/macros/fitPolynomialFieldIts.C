
int fitPolynomialFieldIts()
{
  gSystem->Load("libAliHLTTPC.so");
  AliGPUTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);
  AliGPUTPCGMPolynomialFieldManager::FitFieldIts( fld,  polyField, 1. );
  return 0;
}
