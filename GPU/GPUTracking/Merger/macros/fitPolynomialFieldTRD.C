
int fitPolynomialFieldTRD()
{
  gSystem->Load("libAliHLTTPC.so");
  AliGPUTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);
  AliGPUTPCGMPolynomialFieldManager::FitFieldTRD( fld,  polyField, 2 );
  return 0;
}
