
int fitPolynomialFieldTpc()
{
  gSystem->Load("libAliHLTTPC.so");
  AliGPUTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);
  AliGPUTPCGMPolynomialFieldManager::FitFieldTpc( fld,  polyField,10 );
  return 0;
}
