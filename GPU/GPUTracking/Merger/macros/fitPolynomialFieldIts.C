int fitPolynomialFieldIts()
{
  gSystem->Load("libAliHLTTPC.so");
  GPUTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);
  GPUTPCGMPolynomialFieldManager::FitFieldIts(fld, polyField, 1.);
  return 0;
}
