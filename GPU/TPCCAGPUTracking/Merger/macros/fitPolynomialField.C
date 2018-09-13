
int fitPolynomialField()
{
  gSystem->Load("libAliHLTTPC.so");
  AliHLTTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k2kG);
  AliHLTTPCGMPolynomialFieldManager::FitFieldTPC( fld,  polyField );
  return 0;
}
