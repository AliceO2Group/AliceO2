
int fitPolynomialFieldTRD()
{
  gSystem->Load("libAliHLTTPC.so");
  AliHLTTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);
  AliHLTTPCGMPolynomialFieldManager::FitFieldTRD( fld,  polyField, 2 );
  return 0;
}
