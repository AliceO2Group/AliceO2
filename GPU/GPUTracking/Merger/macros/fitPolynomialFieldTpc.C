int32_t fitPolynomialFieldTpc()
{
  gSystem->Load("libAliHLTTPC");
  GPUTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);
  GPUTPCGMPolynomialFieldManager::FitFieldTpc(fld, polyField, 10);
  return 0;
}
