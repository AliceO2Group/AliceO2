int32_t fitPolynomialFieldTrd()
{
  gSystem->Load("libAliHLTTPC");
  GPUTPCGMPolynomialField polyField;
  AliMagF* fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k5kG);
  GPUTPCGMPolynomialFieldManager::FitFieldTrd(fld, polyField, 2);
  return 0;
}
