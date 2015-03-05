const char* defaultCDBUri="local://./OCDB";
void runHLT(const char* chain, int events=1, int runno=-1, const char* cdbURI=defaultCDBUri)
{
  // setup the OCDB access
  // required to load the GRP entry in order to initialize the magnetic field
  if (runno>=0) {
    AliCDBManager::Instance()->SetDefaultStorage(cdbURI);
    AliCDBManager::Instance()->SetRun(runno);
    AliGRPManager grpman;
    grpman.ReadGRPEntry();
    grpman.SetMagField();
  }

  // init the HLT system
  AliHLTSystem* pHLT=AliHLTPluginBase::GetInstance();

  ///////////////////////////////////////////////////////////////////////////////////////////
  //
  // list of configurations
  //
  ///////////////////////////////////////////////////////////////////////////////////////////

  if (chain) {
    pHLT->ScanOptions("loglevel=0x7c");
    //pHLT->ScanOptions("loglevel=0x7f frameworklog=0x7f");
    pHLT->BuildTaskList(chain);
    pHLT->Run(events);
  }
}
