// Use this macro to extract TRD calibration data from run2 for O2 calibrations class
// Alot of this was taken from OCDBtoTree.C in AliRoot/TRD/macros/
// Usage:
//
// void OCDB2CCDBTrapConfig(int run, const char* storageURI = "alien://folder=/alice/data/2010/OCDB/")
//
//    * run         - name of an ascii file containing run numbers
//    * outFilename       - name of the root file where the TRD OCDB information tree to be stored
//    * storageURI        - path of the OCDB database (if it is on alien, be sure to have a valid/active token)
// NOTES :
// This requires a custom version of AliTRDTrapConfig.h  all private and protected must be changed to public
//
// for running with both O2 and aliroot ...
//     export OCDB_PATH=/cvmfs/alice-ocdb.cern.ch   to use cvmfs instead of the slower pulling from alien.
//
//    .include $ALIROOT/include
//
//    build aliroot as :
//    aliBuild build AliRoot --defaults=o2
//    i have aliBuild build AliRoot --defaults=o2 -z O2 --debug
//    you then call :
//    alienv enter VO_ALICE@O2::latest-O2-o2,VO_ALICE@AliRoot::latest-O2-o2
//    according to my configs, modify as required of course.
//    AliRoot needs to be rebuilt with private and protected removed from AliTRDTrapConfig, one can not
//    extract the required information from the internals. I did not want to add additional functions.
//

#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include "TError.h"
#include "TVectorD.h"
#include "TTreeStream.h"
#include "TObjString.h"
#include "TTimeStamp.h"
#include "TH1.h"
#include "TMath.h"
#include "TObjArray.h"
#include "TFile.h"
#include "TList.h"
#include "TSystem.h"
#include "TGrid.h"
#include "AliCDBManager.h"
#include "AliCDBStorage.h"
#include "AliCDBEntry.h"
#include "AliTRDcalibDB.h"
//  #include "AliGRPObject.h"
//  #include "AliDCSSensor.h"
//  #include "AliTRDSensorArray.h"
#include "AliTRDCalDet.h"
#include "AliTRDCalPad.h"
#include "AliTRDCalROC.h"
#include "AliTRDCalPadStatus.h"
#include "AliTRDCalChamberStatus.h"
#include "AliTRDCalSingleChamberStatus.h"
#include "AliTRDCalOnlineGainTable.h"
#include "AliTRDCalOnlineGainTableROC.h"
#include "AliTRDCalOnlineGainTableMCM.h"
//#include "AliTRDCalDCS.h"
//  #include "AliTRDCalDCSFEE.h"
//  #include "AliTRDCalDCSv2.h"
//  #include "AliTRDCalDCSFEEv2.h"
#include "AliTRDCalTrapConfig.h"
#include "CCDB/CcdbApi.h"

//#include "TRDBase/PadParameters.h"
//#include "TRDBase/PadCalibrations.h"
//#include "TRDBase/PadStatus.h"
//#include "TRDBase/PadNoise.h"
//#include "TRDBase/LocalVDrift.h"
//#include "TRDBase/LocalT0.h"
//#include "TRDBase/LocalGainFactor.h"
//#include "TRDBase/PadNoise.h"
//#include "TRDBase/ChamberCalibrations.h"
//#include "TRDBase/ChamberStatus.h"
//#include "TRDBase/ChamberNoise.h"
//#include "TRDBase/CalOnlineGainTables.h"
//#include "TRDBase/FeeParam.h"
#include "TRDSimulation/TrapConfig.h"

using namespace std;
using namespace o2::ccdb;
using namespace o2::trd;

// global variables
// histograms used for extracting the mean and RMS of calibration parameters

// global constants
AliCDBStorage* storage = NULL;
AliCDBManager* manager = NULL;
Int_t Run(0);
void MakeRunListFromOCDB(const Char_t* directory, const Char_t* outfile, Bool_t fromAlien = kFALSE);
AliCDBEntry* GetCDBentry(const Char_t* path, Bool_t owner = kTRUE);

// NB NB NB NB
// This *WILL NOT WORK* unless you recompile AliTRDtrapConf with all public members in the class.
// It seems this class was never designed to allow copying..
//  I could of course redo AliRoot, but ... whats the point, as this should simply be a one off
//  to get all the trapconfig into CCDB.
//  This is the reason the following method is commented out.
void PrintTrapConfigsPerDetElement(AliTRDtrapConfig* trapconfig)
{
  //
  /*    for(int det=0;<det<540;det++){
        for(int rob=0;rob<??;rob++){
            for(int mcm=0;mcm<??;mcm++){
                //walk through all configs for a singular mcm even if they are identical for rob, det or mcm
                //this will be exhaustive but allow trivial comparison.
            }
        }
    }*/
}

void PrintDmemValue(AliTRDtrapConfig::AliTRDtrapDmemWord* trapval, ostream& output)
{
  output << "\t AllocationMode : " << trapval->fAllocMode << endl;
  output << "\t Array size : " << trapval->fSize << endl;
  for (int dataarray = 0; dataarray < trapval->fSize; dataarray++) {
    output << "\t " << trapval->fData[dataarray] << " : valid : " << trapval->fValid[dataarray] << endl;
  }
}
void PrintRegisterValue(AliTRDtrapConfig::AliTRDtrapRegister* trapval, ostream& output)
{
  output << "\t AllocationMode : " << trapval->fAllocMode << endl;
  output << "\t Array size : " << trapval->fSize << endl;
  for (int dataarray = 0; dataarray < trapval->fSize; dataarray++) {
    output << "\t " << trapval->fData[dataarray] << " : valid : " << trapval->fValid[dataarray] << endl;
  }
}

void PrintTrapConfigsAsStored(AliTRDtrapConfig* trapconfig)
{
  ofstream run2config("run2trapconfig-AsStored.txt");
  run2config << "Trap Registers : " << endl;
  for (int regvalue = 0; regvalue < AliTRDtrapConfig::kLastReg; regvalue++) {
    run2config << " Trap : " << trapconfig->fRegisterValue[regvalue].GetName()
               << " at : 0x " << hex << trapconfig->fRegisterValue[regvalue].GetAddr() << dec
               << " with nbits : " << trapconfig->fRegisterValue[regvalue].GetNbits()
               << " and reset value of : " << trapconfig->fRegisterValue[regvalue].GetResetValue() << endl;
    // now for the inherited AliTRDtrapValue members;
    PrintRegisterValue(&trapconfig->fRegisterValue[regvalue], run2config);
  }

  //  run2config << "done with regiser values now for dmemwords" << endl;
  run2config << "DMEM Words : " << endl;
  for (int dmemwords = 0; dmemwords < AliTRDtrapConfig::fgkDmemWords; dmemwords++) {
    // copy fName, fAddr
    // inherited from trapvalue : fAllocMode, fSize fData and fValid
    //        trapconfig->mDmem[dmemwords].mName= run2config->fDmem[dmemwords].fName; // this gets set on setting the address
    run2config << "Name : " << trapconfig->fDmem[dmemwords].fName << " :address : " << trapconfig->fDmem[dmemwords].fAddr << endl;
    PrintDmemValue(&trapconfig->fDmem[dmemwords], run2config);
  }
  //  ofstream run2cnofig
}

void PrintDmemValue3(TrapConfig::TrapDmemWord* trapval, ostream& output)
{
  output << "\t AllocationMode : " << trapval->getAllocMode() << endl;
  output << "\t Array size : " << trapval->getDataSize() << endl;
  for (int dataarray = 0; dataarray < trapval->getDataSize(); dataarray++) {
    output << "\t " << trapval->getDataRaw(dataarray) << " : valid : " << trapval->getValidRaw(dataarray) << endl;
  }
}
void PrintRegisterValue3(TrapConfig::TrapRegister* trapval, ostream& output)
{
  output << "\t AllocationMode : " << trapval->getAllocMode() << endl;
  output << "\t Array size : " << trapval->getDataSize() << endl;
  for (int dataarray = 0; dataarray < trapval->getDataSize(); dataarray++) {
    output << "\t " << trapval->getDataRaw(dataarray) << " : valid : " << trapval->getValidRaw(dataarray) << endl;
  }
}
void PrintTrapConfigsAsStored3(TrapConfig* trapconfig)
{
  ofstream run3config("run3trapconfig-AsStored.txt");
  run3config << "Trap Registers : " << endl;
  for (int regvalue = 0; regvalue < AliTRDtrapConfig::kLastReg; regvalue++) {
    run3config << " Trap : " << trapconfig->mRegisterValue[regvalue].getName()
               << " at : 0x " << hex << trapconfig->mRegisterValue[regvalue].getAddr() << dec
               << " with nbits : " << trapconfig->mRegisterValue[regvalue].getNbits()
               << " and reset value of : " << trapconfig->mRegisterValue[regvalue].getResetValue() << endl;
    // now for the inherited AliTRDtrapValue members;
    PrintRegisterValue3(&trapconfig->mRegisterValue[regvalue], run3config);
  }

  //  run3config << "done with regiser values now for dmemwords" << endl;
  run3config << "DMEM Words : " << endl;
  for (int dmemwords = 0; dmemwords < AliTRDtrapConfig::fgkDmemWords; dmemwords++) {
    // copy fName, fAddr
    // inherited from trapvalue : fAllocMode, fSize fData and fValid
    //        trapconfig->mDmem[dmemwords].mName= run2config->fDmem[dmemwords].fName; // this gets set on setting the address
    run3config << "Name : " << trapconfig->mDmem[dmemwords].getName() << " :address : " << trapconfig->mDmem[dmemwords].getAddress() << endl;
    PrintDmemValue3(&trapconfig->mDmem[dmemwords], run3config);
  }
}

//__________________________________________________________________________________________
void PrintTrapConfig(Int_t run, const Char_t* storageURI = "alien://folder=/alice/data/2018/OCDB/")
{
  //
  // Main function to steer the extraction of TRD OCDB information
  //

  //std::string outFilename="CalibrationsForRun"+Run;
  TTimeStamp jobStartTime;
  // if the storage is on alien than we need to do some extra stuff
  TString storageString(storageURI);
  if (storageString.Contains("alien://")) {
    TGrid::Connect("alien://");
  }

  // initialize the OCDB manager
  manager = AliCDBManager::Instance();
  manager->SetDefaultStorage(storageURI);
  manager->SetCacheFlag(kTRUE);
  storage = manager->GetDefaultStorage();
  AliCDBEntry* entry = NULL;
  Run = run;

  std::string TRDCalBase = "TRD_test";

  manager->SetRun(Run);

  time_t startTime = 0;
  time_t endTime = 0;
  UInt_t detectorMask = 0;
  // get calibration information
  // process gains
  ///////////////////////////
  //Connect to CCDB
  //
  o2::ccdb::CcdbApi ccdb;
  map<string, string> metadata;               // do we want to store any meta data?
  ccdb.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation

  /*
AliTRDCalTrapConfig has a print command that outputs the list of trapconfigs stored.
AliTRDCalTrapConfig->Print() .... and you get the stuff below via AliInfo, but I can seem to get it right to parse this so
its all going here unfortunately ....

*/

  vector<std::string> run2confignames = {
    /*   "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5505",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2p-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5585",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5766",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b0-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en_ptrg.r5767",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5570",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b2p-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5566",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv2en-pt100_ptrg.r5764",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b0-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en_ptrg.r5766",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv2en-pt100_ptrg.r5764",
    "cf_p_nozs_tb30_trk_ptrg.r4850",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv2en-pt100_ptrg.r5764",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b0-fs1e24-ht200-qs0e24s24e23-pidlinear_ptrg.r5549",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b00-fs1e30-ht200-qs0e23s23e22-nb233-pidlhc11dv3en_ptrg.r5772",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b2p-fs1e30-ht200-qs0e29s29e28-nb233-pidlhc11dv4en-pt100_ptrg.r5773",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b2p-fs1e30-ht200-qs0e23s23e22-nb233-pidlhc11dv3en-pt100_ptrg.r5772",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1-pt100_ptrg.r5762",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e30-ht200-qs0e23s23e22-nb233-pidlhc11dv3en-pt100_ptrg.r5772",
    "cf_pg-fpnp32_zs-s16-deh_tb26_trkl-b5p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5771",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5p-fs1e30-ht200-qs0e23s23e22-nb233-pidlhc11dv3en-pt100_ptrg.r5772",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b0-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en_ptrg.r5765",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5585",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5p-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5505",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5767",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5p-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b0-fs1e24-ht200-qs0e23s23e22-pidlhc11dv2en_ptrg.r5764",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5p-fs1e30-ht200-qs0e29s29e28-nb233-pidlhc11dv4en-pt100_ptrg.r5773",
    "cf_pg-fpnp32_zs-s16-deh_tb26_trkl-b0-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en_ptrg.r5771",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b2n-fs1e30-ht200-qs0e23s23e22-nb233-pidlhc11dv3en-pt100_ptrg.r5772",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1-pt100_ptrg.r5762",
    "cf_pg-fpnp32_zs-s16-deh_tb26_trkl-b2p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5771",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5766",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1-pt100_ptrg.r4946",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5765",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1-pt100_ptrg.r5762",
    "cf_pg-fpnp32_zs-s16-deh_tb22_trkl-b5p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1-pt100_ptrg.r5037",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5767",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2p-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5767",
    "cf_pg-fpnp32_zs-s16-deh_tb22_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1-pt100_ptrg.r5037",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv2en-pt100_ptrg.r5764",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5765",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5p-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5570",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b0-fs1e30-ht200-qs0e29s29e28-nb233-pidlhc11dv4en_ptrg.r5773",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549",
    "cf_pg-fpnp32_zs-s16-deh_tb26_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5771",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e30-ht200-qs0e29s29e28-nb233-pidlhc11dv4en-pt100_ptrg.r5773",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b0-fs1e24-ht200-qs0e24s24e23-pidlinear_ptrg.r5570",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b2n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5566",
    "cf_pg-fpnp32_zs-s16-deh_tb22_trkl-b0-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1_ptrg.r5037",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b2n-fs1e30-ht200-qs0e29s29e28-nb233-pidlhc11dv4en-pt100_ptrg.r5773",
    "cf_pg-fpnp32_zs-s16-deh_tb22_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1hn-pt100_ptrg.r5151",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5767",
    "cf_pg-fpnp32_zs-s16-deh_tb26_trkl-b2n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv3en-pt100_ptrg.r5771",
    "cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b5n-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1-pt100_ptrg.r5762",
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b0-fs1e24-ht200-qs0e24s24e23-pidlinear_ptrg.r5505",*/
    //"cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b0-fs1e24-ht200-qs0e23s23e22-pidlhc11dv1_ptrg.r5762"
    "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549"};

  // now we loop over these extracting the trapconfing and dumping it into the ccdb.
  AliTRDCalTrapConfig* run2caltrapconfig = 0;
  AliTRDtrapConfig* run2trapconfig = 0;
  if ((entry = GetCDBentry("TRD/Calib/TrapConfig", 0))) {
    if ((run2caltrapconfig = (AliTRDCalTrapConfig*)entry->GetObject())) {
      for (auto const& run2trapconfigname : run2confignames) {
        auto o2trapconfig = new TrapConfig(run2trapconfigname);
        run2trapconfig = run2caltrapconfig->Get(run2trapconfigname.c_str());
        PrintTrapConfigsAsStored(run2trapconfig);
        auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
        ccdbmgr.setTimestamp(297595);
        // auto run3trapconfig = ccdbmgr.get<o2::trd::TrapConfig>("TRD_test/TrapConfig2020/cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549");
        auto run3trapconfig = ccdbmgr.get<o2::trd::TrapConfig>("TRD_test/TrapConfig2020/c");
        if (run3trapconfig)
          PrintTrapConfigsAsStored3(run3trapconfig);
        //     ccdb.storeAsTFileAny(o2trapconfig, Form("%s/TrapConfig2020/%s", TRDCalBase.c_str(), run2trapconfigname.c_str()), metadata, 1, 1670700184549); //upper time chosen into the future else the server simply adds a year
        //     cout << "ccdb.storeAsTFileAny(o2trapconfig, Form(\"" << TRDCalBase.c_str() << "/TrapConfig/" << run2trapconfigname.c_str()<< endl; //upper time chosen into the future else the server simply adds a year
        //AliTRDcalibDB *calibdb=AliTRDcalibDB::Instance();
      }
    }
  }

  //const AliTRDCalTrapConfig *caltrap = dynamic_cast<const AliTRDCalTrapConfig*> (calibdb->GetCachedCDBObject(12));

  //cout << "now to print the names of the cal traps" << endl;
  //if(caltrap) caltrap->Print();
  //else cout << "caltrap is null" << endl;
  //RecoParam

  /*
THESE ARE THE ONES NOT CURRENTLY INCLUDED.
trd_ddchamberStatus
trd_gaschromatographXe
trd_gasOverpressure
trd_hvDriftImon
MonitoringData
PIDLQ
trd_envTemp
trd_gasCO2
trd_hvDriftUmon
PIDLQ1D
trd_gaschromatographCO2
trd_gasH2O
trd_hvAnodeImon
TrkAttach
PIDNN
PHQ      PIDThresholds

This pulls stuff from DCS I should hopefully not need this stuff for simulation.
DCS
AliTRDSensorArray descends from AliTRDDCSSensorArray
trd_gaschromatographN2
trd_gasO2
trd_goofieHv
trd_goofiePressure
trd_hvAnodeUmon
trd_goofieN2
trd_goofieTemp
trd_goofieCO2
trd_goofiePeakArea
trd_goofieVelocity
trd_goofieGain
trd_goofiePeakPos
*/
  return;
}

//__________________________________________________________________________________________
AliCDBEntry* GetCDBentry(const Char_t* path, Bool_t owner)
{
  TString spath = path;
  //  ::Info("GetCDBentry", Form("QUERY RUN [%d] for \"%s\".", Run, spath.Data()));
  AliCDBEntry* entry(NULL);
  storage->QueryCDB(Run, spath.Data());
  cout << spath.Data();
  if (!storage->GetQueryCDBList()->GetEntries()) {
    cout << "GetCDBentry" << Form("Missing \"%s\" in run %d.", spath.Data(), Run);
    return NULL;
  } else
    entry = manager->Get(spath.Data());
  if (!entry)
    return NULL;

  entry->SetOwner(owner);
  //  ::Info("GetCDBentry", Form("FOUND ENTRY @ [%p].", (void*)entry));
  return entry;
}
