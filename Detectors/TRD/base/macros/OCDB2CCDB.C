// Use this macro to extract TRD calibration data from run2 for O2 calibrations class
// Alot of this was taken from OCDBtoTree.C in AliRoot/TRD/macros/
// Usage:
//
// void DumpOCDBToCalibrations(const Char_t* outFilename,
// void OCDB2CCDB(int run, const char* storageURI = "alien://folder=/alice/data/2010/OCDB/")
//
//    * run         - name of an ascii file containing run numbers
//    * outFilename       - name of the root file where the TRD OCDB information tree to be stored
//    * firstRun, lastRun - lowest and highest run numbers (from the ascii file) to be dumped
//                          if these numbers are not specified (-1) all run numbers in the input ascii file will
//                          be used. If the run list file is not specified then all runs in this interval
//                          will be queried
//    * storageURI        - path of the OCDB database (if it is on alien, be sure to have a valid/active token)
//
// for running with both O2 and aliroot ...
//   export OCDB_PATH=/cvmfs/alice-ocdb.cern.ch   to use cvmfs instead of the slower pulling from alien.
//
//    .include $ALIROOT/include
//
//    build aliroot as :
//    aliBuild build AliRoot --defaults=o2
//    i have aliBuild build AliRoot --defaults=o2 -z O2 --debug
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
#include "TRDBase/PadParameters.h"
#include "TRDBase/PadCalibrations.h"
#include "TRDBase/PadStatus.h"
#include "TRDBase/PadNoise.h"
#include "TRDBase/LocalVDrift.h"
#include "TRDBase/LocalT0.h"
#include "TRDBase/LocalGainFactor.h"
#include "TRDBase/PadNoise.h"
#include "TRDBase/ChamberCalibrations.h"
#include "TRDBase/ChamberStatus.h"
#include "TRDBase/ChamberNoise.h"
#include "TRDBase/CalOnlineGainTables.h"
#include "TRDBase/FeeParam.h"

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
void PrintCalROC(ofstream& out, const AliTRDCalROC* calroc);
void PrintCalPad(ofstream& out, const AliTRDCalPad* calpad);
void PrintCalSingleChamberStatus(ofstream& out, const AliTRDCalSingleChamberStatus* chamberstatus);
void PrintCalPadStatus(ofstream& out, const AliTRDCalPadStatus* padstatus);
void PrintParam(ofstream& out, AliTRDCalChamberStatus& a);
void PrintCalDet(ofstream& out, const AliTRDCalDet* caldet);

void PrintChamberStatus(ofstream& out, AliTRDCalChamberStatus* a)
{
  for (int i = 0; i < 540; i++)
    out << (int)(a->GetStatus(i)) << endl;
}

void PrintCalDet(ofstream& out, const AliTRDCalDet* caldet)
{

  for (int i = 0; i < AliTRDCalDet::kNdet; i++)
    out << caldet->GetValue(i) << endl;
}

void PrintCalROC(ofstream& out, const AliTRDCalROC* calroc)
{
  //cout << calroc->fPla;
  //cout << calroc->getChamber();
  int rows = calroc->GetNrows();
  int cols = calroc->GetNcols();
  out << rows << endl;
  out << cols << endl;

  for (int i = 0; i < rows * cols; i++) {
    out << calroc->GetValue(i) << endl;
  }
}

void PrintCalPad(ofstream& out, const AliTRDCalPad* calpad)
{
  for (int i = 0; i < AliTRDCalDet::kNdet; i++)
    PrintCalROC(out, calpad->GetCalROC(i));
}

void PrintCalSingleChamberStatus(ofstream& out, AliTRDCalSingleChamberStatus* chamberstatus)
{
  //cout << calroc.fPla;
  //cout << calroc.fCha;
  out << chamberstatus->GetNrows();
  out << chamberstatus->GetNcols();

  for (int i = 0; i < chamberstatus->GetNrows() * chamberstatus->GetNcols(); i++)
    out << (int)(chamberstatus->GetStatus(i)) << endl;
}

void PrintCalPadStatus(ofstream& out, const AliTRDCalPadStatus* padstatus)
{
  for (int i = 0; i < AliTRDCalDet::kNdet; i++) {
    int stack = AliTRDgeometry::GetStack(i);
    int plane = AliTRDgeometry::GetLayer(i);
    int sector = AliTRDgeometry::GetSector(i);
    AliTRDCalSingleChamberStatus* singlechamberstat = padstatus->GetCalROC(plane, stack, sector);
    PrintCalSingleChamberStatus(out, singlechamberstat);
  }
}

void PrintGainTable(ofstream& out, const AliTRDCalOnlineGainTable* tbl)
{
  AliTRDCalOnlineGainTableROC* tblroc = 0;
  AliTRDCalOnlineGainTableMCM* tblmcm = 0;

  for (int i = 9; i < 540; i++) {
    tblroc = tbl->GetGainTableROC(i);
    if (tblroc) {
      out << i << endl;
      for (int j = 0; j < 128; j++) {
        tblmcm = tblroc->GetGainTableMCM(j);
        if (tblmcm) {
          out << j << endl;
          out << tblmcm->GetAdcdac() << endl;
          ;
          out << tblmcm->GetMCMGain() << endl;
          ;
          for (int k = 0; k < 21; k++)
            out << tblmcm->GetFGFN(k) << endl;
          for (int k = 0; k < 21; k++)
            out << tblmcm->GetFGAN(k) << endl;
        } else
          cout << "tblmcm is null" << endl;
      }
    } else
      cout << "tblroc is null" << endl;
  }
}

void UnpackGainTable(std::string& gainkey, CalOnlineGainTables* gtbl)
{
  AliTRDCalOnlineGainTable* tbl = 0;
  AliTRDCalOnlineGainTableROC* tblroc = 0;
  AliTRDCalOnlineGainTableMCM* tblmcm = 0;
  AliCDBEntry* entry = NULL;
  if ((entry = GetCDBentry(Form("TRD/Calib/%s", gainkey.c_str()), 0))) {
    tbl = (AliTRDCalOnlineGainTable*)entry->GetObject();
    for (int i = 9; i < 540; i++) {
      tblroc = tbl->GetGainTableROC(i);
      if (tblroc) {
        for (int j = 0; j < 128; j++) {
          tblmcm = tblroc->GetGainTableMCM(j);
          if (tblmcm) {
            gtbl->setAdcdac(i * 128 + j, tblmcm->GetAdcdac());
            gtbl->setMCMGain(i * 128 + j, tblmcm->GetMCMGain());
            for (int k = 0; k < 21; k++) {
              gtbl->setFGFN(i * 128 + j, k, tblmcm->GetFGFN(k));
              gtbl->setFGAN(i * 128 + j, k, tblmcm->GetFGAN(k));
            }
          } else
            cout << "tblmcm is null" << endl;
        }
      } else
        cout << "tblroc is null" << endl;
    }
  }
}

//__________________________________________________________________________________________
void OCDB2CCDB(Int_t run, const Char_t* storageURI = "alien://folder=/alice/data/2010/OCDB/")
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

  AliTRDCalChamberStatus* chamberStatus = 0;
  AliTRDCalDet *chamberExB = 0, *chamberVDrift = 0, *chamberGainFactor = 0, *chamberT0 = 0;
  AliTRDCalPad* LocalT0 = 0;
  Float_t runBadPadFraction = 0.0;

  if ((entry = GetCDBentry("TRD/Calib/ChamberStatus", 0))) {
    if ((chamberStatus = (AliTRDCalChamberStatus*)entry->GetObject())) {
      auto o2chamberstatus = new o2::trd::ChamberStatus();
      for (int i = 0; i < 540; i++) {
        char a = chamberStatus->GetStatus(i);
        o2chamberstatus->setRawStatus(i, a);
      }
      // store abitrary user object in strongly typed manner
      ccdb.storeAsTFileAny(o2chamberstatus, "TRD_test/ChamberStatus", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
      // // read like this (you have to specify the type)
      //auto o2chamberstatusback = ccdb.retrieveFromTFileAny<o2::trd::ChamberStatus>("TRD_test/ChamberStatus", metadata);
    } else
      cout << "attempt to get object chamber status from ocdb entry. Ergo not writing one to ccdb." << endl;
  } else
    cout << "failed to retrieve ocdb entry for Chamber Status" << endl;

  //Now for ChamberCalibrations which holds 4 old values of ChamberVDrift, ChamberT0,
  //ChamberGainFactor and ChamberExB
  auto o2chambercalibrations = new o2::trd::ChamberCalibrations();
  if ((entry = GetCDBentry("TRD/Calib/ChamberVdrift", 0))) {
    if ((chamberVDrift = (AliTRDCalDet*)entry->GetObject())) {
      for (int i = 0; i < 540; i++) {
        o2chambercalibrations->setVDrift(i, chamberVDrift->GetValue(i));
      }
    } else
      cout << "attempt to get object chamber Vdrift from ocdb entry. Will not be writing ChamberSettings" << endl;
  } else
    cout << "failed to retrieve ocdb entry for Chamber VDrift" << endl;

  if ((entry = GetCDBentry("TRD/Calib/ChamberT0", 0))) {
    if ((chamberT0 = (AliTRDCalDet*)entry->GetObject())) {
      for (int i = 0; i < 540; i++)
        o2chambercalibrations->setT0(i, chamberT0->GetValue(i));
    } else
      cout << "attempt to get object chamber T0 from ocdb entry. Will not be writing ChamberSettings" << endl;
  } else
    cout << "failed to retrieve ocdb entry for Chamber T0" << endl;

  if ((entry = GetCDBentry("TRD/Calib/ChamberExB", 0))) {
    if ((chamberExB = (AliTRDCalDet*)entry->GetObject())) {
      for (int i = 0; i < 540; i++)
        o2chambercalibrations->setExB(i, chamberExB->GetValue(i));
    } else
      cout << "attempt to get object chamber T0 from ocdb entry. Will not be writing ChamberSettings" << endl;
  } else
    cout << "failed to retrieve ocdb entry for Chamber T0" << endl;
  if ((entry = GetCDBentry("TRD/Calib/ChamberGainFactor", 0))) {
    if ((chamberGainFactor = (AliTRDCalDet*)entry->GetObject())) {
      for (int i = 0; i < 540; i++)
        o2chambercalibrations->setGainFactor(i, chamberGainFactor->GetValue(i));
    } else
      cout << "attempt to get object chamber GainFactro from ocdb entry. Will not be writing ChamberSettings" << endl;
  } else
    cout << "failed to retrieve ocdb entry for Chamber GainFactor" << endl;

  if (chamberGainFactor && chamberExB && chamberT0 && chamberVDrift) {
    //if all 4 mmebers of calibrations is here then write to ccdb.
    ccdb.storeAsTFileAny(o2chambercalibrations, "TRD_test/ChamberCalibrations", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
  } else
    cout << "something wrong with one of the members of ChamberCalibrations and not writing to ccdb, fix :: " << chamberGainFactor << "&&" << chamberExB << "&&" << chamberT0 << "&&" << chamberVDrift << endl;

  AliTRDCalPad* localvdrift = 0;
  auto o2localvdrift = new o2::trd::LocalVDrift();
  if ((entry = GetCDBentry("TRD/Calib/LocalVdrift", 0))) {
    if ((localvdrift = (AliTRDCalPad*)entry->GetObject())) {
      for (int i = 0; i < AliTRDCalDet::kNdet; i++) {
        AliTRDCalROC* calroc = 0;
        calroc = localvdrift->GetCalROC(i);
        if (calroc) {
          int rows = calroc->GetNrows();
          int cols = calroc->GetNcols();
          for (int j = 0; j < calroc->GetNchannels(); j++) {
            o2localvdrift->setPadValue(i, j, calroc->GetValue(j));
          }
        } else
          cout << "calroc is undefiend" << endl;
      }
      ccdb.storeAsTFileAny(o2localvdrift, "TRD_test/LocalVDrift", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
    } else
      cout << "attempt to get object LocalVdrift from ocdb entry. Will not be writing LocalVDritf" << endl;
  } else
    cout << "failed to retrieve ocdb entry for LocalVdrift" << endl;

  AliTRDCalPad* localT0 = 0;
  auto o2localt0 = new o2::trd::LocalT0();
  if ((entry = GetCDBentry("TRD/Calib/LocalT0", 0))) {
    if ((localT0 = (AliTRDCalPad*)entry->GetObject())) {
      for (int i = 0; i < AliTRDCalDet::kNdet; i++) {
        AliTRDCalROC* calroc = localT0->GetCalROC(i);
        int rows = calroc->GetNrows();
        int cols = calroc->GetNcols();
        for (int j = 0; j < calroc->GetNchannels(); j++) {
          o2localt0->setPadValue(i, j, calroc->GetValue(j));
        }
      }
      ccdb.storeAsTFileAny(o2localt0, "TRD_test/LocalT0", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
    } else
      cout << "attempt to get object chamber LocalT0 from ocdb entry. Will not be writing LocalT0" << endl;
  } else
    cout << "failed to retrieve ocdb entry for LocalT0" << endl;

  AliTRDCalPad* padnoise = 0;
  auto o2padnoise = new o2::trd::PadNoise();
  if ((entry = GetCDBentry("TRD/Calib/PadNoise", 0))) {
    if ((padnoise = (AliTRDCalPad*)entry->GetObject())) {
      for (int i = 0; i < AliTRDCalDet::kNdet; i++) {
        AliTRDCalROC* calroc = padnoise->GetCalROC(i);
        int rows = calroc->GetNrows();
        int cols = calroc->GetNcols();
        std::vector<float> values;
        for (int j = 0; j < calroc->GetNchannels(); j++) {
          o2padnoise->setPadValue(i, j, calroc->GetValue(j));
        }
      }
      ccdb.storeAsTFileAny(o2padnoise, "TRD_test/PadNoise", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
    } else
      cout << "attempt to get object PadNoise from ocdb entry. Will not be writing PadNoise" << endl;
  }

  AliTRDCalPad* localgainfactor = 0;
  auto o2localgainfactor = new o2::trd::LocalGainFactor();
  if ((entry = GetCDBentry("TRD/Calib/LocalGainFactor", 0))) {
    if ((localgainfactor = (AliTRDCalPad*)entry->GetObject())) {
      for (int i = 0; i < AliTRDCalDet::kNdet; i++) {
        AliTRDCalROC* calroc = localgainfactor->GetCalROC(i);
        int rows = calroc->GetNrows();
        int cols = calroc->GetNcols();
        for (int j = 0; j < calroc->GetNchannels(); j++) {
          o2localgainfactor->setPadValue(i, j, calroc->GetValue(j));
        }
      }
      ccdb.storeAsTFileAny(o2localgainfactor, "TRD_test/LocalGainFactor", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
    } else
      cout << "attempt to get object LocalGainFactor from ocdb entry. Will not be writing LocalGainFactor" << endl;
  } else
    cout << "failed to retrieve ocdb entry for LocalGainFactor" << endl;

  AliTRDCalPadStatus* padStatus = 0;
  auto o2padstatus = new o2::trd::PadStatus();
  if ((entry = GetCDBentry("TRD/Calib/PadStatus", 0))) {
    if ((padStatus = (AliTRDCalPadStatus*)entry->GetObject())) {
      for (int i = 0; i < AliTRDCalDet::kNdet; i++) {
        AliTRDCalSingleChamberStatus* rocstatus = padStatus->GetCalROC(i);
        int rows = rocstatus->GetNrows();
        int cols = rocstatus->GetNcols();
        for (int j = 0; j < rocstatus->GetNchannels(); j++) {
          o2padstatus->setStatus(i, j, rocstatus->GetStatus(j));
        }
      }
      ccdb.storeAsTFileAny(o2padstatus, "TRD_test/PadStatus", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
    }
  }

  AliTRDCalDet* chambernoise = 0;
  auto o2chambernoise = new o2::trd::ChamberNoise();
  if ((entry = GetCDBentry("TRD/Calib/DetNoise", 0))) {
    if ((chambernoise = (AliTRDCalDet*)entry->GetObject())) {
      for (int i = 0; i < AliTRDCalDet::kNdet; i++) {
        o2chambernoise->setNoise(i, chambernoise->GetValue(i));
      }
      ccdb.storeAsTFileAny(o2chambernoise, "TRD_test/ChamberNoise", metadata, 1000000000000 + Run, 1000000000000 + Run + 1);
    } else
      cout << "attempt to get object ChamberNoise from ocdb entry. Will not be writing ChamberNoise" << endl;
  }
/*
  PRFWidth is stored in a CalPad for some reason.
  Its values appear to be the same values as stored in the static class.

  AliTRDCalPad* prfwidth = 0;
  auto o2padresponse = new o2::trd::PadResponse();
  if ((entry = GetCDBentry("TRD/Calib/PRFWidth", 0))) {
    if ((prfwidth = (AliTRDCalPad*)entry->GetObject())) {
      for (int i = 0; i < AliTRDCalDet::kNdet; i++) { //need the length of calroc.
        AliTRDCalROC* calroc = prfwidth->GetCalROC(i);
        if (calroc) {
          int rows = calroc->GetNrows();
          int cols = calroc->GetNcols();
          //                o2padresponse->setPRF()
          for (int j = 0; j < calroc->GetNchannels(); j++) {
            //              o2padresponse->setPRF(i,j, calroc->GetValue(j));
            cout << "PRF[" << i << "][" << j << "]=" << calroc->GetValue(j);
          }
        }
        cout << "calroc is null for i=" << i << endl;
      }
      //ccdb.storeAsTFileAny(o2padresponse,"TRD_test/PRFWidth",metadata,1000000000000+Run,1000000000000+Run+1);
    } else
      cout << "attempt to get object PRFWidth from ocdb entry. Will not be writing PRFWidth" << endl;
  } else
    cout << "failed to retrieve ocdb entry for PRFWidth" << endl;
*/
  auto o2gtbl = new CalOnlineGainTables();
  std::string tablekey = "Krypton_2011-01";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Krypton_2011-02";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Krypton_2011-03";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Krypton_2012-01";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Krypton_2015-01";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Krypton_2015-02";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Krypton_2018-01";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Gaintbl_Uniform_FGAN0_2011-01";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.
  tablekey = "Gaintbl_Uniform_FGAN0_2012-01";
  UnpackGainTable(tablekey, o2gtbl);
  ccdb.storeAsTFileAny(o2gtbl, Form("%s/%s", TRDCalBase.c_str(), tablekey.c_str()), metadata, 1000000000000 + Run); //no uppper timestamp to leave it "always" valid.

  //AliTRDcalibDB *calibdb=AliTRDcalibDB::Instance();

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
