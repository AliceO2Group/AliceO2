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
//    you then call :
//    alienv enter VO_ALICE@O2::latest-O2-o2,VO_ALICE@AliRoot::latest-O2-o2
//    according to my configs, modify as required of course.
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

  for (int i = 0; i < 540; i++) {
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
    for (int i = 0; i < 540; i++) {
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
void Readocdb(Int_t run, const Char_t* storageURI = "alien://folder=/alice/data/2010/OCDB/")
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
        cout << "Reading in ChamberStatus : " << i << "::" << (int)a << endl;
        o2chamberstatus->setRawStatus(i, a);
        cout << "o2chamberstatus is : " << (int)o2chamberstatus->getStatus(i) << endl;
        ;
      }
      // store abitrary user object in strongly typed manner
      //      ccdb.storeAsTFileAny(o2chamberstatus, "TRD_test/ChamberStatus", metadata, Run, Run + 1);
      // // read like this (you have to specify the type)
      //auto o2chamberstatusback = ccdb.retrieveFromTFileAny<o2::trd::ChamberStatus>("TRD_test/ChamberStatus", metadata);
    } else
      cout << "attempt to get object chamber status from ocdb entry. Ergo not writing one to ccdb." << endl;
  } else
    cout << "failed to retrieve ocdb entry for Chamber Status" << endl;

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
