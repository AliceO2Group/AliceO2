/**
 * @file initTPCcalibration.C
 * @brief A macro to initialize AliTPCTransform cluster transformation in AliTPCcalib
 *
 * <pre>
 * Usage:
 *
 * aliroot $ALICE_ROOT/GPU/TPCFastTransformation/macro/initTPCcalibration.C'("uri", runNumber, isMC)'
 * uri == "alien://Folder=/alice/data/2015/OCDB"
 * uri == "local://$HOME/alice/OCDB"
 * uri == "OCDBsim.root"
 *
 * </pre>
 *
 * Parameters: <br>
 * - uri       the OCDB URI. When ==nullptr, AliCDBManager::Instance().IsDefaultStorageSet() should be 1
 * - runNumber run number
 * - isMC      initialize for Monte Carlo
 *
 * @author sergey gorbunov
 *
 */

/*
   aliroot
   .L initTPCcalibration.C
   initTPCcalibration("alien://Folder=/alice/data/2015/OCDB",246984,1)
   initTPCcalibration("$ALICE_ROOT/../aliceEventsPbPb/OCDBsim.root",246984,1)
 */

#include "AliTPCcalibDB.h"
#include "Riostream.h"
#include "TGeoGlobalMagField.h"
#include "AliGRPObject.h"
#include "AliGRPManager.h"
#include "AliGeomManager.h"
#include "AliTracker.h"
#include "AliCDBRunRange.h"
#include "AliCDBManager.h"
#include "AliCDBStorage.h"
#include "AliTPCRecoParam.h"
#include "AliCDBEntry.h"
#include "TMap.h"
#include "AliRawEventHeaderBase.h"
#include "AliEventInfo.h"
#include "AliRunInfo.h"
#include "AliTPCTransform.h"

using namespace std;

int32_t initTPCcalibration(const Char_t* cdbUri, int32_t runNumber, bool isMC)
{

  // --------------------------------------
  // -- Setup CDB
  // --------------------------------------

  // cdbUri = "local://$ALICE_ROOT/OCDB";
  // cdbUri = "alien://Folder=/alice/data/2015/OCDB";
  // cdbUri = "OCDBsim.root";
  // cdbUri = "$ALICE_ROOT/../aliceEventsPbPb/OCDBsim.root";
  // cdbUri="/home/gorbunov/alice/aliceEventsPbPb/OCDB.root";

  AliCDBManager* cdbm = AliCDBManager::Instance();
  if (!cdbm) {
    cerr << "Error : Can not get AliCDBManager" << endl;
    return -1;
  }

  if (cdbUri != 0) {
    TString storage = cdbUri;
    cout << storage.Data() << endl;
    if (storage.Contains(".root")) {
      // local file
      cout << "Snapshot mode" << endl;
      cdbm->SetSnapshotMode(cdbUri);
      cdbm->SetDefaultStorage("local://$ALICE_ROOT/OCDB");
    } else {
      if (!storage.Contains("://")) { // add prefix to local path
        storage = "local://";
        storage += cdbUri;
      }
      cdbm->SetDefaultStorage(storage);
    }
  }

  if (!cdbm->IsDefaultStorageSet()) {
    cerr << "OCDB storage is not set!!" << endl;
    return -1;
  }

  cdbm->SetRun(runNumber);

  AliGRPManager grp;
  grp.ReadGRPEntry();
  grp.SetMagField();

  const AliGRPObject* grpObj = grp.GetGRPData();

  if (!grpObj) {
    cerr << "No GRP object found!!" << endl;
    return -1;
  }

  if (!AliGeomManager::GetGeometry()) {
    AliGeomManager::LoadGeometry();
  }
  if (!AliGeomManager::GetGeometry()) {
    cerr << "Can not initialise geometry" << endl;
    return -1;
  }

  AliTPCcalibDB* tpcCalib = AliTPCcalibDB::Instance();
  if (!tpcCalib) {
    cerr << "AliTPCcalibDB does not exist" << endl;
    return -1;
  }

  const AliMagF* field = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();

  if (!field) {
    cerr << "no magnetic field found " << endl;
    return -1;
  }

  tpcCalib->SetExBField(field);
  tpcCalib->SetRun(runNumber);
  tpcCalib->UpdateRunInformations(runNumber);

  if (!tpcCalib->GetTransform()) {
    cerr << "No TPC transformation found" << endl;
    return -1;
  }

  // -- Get AliRunInfo variables

  AliRunInfo runInfo(grpObj->GetLHCState(), grpObj->GetBeamType(), grpObj->GetBeamEnergy(), grpObj->GetRunType(), grpObj->GetDetectorMask());
  AliEventInfo evInfo;
  evInfo.SetEventType(AliRawEventHeaderBase::kPhysicsEvent);

  AliCDBEntry* entry = AliCDBManager::Instance()->Get("TPC/Calib/RecoParam");

  if (!entry) {
    cerr << "No TPC reco param entry found in data base" << endl;
    return -1;
  }

  TObject* aliRecoParamObj = entry->GetObject();
  if (!aliRecoParamObj) {
    cerr << " Empty TPC reco param entry in data base" << endl;
    return -1;
  }

  AliRecoParam aliRecoParam;

  if (dynamic_cast<TObjArray*>(aliRecoParamObj)) {
    // cout<<"\n\nSet reco param from AliHLTTPCClusterTransformation: TObjArray found \n"<<endl;
    TObjArray* copy = (TObjArray*)(static_cast<TObjArray*>(aliRecoParamObj)->Clone());
    aliRecoParam.AddDetRecoParamArray(1, copy);
  } else if (dynamic_cast<AliDetectorRecoParam*>(aliRecoParamObj)) {
    // cout<<"\n\nSet reco param from AliHLTTPCClusterTransformation: AliDetectorRecoParam found \n"<<endl;
    AliDetectorRecoParam* copy = (AliDetectorRecoParam*)static_cast<AliDetectorRecoParam*>(aliRecoParamObj)->Clone();
    aliRecoParam.AddDetRecoParam(1, copy);
  } else {
    cerr << "Unknown format of the TPC Reco Param entry in the data base" << endl;
    return -1;
  }

  aliRecoParam.SetEventSpecie(&runInfo, evInfo, 0);

  //

  AliTPCRecoParam* recParam = (AliTPCRecoParam*)aliRecoParam.GetDetRecoParam(1);

  if (!recParam) {
    cerr << "No TPC Reco Param entry found for the given event specification" << endl;
    return -1;
  }

  recParam = new AliTPCRecoParam(*recParam);

  uint32_t timeStamp = grpObj->GetTimeStart();

  if (isMC && !recParam->GetUseCorrectionMap()) {
    timeStamp = 0;
  }

  tpcCalib->GetTransform()->SetCurrentRecoParam(recParam);

  AliTPCTransform* origTransform = tpcCalib->GetTransform();
  origTransform->SetCurrentTimeStamp(static_cast<uint32_t>(timeStamp));

  Double_t bz = AliTracker::GetBz();
  cout << "\n\nBz field is set to " << bz << ", time stamp is set to " << timeStamp << endl
       << endl;

  return 0;
}
