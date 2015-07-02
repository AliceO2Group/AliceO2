#include "TFile.h"

#include "o2cdb/Condition.h"
#include "o2cdb/ConditionId.h"
#include "o2cdb/ConditionMetaData.h"
#include "o2cdb/IdRunRange.h"
#include "o2cdb/Storage.h"
#include "o2cdb/Manager.h"

#include "tpc/dirty/AliTPCParam.h"

using namespace AliceO2::CDB;

/*

gSystem->AddIncludePath("-I$ALICEO2/")
.L MakeCDBEntry.C+
MakeCDBEntry()

*/

void MakeCDBEntry()
{
  // defaults
  Manager *cdbManager = Manager::Instance();
  Storage* targetStorage = cdbManager->getStorage("local://../o2cdb");
  IdRunRange rangeZeroInfty(0,IdRunRange::Infinity());
  ConditionMetaData *metaData = new ConditionMetaData("Jens");

  //
  // get the tpcParameters object and dump it on the CDB
  //
  TFile f("param.root");
  AliTPCParam *par=(AliTPCParam*)f.Get("tpcParameters");
  f.Close();

  ConditionId tpcParametersId("TPC/Calib/Parameters",rangeZeroInfty);
  targetStorage->putObject(par, tpcParametersId, metaData);
}
