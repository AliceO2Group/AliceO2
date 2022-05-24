#include "TObject.h"
#include "TString.h"
#include "TSystem.h"
#include "TKey.h"
#include "TFile.h"
#include "TDirectory.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCBase/Constants.h"
#include "ZDCSimulation/SimCondition.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCReconstruction/ZDCTDCCorr.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/InterCalibConfig.h"
#include "ZDCCalib/WaveformCalibConfig.h"
#include "ZDCCalib/WaveformCalibParam.h"

void InspectCCDBFile()
{
  TString dn = gDirectory->GetName();
  auto p_und = dn.First('_');
  auto p_dot = dn.Last('.');
  if (p_und >= 0 && p_dot > 0 && p_dot > p_und) {
    TSubString dat = dn(p_und + 1, p_dot - p_und - 1);
    TString data = dat;
    if (data.IsDec()) {
      auto val = data.Atoll();
      val = val / 1000;
      gSystem->Exec(TString::Format("date -d \"@%lld\"", val));
    }
  }
  TIter nextkey(gDirectory->GetListOfKeys());
  TKey* key;
  while ((key = (TKey*)nextkey())) {
    TString cn = key->GetClassName();
    if (cn.EqualTo("o2::zdc::ModuleConfig")) {
      o2::zdc::ModuleConfig* ob = (o2::zdc::ModuleConfig*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathConfigModule.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::SimCondition")) {
      // o2::zdc::SimCondition *ob=(o2::zdc::SimCondition *)key->ReadObj();
      o2::zdc::SimCondition* ob = (o2::zdc::SimCondition*)gFile->GetObjectUnchecked("ccdb_object");
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathConfigSim.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::RecoConfigZDC")) {
      o2::zdc::RecoConfigZDC* ob = (o2::zdc::RecoConfigZDC*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathRecoConfigZDC.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::ZDCTDCCorr")) {
      o2::zdc::ZDCTDCCorr* ob = (o2::zdc::ZDCTDCCorr*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathTDCCorr.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::ZDCTDCParam")) {
      o2::zdc::ZDCTDCParam* ob = (o2::zdc::ZDCTDCParam*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathTDCCalib.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::ZDCEnergyParam")) {
      o2::zdc::ZDCEnergyParam* ob = (o2::zdc::ZDCEnergyParam*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathEnergyCalib.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::ZDCTowerParam")) {
      o2::zdc::ZDCTowerParam* ob = (o2::zdc::ZDCTowerParam*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathTowerCalib.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::InterCalibConfig")) {
      o2::zdc::InterCalibConfig* ob = (o2::zdc::InterCalibConfig*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathInterCalibConfig.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::WaveformCalibConfig")) {
      o2::zdc::WaveformCalibConfig* ob = (o2::zdc::WaveformCalibConfig*)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathWaveformCalibConfig.data());
      ob->print();
    } else if (cn.EqualTo("o2::zdc::WaveformCalibParam")) {
      o2::zdc::WaveformCalibParam* ob = (o2::zdc::WaveformCalibParam*)gFile->GetObjectUnchecked("ccdb_object");
      // o2::zdc::WaveformCalibParam *ob=(o2::zdc::WaveformCalibParam *)key->ReadObj();
      printf("%s %s %d %s @ %s\n", "OBJ", key->GetName(), key->GetCycle(), key->GetTitle(), o2::zdc::CCDBPathWaveformCalib.data());
      ob->print();
      ob->saveDebugHistos("InspectCCDBFile_WaveformCalibParam.root");
    } else {
      printf("%s %s %d %s\n", key->GetClassName(), key->GetName(), key->GetCycle(), key->GetTitle());
    }
  }

  //   TObject *ob = (TObject*)gDirectory->Get("ccdb_object");
  //   if(ob == nullptr){
  //     printf("Object not found\n");
  //     return;
  //   }
  //   printf("%s %d\n", ob->Class_Name(), ob->Class_Version());
}
