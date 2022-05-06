// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>
#include <string>
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include <unordered_map>
#include <chrono>

using DPID = o2::dcs::DataPointIdentifier;

int makeGRPCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for GRP with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;

  std::vector<std::string> aliasesBFieldDouble = {"L3Current", "DipoleCurrent"};
  std::vector<std::string> aliasesBFieldBool = {"L3Polarity", "DipolePolarity"};
  std::vector<std::string> aliasesEnvVar = {"CavernTemperature", "CavernAtmosPressure", "SurfaceAtmosPressure", "CavernAtmosPressure2"};
  std::vector<std::string> compactAliasesLHCIFDouble = {"LHC_IntensityBeam[1..2]_totalIntensity", "ALI_Background[1..3]",
                                                        "ALI_Lumi_Total_Inst",
                                                        "BPTX_deltaT_B1_B2", "BPTX_deltaTRMS_B1_B2",
                                                        "BPTX_Phase_B[1..2]", "BPTX_PhaseRMS_B[1..2]", "BPTX_Phase_Shift_B[1..2]"};
  std::vector<std::string> aliasesLHCIFDouble = o2::dcs::expandAliases(compactAliasesLHCIFDouble);
  std::vector<std::string> aliasesLHCIFString = {"ALI_Lumi_Source_Name", "MACHINE_MODE", "BEAM_MODE"};
  std::vector<std::string> aliasesLHCIFCollimators = {"LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_downstream", "LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_upstream",
                                                      "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_downstream", "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_upstream",
                                                      "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_downstream", "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_upstream"};

  DPID dpidtmp;
  for (size_t i = 0; i < aliasesBFieldDouble.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesBFieldDouble[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "GRPDATAPOINTS";
  }
  for (size_t i = 0; i < aliasesBFieldBool.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesBFieldBool[i], o2::dcs::DeliveryType::DPVAL_BOOL);
    dpid2DataDesc[dpidtmp] = "GRPDATAPOINTS";
  }
  for (size_t i = 0; i < aliasesEnvVar.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesEnvVar[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "GRPDATAPOINTS";
  }
  for (size_t i = 0; i < aliasesLHCIFDouble.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesLHCIFDouble[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "GRPDATAPOINTS";
  }
  for (size_t i = 0; i < aliasesLHCIFCollimators.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesLHCIFCollimators[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "GRPDATAPOINTS";
  }
  for (size_t i = 0; i < aliasesLHCIFString.size(); ++i) {
    DPID::FILL(dpidtmp, aliasesLHCIFString[i], o2::dcs::DeliveryType::DPVAL_STRING);
    dpid2DataDesc[dpidtmp] = "GRPDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "GRP/Config/DCSDPconfig", md, ts, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);

  return 0;
}
