/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// The AliDAQ class is responsible for handling all the information about   //
// Data Acquisition configuration. It defines the detector indexing,        //
// the number of DDLs and LDCs per detector.                                //
// The number of LDCs per detector is used only in the simulation in order  //
// to define the configuration of the dateStream application. Therefore the //
// numbers in the corresponding array can be changed without affecting the  //
// rest of the aliroot code.                                                //
// The equipment ID (DDL ID) is an integer (32-bit) number defined as:      //
// Equipment ID = (detectorID << 8) + DDLIndex                              //
// where the detectorID is given by fgkDetectorName array and DDLIndex is   //
// the index of the corresponding DDL inside the detector partition.        //
// Due to DAQ/HLT limitations, the ddl indexes should be consequtive, or    //
// at least without big gaps in between.                                    //
// The sub-detector code use only this class in the simulation and reading  //
// of the raw data.                                                         //
//                                                                          //
// cvetan.cheshkov@cern.ch  2006/06/09                                      //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#include <TClass.h>
#include <TString.h>

#include "AliDAQ.h"
#include "AliLog.h"

ClassImp(AliDAQ)

const char* AliDAQ::fgkDetectorName[AliDAQ::kNDetectors] = {
  "ITSSPD",
  "ITSSDD",
  "ITSSSD",
  "TPC",
  "TRD",
  "TOF",
  "HMPID",
  "PHOS",
  "CPV",
  "PMD",
  "MUONTRK",
  "MUONTRG",
  "FMD",
  "T0",
  "VZERO",
  "ZDC",
  "ACORDE",
  "TRG",
  "EMCAL",
  "DAQ_TEST",
  "EMPTY",
  "AD",
  "MFT",
  "FIT",
  "HLT"
};

Int_t AliDAQ::fgkNumberOfDdls[AliDAQ::kNDetectors] = {
  20,   // ITSSPD
  24,   // ITSSDD
  16,   // ITSSSD
  216,  // TPC
  18,   // TRD
  72,   // TOF
  14,   // HMPID
  21,   // PHOS
  10,   // CPV
  6,    // PMD
  20,   // MUONTRK
  2,    // MUONTRG
  3,    // FMD
  1,    // T0
  1,    // VZERO
  1,    // ZDC
  1,    // ACORDE
  2,    // TRG
  46,   // EMCAL (including DCal)
  12,   // DAQ_TEST
  0,    // EMPTY
  1,    // AD
  10,   // MFT
  1,    // FIT
  28    // HLT
};

Float_t AliDAQ::fgkNumberOfLdcs[AliDAQ::kNDetectors] = {
  7,    // ITSSPD
  8,    // ITSSDD
  6,    // ITSSSD
  36,   // TPC
  9,    // TRD
  24,   // TOF
  5,    // HMPID
  8,    // PHOS
  1,    // CPV
  2,    // PMD
  7,    // MUONTRK
  1,    // MUONTRG
  1,    // FMD
  1,    // T0
  1,    // VZERO
  1,    // ZDC
  1,    // ACORDE
  1,    // TRG
  15,   // EMCAL
  2,    // DAQ_TEST
  0,    // EMPTY
  1,    // AD
  1,    // MFT
  1,    // FIT
  14    // HLT
};

const char* AliDAQ::fgkOfflineModuleName[AliDAQ::kNDetectors] = {
  "ITS",
  "ITS",
  "ITS",
  "TPC",
  "TRD",
  "TOF",
  "HMPID",
  "PHOS",
  "CPV",
  "PMD",
  "MUON",
  "MUON",
  "FMD",
  "T0",
  "VZERO",
  "ZDC",
  "ACORDE",
  "CTP",
  "EMCAL",
  "DAQ_TEST",
  "EMPTY",
  "AD",
  "MFT",
  "FIT",
  "HLT"
};

const char* AliDAQ::fgkOnlineName[AliDAQ::kNDetectors] = {
  "SPD",
  "SDD",
  "SSD",
  "TPC",
  "TRD",
  "TOF",
  "HMP",
  "PHS",
  "CPV",
  "PMD",
  "MCH",
  "MTR",
  "FMD",
  "T00",
  "V00",
  "ZDC",
  "ACO",
  "TRI",
  "EMC",
  "TST",
  "EMP",
  "AD0",
  "MFT",
  "FIT",
  "HLT"
};

Int_t AliDAQ::fgkRunPeriod = 1;
Int_t AliDAQ::fgkFirstSTUDDL = 44;
Int_t AliDAQ::fgkLastSTUDDL = 45;

AliDAQ::AliDAQ(const AliDAQ& source) :
  TObject(source)
{
  // Copy constructor
  // Nothing to be done
}

AliDAQ& AliDAQ::operator = (const AliDAQ& /* source */)
{
  // Assignment operator
  // Nothing to be done
  return *this;
}

Int_t AliDAQ::DetectorID(const char *detectorName)
{
  // Return the detector index
  // corresponding to a given
  // detector name
  TString detStr = detectorName;

  Int_t iDet;
  if (detStr.CompareTo(fgkDetectorName[kNDetectors-1],TString::kIgnoreCase)==0) { //HLT
    return kHLTId;
  }
  for(iDet = 0; iDet < kNDetectors; iDet++) {
    if (detStr.CompareTo(fgkDetectorName[iDet],TString::kIgnoreCase) == 0)
      break;
  }
  if (iDet == kNDetectors) {
    AliErrorClass(Form("Invalid detector name: %s !",detectorName));
    return -1;
  }
  return iDet;
}

const char *AliDAQ::DetectorName(Int_t detectorID)
{
  // Returns the name of particular
  // detector identified by its index
  if (detectorID==kHLTId) return fgkDetectorName[kNDetectors-1];
  if (detectorID < 0 || detectorID > kNDetectors-1) { //Accept also kNDetectors - 1 as HLT id
    AliErrorClass(Form("Invalid detector index: %d (%d -> %d, %d) !",detectorID,0,kNDetectors-2,kHLTId));
    return "";
  }
  return fgkDetectorName[detectorID];
}

Int_t AliDAQ::DdlIDOffset(const char *detectorName)
{
  // Returns the DDL ID offset
  // for a given detector
  Int_t detectorID = DetectorID(detectorName);
  if (detectorID < 0)
    return -1;
  
  return DdlIDOffset(detectorID);
}

Int_t AliDAQ::DdlIDOffset(Int_t detectorID)
{
  // Returns the DDL ID offset
  // for a given detector identified
  // by its index
  if (detectorID < 0 || (detectorID > kNDetectors-1 && detectorID!=kHLTId)) {//Accept also kNDetectors - 1 as HLT id
    AliErrorClass(Form("Invalid detector index: %d (%d -> %d, %d) !",detectorID,0,kNDetectors-2,kHLTId));
    return -1;
  }
  // HLT has a DDL offset = 30, we accept detectorID == kNDetectors - 1 as HLT detector-id as well.
  if (detectorID==kNDetectors-1) return (kHLTId << 8);

  return (detectorID << 8);
}

const char *AliDAQ::DetectorNameFromDdlID(Int_t ddlID,Int_t &ddlIndex)
{
  // Returns the detector name for
  // a given DDL ID
  ddlIndex = -1;
  Int_t detectorID = DetectorIDFromDdlID(ddlID,ddlIndex);
  if (detectorID < 0)
    return "";

  return DetectorName(detectorID);
}

Int_t AliDAQ::DetectorIDFromDdlID(Int_t ddlID,Int_t &ddlIndex)
{
  // Returns the detector ID and
  // the ddl index within the
  // detector range for
  // a given input DDL ID
  Int_t detectorID = ddlID >> 8;

  if (detectorID < 0 || (detectorID >= kNDetectors-1 && detectorID!=kHLTId)) {//detectorID comes from ddlID, so it cannot be kNDetectors - 1!
    AliErrorClass(Form("Invalid detector index: %d (%d -> %d) !",detectorID,0,kNDetectors-1));
    return -1;
  }
  ddlIndex = ddlID & 0xFF;
  if (ddlIndex >= NumberOfDdls(detectorID)) {
    AliErrorClass(Form("Invalid DDL index %d (%d -> %d) for detector %d",
		       ddlIndex,0,NumberOfDdls(detectorID),detectorID));
    ddlIndex = -1;
    return -1;
  }
  return detectorID;
}

Int_t AliDAQ::DdlID(const char *detectorName, Int_t ddlIndex)
{
  // Returns the DDL ID starting from
  // the detector name and the DDL
  // index inside the detector
  Int_t detectorID = DetectorID(detectorName);
  if (detectorID < 0)
    return -1;

  return DdlID(detectorID,ddlIndex);
}

Int_t AliDAQ::DdlID(Int_t detectorID, Int_t ddlIndex)
{
  // Returns the DDL ID starting from
  // the detector ID and the DDL
  // index inside the detector
  Int_t ddlID = DdlIDOffset(detectorID);
  if (ddlID < 0)
    return -1;
 
  if (ddlIndex >= NumberOfDdls(detectorID)) {
    AliErrorClass(Form("Invalid DDL index %d (%d -> %d) for detector %d",
		       ddlIndex,0,NumberOfDdls(detectorID),detectorID));
    return -1;
  }

  ddlID += ddlIndex;
  return ddlID;
}

const char *AliDAQ::DdlFileName(const char *detectorName, Int_t ddlIndex)
{
  // Returns the DDL file name
  // (used in the simulation) starting from
  // the detector name and the DDL
  // index inside the detector
  Int_t detectorID = DetectorID(detectorName);
  if (detectorID < 0)
    return "";

  return DdlFileName(detectorID,ddlIndex);
}

const char *AliDAQ::DdlFileName(Int_t detectorID, Int_t ddlIndex)
{
  // Returns the DDL file name
  // (used in the simulation) starting from
  // the detector ID and the DDL
  // index inside the detector
  Int_t ddlID = DdlIDOffset(detectorID);
  if (ddlID < 0)
    return "";
  
  if (ddlIndex >= NumberOfDdls(detectorID)) {
    AliErrorClass(Form("Invalid DDL index %d (%d -> %d) for detector %d",
		       ddlIndex,0,NumberOfDdls(detectorID),detectorID));
    return "";
  }

  ddlID += ddlIndex;
  static TString fileName;

  fileName = DetectorName(detectorID);
  fileName += "_";
  fileName += ddlID;
  fileName += ".ddl";
  return fileName.Data();
}

Int_t AliDAQ::NumberOfDdls(const char *detectorName)
{
  // Returns the number of DDLs for
  // a given detector
  Int_t detectorID = DetectorID(detectorName);
  if (detectorID < 0)
    return -1;

  return NumberOfDdls(detectorID);
}

Int_t AliDAQ::NumberOfDdls(Int_t detectorID)
{
  // Returns the number of DDLs for
  // a given detector
  if (detectorID < 0 || (detectorID > kNDetectors-1 && detectorID!=kHLTId)) {//Accept also kNDetectors - 1 as HLT id
    AliErrorClass(Form("Invalid detector index: %d (%d -> %d, %d) !",detectorID,0,kNDetectors-2,kHLTId));
    return -1;
  }

  int detectorIDindex = detectorID;
  if (detectorID==kHLTId) detectorIDindex=kNDetectors-1;

  return fgkNumberOfDdls[detectorIDindex];
}

Float_t AliDAQ::NumberOfLdcs(const char *detectorName)
{
  // Returns the number of DDLs for
  // a given detector
  Int_t detectorID = DetectorID(detectorName);
  if (detectorID < 0)
    return -1;

  return NumberOfLdcs(detectorID);
}

Float_t AliDAQ::NumberOfLdcs(Int_t detectorID)
{
  // Returns the number of DDLs for
  // a given detector
  if (detectorID < 0 || (detectorID > kNDetectors-1 && detectorID!=kHLTId)) {//Accept also kNDetectors - 1 as HLT id
    AliErrorClass(Form("Invalid detector index: %d (%d -> %d, %d) !",detectorID,0,kNDetectors-2,kHLTId));
    return -1;
  }

  int detectorIDindex = detectorID;
  if (detectorID==kHLTId) detectorIDindex=kNDetectors-1;

  return fgkNumberOfLdcs[detectorIDindex];
}

void AliDAQ::PrintConfig()
{
  // Print the DAQ configuration
  // for all the detectors
  printf("===================================================================================================\n"
	 "|                              ALICE Data Acquisition Configuration                               |\n"
	 "===================================================================================================\n"
	 "| Detector ID | Detector Name | DDL Offset | # of DDLs | # of LDCs | Online Name | AliRoot Module |\n"
	 "===================================================================================================\n");
  for(Int_t iDet = 0; iDet < kNDetectors; iDet++) {
    printf("|%11d  |%13s  |%10d  |%9d  |%9.1f  |%11s  |%14s  |\n",
	   iDet,DetectorName(iDet),DdlIDOffset(iDet),NumberOfDdls(iDet),NumberOfLdcs(iDet),
	   OnlineName(iDet),OfflineModuleName(iDet));
  }
  printf("===================================================================================================\n");

}

const char *AliDAQ::ListOfTriggeredDetectors(UInt_t detectorPattern)
{
  // Returns a string with the list of
  // active detectors. The input is the
  // trigger pattern word contained in
  // the raw-data event header.

  static TString detList;
  detList = "";
  for(Int_t iDet = 0; iDet < (kNDetectors-1); iDet++) {
    if ((detectorPattern >> iDet) & 0x1) {
      detList += fgkDetectorName[iDet];
      detList += " ";
    }
  }

  // Always remember HLT
  if ((detectorPattern >> kHLTId) & 0x1) detList += fgkDetectorName[kNDetectors-1];

  return detList.Data();
}

UInt_t  AliDAQ::DetectorPattern(const char *detectorList)
{
  // Returns a 32-bit word containing the
  // the detector pattern corresponding to a given
  // list of detectors
  UInt_t pattern = 0;
  TString detList = detectorList;
  for(Int_t iDet = 0; iDet < (kNDetectors-1); iDet++) {
    TString det = fgkDetectorName[iDet];
    if((detList.CompareTo(det) == 0) || 
       detList.BeginsWith(det) ||
       detList.EndsWith(det) ||
       detList.Contains( " "+det+" " )) pattern |= (1 << iDet) ;
  }

  // HLT
  TString hltDet = fgkDetectorName[kNDetectors-1];
  if((detList.CompareTo(hltDet) == 0) || 
       detList.BeginsWith(hltDet) ||
       detList.EndsWith(hltDet) ||
       detList.Contains( " "+hltDet+" " )) pattern |= (1 << kHLTId) ;
  
  return pattern;
}

UInt_t  AliDAQ::DetectorPatternOffline(const char *detectorList)
{
  // Returns a 32-bit word containing the
  // the detector pattern corresponding to a given
  // list of detectors.
  // The list of detectors must follow offline module
  // name convention.
  UInt_t pattern = 0;
  TString detList = detectorList;
  for(Int_t iDet = 0; iDet < (kNDetectors-1); iDet++) {
    TString det = fgkOfflineModuleName[iDet];
    if((detList.CompareTo(det) == 0) || 
       detList.BeginsWith(det) ||
       detList.EndsWith(det) ||
       detList.Contains( " "+det+" " )) pattern |= (1 << iDet) ;
  }

  // HLT
  TString hltDet = fgkOfflineModuleName[kNDetectors-1];
  if((detList.CompareTo(hltDet) == 0) || 
       detList.BeginsWith(hltDet) ||
       detList.EndsWith(hltDet) ||
       detList.Contains( " "+hltDet+" " )) pattern |= (1 << kHLTId) ;
  
  return pattern;
}

const char *AliDAQ::OfflineModuleName(const char *detectorName)
{
  // Returns the name of the offline module
  // for a given detector (online naming convention)
  Int_t detectorID = DetectorID(detectorName);
  if (detectorID < 0)
    return "";

  return OfflineModuleName(detectorID);
}

const char *AliDAQ::OfflineModuleName(Int_t detectorID)
{
  // Returns the name of the offline module
  // for a given detector (online naming convention)
  if (detectorID < 0 || (detectorID > kNDetectors-1 && detectorID!=kHLTId)) { //Accept also kNDetectors - 1 as HLT id
    AliErrorClass(Form("Invalid detector index: %d (%d -> %d, %d) !",detectorID,0,kNDetectors-2,kHLTId));
    return "";
  }

  int detectorIDindex=detectorID;
  if (detectorID==kHLTId) detectorIDindex = kNDetectors-1;

  return fgkOfflineModuleName[detectorIDindex];
}

const char *AliDAQ::OnlineName(const char *detectorName)
{
  // Returns the name of the online detector name (3 characters)
  // for a given detector
  Int_t detectorID = DetectorID(detectorName);
  if (detectorID < 0)
    return "";

  return OnlineName(detectorID);
}

const char *AliDAQ::OnlineName(Int_t detectorID)
{
  // Returns the name of the online detector name (3 characters)
  // for a given detector
  if (detectorID < 0 || (detectorID > kNDetectors-1 && detectorID!=kHLTId)) { //Accept also kNDetectors - 1 as HLT id
    AliErrorClass(Form("Invalid detector index: %d (%d -> %d, %d) !",detectorID,0,kNDetectors-2,kHLTId));
    return "";
  }

  int detectorIDindex=detectorID;
  if (detectorID==kHLTId) detectorIDindex = kNDetectors-1;

  return fgkOnlineName[detectorIDindex];
}

void AliDAQ::SetRun1(){
  // Set RunPeriod
  fgkRunPeriod = 1;
  // STU
  fgkFirstSTUDDL=44;
  fgkLastSTUDDL=44;
  
  // Change the default values to the ones used in Run1
  // DDL
  fgkNumberOfDdls[6] = 20; // HMPID in Run1
  fgkNumberOfDdls[17] = 1; // TRG
  fgkNumberOfDdls[18] = 46; // EMCAL
  fgkNumberOfDdls[19] = 1; // DAQ_TEST

  // LDC
  fgkNumberOfLdcs[0] = 4; // ITSSPD
  fgkNumberOfLdcs[1] = 4; // ITSSDD
  fgkNumberOfLdcs[2] = 4; // ITSSSD
  fgkNumberOfLdcs[4] = 3; // TRD
  fgkNumberOfLdcs[5] = 12; // TOF
  fgkNumberOfLdcs[6] = 4; // HMPID
  fgkNumberOfLdcs[7] = 4; // PHOS
  fgkNumberOfLdcs[8] = 2; // CPV
  fgkNumberOfLdcs[9] = 1; // PMD
  fgkNumberOfLdcs[10] = 5; // MUONTRK
  fgkNumberOfLdcs[13] = 0.5; // T0
  fgkNumberOfLdcs[14] = 0.5; // VZERO
  fgkNumberOfLdcs[18] = 8; // EMCAL
  fgkNumberOfLdcs[19] = 1; // DAQ_TEST
  fgkNumberOfLdcs[24] = 7; // HLT
}

void AliDAQ::SetRun2(){
  // Set RunPeriod
  fgkRunPeriod = 2;
  // STU
  fgkFirstSTUDDL=44;
  fgkLastSTUDDL=45;
 
  // Change the default values to the ones used in Run2
  // DDL
  fgkNumberOfDdls[6] = 14; // HMPID in Run2
  fgkNumberOfDdls[17] = 2; // TRG
  fgkNumberOfDdls[18] = 46; // EMCAL
  fgkNumberOfDdls[19] = 12; // DAQ_TEST

  // LDC
  fgkNumberOfLdcs[0] = 7; // ITSSPD
  fgkNumberOfLdcs[1] = 8; // ITSSDD
  fgkNumberOfLdcs[2] = 6; // ITSSSD
  fgkNumberOfLdcs[4] = 9; // TRD
  fgkNumberOfLdcs[5] = 24; // TOF
  fgkNumberOfLdcs[6] = 5; // HMPID
  fgkNumberOfLdcs[7] = 8; // PHOS
  fgkNumberOfLdcs[8] = 1; // CPV
  fgkNumberOfLdcs[9] = 2; // PMD
  fgkNumberOfLdcs[10] = 7; // MUONTRK
  fgkNumberOfLdcs[13] = 1; // T0
  fgkNumberOfLdcs[14] = 1; // VZERO
  fgkNumberOfLdcs[18] = 15; // EMCAL
  fgkNumberOfLdcs[19] = 2; // DAQ_TEST
  fgkNumberOfLdcs[24] = 14; // HLT
}
