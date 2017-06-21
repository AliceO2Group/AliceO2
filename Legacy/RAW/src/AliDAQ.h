#ifndef ALIDAQ_H
#define ALIDAQ_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

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

#include <TObject.h>

class AliDAQ: public TObject {
 public:

  AliDAQ() {};
  AliDAQ(const AliDAQ& source);
  AliDAQ& operator = (const AliDAQ& source);
  virtual ~AliDAQ() {};

  static Int_t       DetectorID(const char *detectorName);
  static const char *DetectorName(Int_t detectorID);

  static Int_t       DdlIDOffset(const char *detectorName);
  static Int_t       DdlIDOffset(Int_t detectorID);

  static const char *DetectorNameFromDdlID(Int_t ddlID, Int_t &ddlIndex);
  static Int_t       DetectorIDFromDdlID(Int_t ddlID, Int_t &ddlIndex);

  static Int_t       DdlID(const char *detectorName, Int_t ddlIndex);
  static Int_t       DdlID(Int_t detectorID, Int_t ddlIndex);
  static const char *DdlFileName(const char *detectorName, Int_t ddlIndex);
  static const char *DdlFileName(Int_t detectorID, Int_t ddlIndex);

  static Int_t       NumberOfDdls(const char *detectorName);
  static Int_t       NumberOfDdls(Int_t detectorID);

  static Float_t     NumberOfLdcs(const char *detectorName);
  static Float_t     NumberOfLdcs(Int_t detectorID);

  static void        PrintConfig();

  static const char *ListOfTriggeredDetectors(UInt_t detectorPattern);
  static UInt_t      DetectorPattern(const char *detectorList);
  static UInt_t      DetectorPatternOffline(const char *detectorList);

  static const char *OfflineModuleName(const char *detectorName);
  static const char *OfflineModuleName(Int_t detectorID);

  static const char *OnlineName(const char *detectorName);
  static const char *OnlineName(Int_t detectorID);

  static void SetRun1();
  static void SetRun2();
  static Int_t GetRunPeriod()  {return fgkRunPeriod;}
  static Int_t GetFirstSTUDDL() {return fgkFirstSTUDDL;}
  static Int_t GetLastSTUDDL() {return fgkLastSTUDDL;}
  
  enum {
    kNDetectors = 25,    // Number of detectors
    kHLTId = 30          // HLT detector index
  };

  enum DetectorBits {kSPD = 0x0001, kSDD = 0x0002, kSSD = 0x0004, kITS = 0x0007, 
		     kTPC = 0x0008, kTRD = 0x0010, kTOF = 0x0020, kHMPID = 0x0040, 
		     kPHOS = 0x0080, kCPV = 0x0100, kPMD = 0x0200, kMUONTRK = 0x0400,
		     kMUONTRG = 0x0800, kMUON = 0x0c00, kFMD = 0x1000, kT0 = 0x2000, kVZERO = 0x4000,
		     kZDC = 0x8000, kACORDE = 0x10000, kTRG = 0x20000, kEMCAL = 0x40000,
		     kDAQTEST = 0x80000, kEMPTY= 0x100000, kAD = 0x200000, kMFT = 0x400000, kFIT = 0x800000, kHLT = 0x40000000};

  enum DetectorBitsQualityFlag {kACORDE_QF   = 0x000001, kAD_QF       = 0x000002, kCPV_QF  = 0x000004, kDAQ_TEST_QF = 0x000008, 
				kEMCAL_QF    = 0x000010, kFMD_QF      = 0x000020, kHLT_QF  = 0x000040, kHMPID_QF    = 0x000080, 
				kMUON_TRG_QF = 0x000100, kMUON_TRK_QF = 0x000200, kPHOS_QF = 0x000400, kPMD_QF      = 0x000800, 
				kSDD_QF      = 0x001000, kSPD_QF      = 0x002000, kSSD_QF  = 0x004000, kT0_QF       = 0x008000, 
				kTOF_QF      = 0x010000, kTPC_QF      = 0x020000, kTRD_QF  = 0x040000, kTRIGGER_QF  = 0x080000, 
				kV0_QF       = 0x100000, kZDC_QF      = 0x200000};

 private:

  static const char *fgkDetectorName[kNDetectors]; // Detector names
  static Int_t       fgkNumberOfDdls[kNDetectors]; // Number of DDLs per detector
  static Float_t     fgkNumberOfLdcs[kNDetectors]; // Number of LDCs per detector (not fixed - used only for the raw data simulation)
  static const char* fgkOfflineModuleName[kNDetectors]; // Names of the offline modules corresponding to the detectors
  static const char* fgkOnlineName[kNDetectors]; // Online (DAQ/ECS) detector names
  static Int_t fgkRunPeriod; // 1 corresponds to Run1, 1 - to Run2
  static Int_t fgkFirstSTUDDL; // ID of the first STU DDLwithin the EMCAL range
  static Int_t fgkLastSTUDDL; // ID of the last STU DDL within the EMCAL range

  ClassDef(AliDAQ, 6)   // ALICE DAQ Configuration class
};

#endif
