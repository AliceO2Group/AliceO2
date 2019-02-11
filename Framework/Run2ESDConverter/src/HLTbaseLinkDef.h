// $Id$
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ class AliHLTAsyncProcessor;
#pragma link C++ class AliHLTComponent+;
#pragma link C++ class AliHLTComponentHandler+;
#pragma link C++ class AliHLTSystem+;
#pragma link C++ class AliHLTReconstructorBase+;
#pragma link C++ class AliHLTPluginBase+;
#pragma link C++ class AliHLTProcessor+;
#pragma link C++ class AliHLTCalibrationProcessor+;
#pragma link C++ class AliHLTConfiguration+;
#pragma link C++ class AliHLTComponentConfiguration+;
#pragma link C++ class AliHLTConfigurationHandler+;
#pragma link C++ class AliHLTOnlineConfiguration+;
#pragma link C++ class AliHLTTTreeProcessor+;
#pragma link C++ class AliHLTTask+;
#pragma link C++ class AliHLTDumpTask+;
#pragma link C++ class AliHLTControlTask+;
#pragma link C++ class AliHLTLogging+;
#pragma link C++ class AliHLTErrorGuard+;
#pragma link C++ class AliHLTDataBuffer+;
#pragma link C++ class AliHLTDataBuffer::AliHLTRawBuffer+;
#pragma link C++ class AliHLTDataBuffer::AliHLTRawPage+;
#pragma link C++ class AliHLTConsumerDescriptor+;
#pragma link C++ class AliHLTDataSource+;
#pragma link C++ class AliHLTDataSink+;
#pragma link C++ class AliHLTOfflineInterface+;
#pragma link C++ class AliHLTOfflineDataSource+;
#pragma link C++ class AliHLTModuleAgent+;
#pragma link C++ class AliHLTModulePreprocessor+;
#pragma link C++ class AliHLTShuttleInterface+;
#pragma link C++ class AliHLTDimServer+;
#pragma link C++ class AliHLTHOMERLibManager+;
#pragma link C++ class AliHLTHOMERManager+;
#pragma link C++ class AliHLTDirectHOMERManager+;
#pragma link C++ class AliHLTHOMERProxyHandler+;
#pragma link C++ class AliHLTHOMERBlockDesc+;
#pragma link C++ class AliHLTHOMERSourceDesc+;
#pragma link C++ class AliHLTEsdManager+;
#pragma link C++ class AliHLTDAQ+;
#pragma link C++ class AliHLTOUT+;
#pragma link C++ class AliHLTOUTHomerBuffer+;
#pragma link C++ class AliHLTOUTTask+;
#pragma link C++ class AliHLTOUTHandler+;
#pragma link C++ class AliHLTOUTHandlerIgnore+;
#pragma link C++ class AliHLTOUTHandlerEquId+;
#pragma link C++ class AliHLTOUTHandlerDetectorDDL+;
#pragma link C++ class AliHLTOUTHandlerChain+;
#pragma link C++ class AliHLTOUTHandlerEsdBranch+;
#pragma link C++ class AliHLTMemoryFile+;
#pragma link C++ class AliHLTMessage+;
#pragma link C++ class AliHLTEventStatistics+;
#pragma link C++ class AliHLTBlockDataCollection+;
#pragma link C++ class AliHLTTriggerDecision+;
#pragma link C++ class AliHLTComponentBenchmark+;
#pragma link C++ class AliHLTDataDeflater+;
#pragma link C++ class AliHLTDataDeflaterSimple+;
#pragma link C++ class AliHLTDataDeflaterHuffman+;
#pragma link C++ class AliHLTHuffmanNode+;
#pragma link C++ class AliHLTHuffmanTreeNode+;
#pragma link C++ class AliHLTHuffmanLeaveNode+;
#pragma link C++ class AliHLTHuffman+;
#pragma link C++ class AliHLTDataInflater+;
#pragma link C++ class AliHLTDataInflaterSimple+;
#pragma link C++ class AliHLTDataInflaterHuffman+;
#pragma link C++ class AliHLTObjArray-;
#pragma link C++ class AliHLTList-;

#include "RVersion.h"
#if ROOT_VERSION_CODE < 334336 //ROOT_VERSION(5,26,0)

#pragma link C++ class AliHLTGlobalTriggerDecision-;  // '-' option since the class uses a custom streamer.
#pragma link C++ class AliHLTReadoutList-;  // '-' option since the class uses a custom streamer.

#else // ROOT version check

#pragma link C++ class AliHLTGlobalTriggerDecision+;

// Scheme rule to mark all objects in the trigger decision loaded from file as
// deletable. Meaning the new object owns all the input objects.
#pragma read sourceClass="AliHLTGlobalTriggerDecision" version="[1-]" targetClass="AliHLTGlobalTriggerDecision"\
  source="" target="" code="{ newObj->MarkInputObjectsAsOwned(); }"

#pragma link C++ class AliHLTReadoutList+;

// Do nothing special with schema evolution for new versions of the readout list.
#pragma read sourceClass="AliHLTReadoutList" version="[4-]" targetClass="AliHLTReadoutList"

// For old versions we need to convert the format of the readout list into the new one.
#pragma read sourceClass="AliHLTReadoutList" version="[3]" targetClass="AliHLTReadoutList"\
                            source="AliHLTEventDDL fReadoutList" target="fReadoutList"\
  code="{\
          fReadoutList.fCount = gkAliHLTDDLListSize;\
          for (int i = 0; i<30; ++i) fReadoutList.fList[i] = onfile.fReadoutList.fList[i];\
          fReadoutList.fList[30] = 0x0;\
          fReadoutList.fList[31] = onfile.fReadoutList.fList[30];\
  }"
#pragma read sourceClass="AliHLTReadoutList" version="[1-2]" targetClass="AliHLTReadoutList"\
  source="AliHLTEventDDL fReadoutList" target="fReadoutList"\
  code="{\
          fReadoutList.fCount = gkAliHLTDDLListSize;\
          for (int i = 0; i < 28; ++i) fReadoutList.fList[i] = onfile.fReadoutList.fList[i];\
          fReadoutList.fList[28] = 0x0;\
          fReadoutList.fList[29] = onfile.fReadoutList.fList[28];\
          fReadoutList.fList[30] = 0x0;\
          fReadoutList.fList[31] = onfile.fReadoutList.fList[29];\
  }"

#endif // ROOT version check

#pragma link C++ class AliHLTTriggerDomain+;
#pragma link C++ class AliHLTDomainEntry+;
#pragma link C++ class AliHLTTriggerMenu+;
#pragma link C++ class AliHLTTriggerMenuItem+;

// For old versions of the trigger menu item we need to set the missing values to appropriate defaults.
#pragma read sourceClass="AliHLTTriggerMenuItem" version="[1-3]" targetClass="AliHLTTriggerMenuItem"\
  source="" target=""\
  code="{\
    newObj->DefaultResult(true);\
    newObj->ScaleDown(1);\
  }"

#pragma read sourceClass="AliHLTHuffmanNode" version="[1-]" targetClass="AliHLTHuffmanNode"\
  source="float fWeight" target="fWeightDouble"\
  code="{\
    fWeightDouble = onfile.fWeight;\
  }"

#pragma link C++ class AliHLTTriggerMenuSymbol+;
#pragma link C++ class AliHLTRunStatistics+;
#pragma link C++ class AliHLTSpacePointContainer+;
#pragma link C++ class AliHLTIndexGrid<float, AliHLTSpacePointContainer::AliHLTSpacePointProperties>+;
#pragma link C++ class AliHLTIndexGrid<float, AliHLTUInt32_t>+;
#pragma link C++ class AliHLTIndexGrid<int, AliHLTUInt32_t>+;
#pragma link C++ class AliHLTTrackGeometry+;
#pragma link C++ class AliHLTMisc+;
#pragma link C++ class AliHLTCTPData+;

#pragma read sourceClass="AliHLTCTPData" targetClass="AliHLTCTPData" source="AliHLTUInt64_t fMask"  version="[-2]" target="fMask" targetType="AliHLTTriggerMask_t" include="Rtypes.h" code="{fMask.reset(); for(Int_t i=0; i<64; ++i) fMask.set(i,TESTBIT(onfile.fMask,i));}"
#pragma read sourceClass="AliHLTCTPData" targetClass="AliHLTCTPData" source="AliHLTUInt64_t fTriggers"  version="[-2]" target="fTriggers" targetType="AliHLTTriggerMask_t" include="Rtypes.h" code="{fTriggers.reset(); for(Int_t i=0; i<64; ++i) fTriggers.set(i,TESTBIT(onfile.fTriggers,i));}"

#pragma link C++ class AliHLTScalars+;
#pragma link C++ class AliHLTScalars::AliScalar+;

// Need to initialise the hash table which is transient after reading the class.
#pragma read sourceClass="AliHLTScalars" version="[1-]" targetClass="AliHLTScalars"\
  source="" target="fMap" code="{fMap.AddAll(newObj->GetScalars());}"

#pragma link C++ struct AliHLTComponentEventData+;
#pragma link C++ struct AliHLTComponentBlockData+;
#pragma link C++ struct AliHLTComponentDataType+;
#pragma link C++ struct AliHLTEventDDLV0+; // Only added to have proper dictionary generation and ROOT I/O for AliHLTReadoutList class.
#pragma link C++ struct AliHLTEventDDLV1+; // Only added to have proper dictionary generation and ROOT I/O for AliHLTReadoutList class.
#pragma link C++ struct AliHLTEventDDLV2+; // Only added to have proper dictionary generation and ROOT I/O for AliHLTReadoutList class.
#pragma link C++ struct AliHLTRunDesc+;
#pragma link C++ struct AliHLTComponentStatistics+;
#pragma link C++ struct AliHLTComponentTableEntry;

#pragma link C++ function operator==( const AliHLTComponentDataType&, const AliHLTComponentDataType&);
#pragma link C++ function operator!=( const AliHLTComponentDataType&, const AliHLTComponentDataType&);
#pragma link C++ function operator|(const AliHLTComponentDataType, const char*);
#pragma link C++ function AliHLTComponentDataTypeInitializer(const char*, const char*);
#pragma link C++ function AliHLTComponentDataTypeInitializer(const AliHLTComponentDataType, const char*);
#pragma link C++ function operator<<(ostream &, const AliHLTComponentDataType &);
#pragma link C++ function operator<<(ostream &, const AliHLTSpacePointContainer &);

#pragma link C++ global kAliHLTComponentDataTypefOriginSize;
#pragma link C++ global kAliHLTComponentDataTypefIDsize;

#pragma link C++ global kAliHLTDataOriginVoid;
#pragma link C++ global kAliHLTDataOriginAny;
#pragma link C++ global kAliHLTDataOriginOut;
#pragma link C++ global kAliHLTDataOriginHLT;
#pragma link C++ global kAliHLTDataOriginOffline;
#pragma link C++ global kAliHLTDataOriginPrivate;
#pragma link C++ global kAliHLTDataOriginTPC;
#pragma link C++ global kAliHLTDataOriginPHOS;
#pragma link C++ global kAliHLTDataOriginFMD;
#pragma link C++ global kAliHLTDataOriginMUON;
#pragma link C++ global kAliHLTDataOriginTRD;
#pragma link C++ global kAliHLTDataOriginITS;
#pragma link C++ global kAliHLTDataOriginITSSPD;
#pragma link C++ global kAliHLTDataOriginITSSDD;
#pragma link C++ global kAliHLTDataOriginITSSSD;
#pragma link C++ global kAliHLTDataOriginSample;
#pragma link C++ global kAliHLTDataOriginEMCAL;

#pragma link C++ global kAliHLTAnyDataType;
#pragma link C++ global kAliHLTAllDataType;
#pragma link C++ global kAliHLTVoidDataType;
#pragma link C++ global kAliHLTMultipleDataType;
#pragma link C++ global kAliHLTDataTypeFXSCalib;
#pragma link C++ global kAliHLTDataTypeDDL;
#pragma link C++ global kAliHLTDataTypeDAQRDOUT;
#pragma link C++ global kAliHLTDataTypeClusters;
#pragma link C++ global kAliHLTDataTypeSOR;
#pragma link C++ global kAliHLTDataTypeEOR;
#pragma link C++ global kAliHLTDataTypeRunType;
#pragma link C++ global kAliHLTDataTypeEvent;
#pragma link C++ global kAliHLTDataTypeECSParam;
#pragma link C++ global kAliHLTDataTypeComConf;
#pragma link C++ global kAliHLTDataTypeConfig;
#pragma link C++ global kAliHLTDataTypeInfo;
#pragma link C++ global kAliHLTDataTypeCDBEntry;
#pragma link C++ global kAliHLTDataTypeUpdtDCS;
#pragma link C++ global kAliHLTDataTypeDDLRaw;
#pragma link C++ global kAliHLTDataTypeMCObject;
#pragma link C++ global kAliHLTDataTypeESDObject;
#pragma link C++ global kAliHLTDataTypeESDTree;
#pragma link C++ global kAliHLTDataTypeAliTreeD;
#pragma link C++ global kAliHLTDataTypeAliTreeR;
#pragma link C++ global kAliHLTDataTypeHwAddr16;
#pragma link C++ global kAliHLTDataTypeEventStatistics;
#pragma link C++ global kAliHLTDataTypeEventSummary;
#pragma link C++ global kAliHLTDataTypeRunStatistics;
#pragma link C++ global kAliHLTDataTypeRunSummary;
#pragma link C++ global kAliHLTDataTypeTriggerDecision;
#pragma link C++ global kAliHLTDataTypeGlobalTrigger;
#pragma link C++ global kAliHLTDataTypeComponentStatistics;
#pragma link C++ global kAliHLTDataTypeComponentTable;
#pragma link C++ global kAliHLTDataTypeTObject;
#pragma link C++ global kAliHLTDataTypeTObjArray;
#pragma link C++ global kAliHLTDataTypeTTree;
#pragma link C++ global kAliHLTDataTypeHistogram;
#pragma link C++ global kAliHLTDataTypeTNtuple;
#pragma link C++ global kAliHLTDataTypeTrack;
#pragma link C++ global kAliHLTDataTypeTrackMC;
#pragma link C++ global kAliHLTDataTypeExternalTrackParam;
#pragma link C++ global kAliHLTDataTypeJet;

#pragma link C++ global kAliHLTVoidEventID;
#pragma link C++ global kAliHLTVoidDataSpec;
#pragma link C++ global kAliHLTVoidRunNo;
#pragma link C++ global kAliHLTVoidRunType;
#pragma link C++ global kAliHLTVoidRunDesc;

#pragma link C++ global gkAliEventTypeUnknown;
#pragma link C++ global gkAliEventTypeStartOfRun;
#pragma link C++ global gkAliEventTypeData;
#pragma link C++ global gkAliEventTypeEndOfRun;
#pragma link C++ global gkAliEventTypeCorruptID;
#pragma link C++ global gkAliEventTypeCalibration;
#pragma link C++ global gkAliEventTypeDataReplay;
#pragma link C++ global gkAliEventTypeConfiguration;
#pragma link C++ global gkAliEventTypeReadPreprocessor;
#pragma link C++ global gkAliEventTypeTick;
#pragma link C++ global gkAliEventTypeMax;

#pragma link C++ global kHLTLogNone;
#pragma link C++ global kHLTLogBenchmark;
#pragma link C++ global kHLTLogDebug;
#pragma link C++ global kHLTLogInfo;
#pragma link C++ global kHLTLogWarning;
#pragma link C++ global kHLTLogError;
#pragma link C++ global kHLTLogFatal;
#pragma link C++ global kHLTLogImportant;
#pragma link C++ global kHLTLogAll;
#pragma link C++ global kHLTLogDefault;

#pragma link C++ class AliHLTOUTComponent+;
#endif
