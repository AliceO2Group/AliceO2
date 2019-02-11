#ifdef __CINT__
#include <RVersion.h>

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class AliRawVEvent+;
#pragma link C++ class AliRawEvent-;
#pragma link C++ class AliRawEventV2+;
#pragma link C++ class AliRawEventHeaderBase+;
#pragma link C++ class AliRawEquipmentHeader;
#pragma link C++ class AliRawVEquipment+;
#pragma link C++ class AliRawEquipment-;
#pragma link C++ class AliRawEquipmentV2+;
#pragma link C++ class AliRawData;
#pragma link C++ class AliRawDataArrayV2+;
#pragma link C++ class AliRawDataArray;
#pragma link C++ class AliStats;
#pragma link C++ class AliRawEventTag+;
#pragma link C++ class AliAltroMapping+;
#pragma link C++ class AliCaloAltroMapping+;

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
#pragma link C++ class AliRawEventHeaderV3_11+;
#pragma link C++ class AliRawEventHeaderV3_12+;
#pragma link C++ class AliRawEventHeaderV3_13+;
#pragma link C++ class AliRawEventHeaderV3_9+;
#pragma link C++ class AliRawEventHeaderV3_14+;
#else
#pragma link C++ defined_in AliRawEventHeaderVersions.h;
#endif

#endif
