//The main O2 data header, v0.1
//origin: CWG4
//blame: Mikolaj Krzewicki, mkrzewic@cern.ch
//       Matthias Richter, Matthias.Richter@cern.ch
//       David Rohr, drohr@cern.ch

#ifndef ALICEO2_BASE_DATA_HEADER_
#define ALICEO2_BASE_DATA_HEADER_

#include <cstdint>
#include <cstring>
#include <cstdio>

namespace AliceO2 {
namespace Base {

//constant field lengths for char fields
const uint32_t gSizeMagicString = 4;
const uint32_t gSizeDataOriginString = 4;
const uint32_t gSizePayloadSerializationString = 8;
const uint32_t gSizeDataDescriptionString = 16;

struct DataHeader;
struct DataOrigin;
struct DataDescription;
struct PayloadSerialization;

//____________________________________________________________________________
//the main header struct
struct DataHeader
{
  //other constants
  static const uint32_t sVersion = 1;
  static const char* sMagicString;

  //__the data layout:
  
  //a magic string
  union {
    char     magicString[gSizeMagicString];
    uint32_t  magicStringInt;
  };
  
  //origin of the data (originating detector)
  union {
    char     dataOrigin[gSizeDataOriginString];
    uint32_t  dataOriginInt;
  };

  //serialization method
  union {
    char     payloadSerialization[gSizePayloadSerializationString];
    uint64_t  payloadSerializationInt;
  };
  
  //data type descriptor
  union {
    char     dataDescription[gSizeDataDescriptionString];
    uint64_t  dataDescriptionInt[2];
  };

  //sub specification (e.g. link number)
  uint64_t    subSpecification;

  //flags, first bit indicates that a sub header follows
  uint32_t    flags;

  uint32_t    headerVersion;  //version of this header
  uint32_t    headerSize;     //size of this header
  uint32_t    payloadSize;    //size of the associated data

  //___NEVER MODIFY THE ABOVE
  //___NEW STUFF GOES BELOW

  //___the functions:
  DataHeader(); //ctor
  DataHeader(const DataHeader&); //copy ctor
  DataHeader& operator=(const DataHeader&); //assignment
  DataHeader& operator=(const DataOrigin&); //assignment
  DataHeader& operator=(const DataDescription&); //assignment
  DataHeader& operator=(const PayloadSerialization&); //assignment
  bool operator==(const DataHeader&); //comparison
  bool operator==(const DataOrigin&); //comparison
  bool operator==(const DataDescription&); //comparison
  bool operator==(const PayloadSerialization&); //comparison
  void print() const;
};

//____________________________________________________________________________
struct DataOrigin
{
  //origin of the data (originating detector)
  union {
    char     dataOrigin[gSizeDataOriginString];
    uint32_t  dataOriginInt;
  };
  DataOrigin(const char* origin)
    //: dataOriginInt(*(reinterpret_cast<const uint32_t*>(origin))) {}
    : dataOrigin() {
      memset(dataOrigin, ' ', gSizeDataOriginString-1);
      if (origin) {
        strncpy(dataOrigin, origin, gSizeDataOriginString-1);
      }
      dataOrigin[gSizeDataOriginString-1] = '\0';
    }
  void print() const {printf("Data origin  : %s\n", dataOrigin);}
};

//____________________________________________________________________________
struct DataDescription
{
  //data type descriptor
  union {
    char     dataDescription[gSizeDataDescriptionString];
    uint64_t  dataDescriptionInt[2];
  };
  DataDescription(const char* desc)
    //: dataDescriptionInt { (reinterpret_cast<const uint64_t*>(desc))[0],
    //                       (reinterpret_cast<const uint64_t*>(desc))[1]
    //                     }  {}
    : dataDescription() {
      memset(dataDescription, ' ', gSizeDataDescriptionString-1);
      if (desc) {
        strncpy(dataDescription, desc, gSizeDataDescriptionString-1);
      }
      dataDescription[gSizeDataDescriptionString-1] = '\0';
    }
  void print() const {printf("Data descr.  : %s\n", dataDescription);}
};

//____________________________________________________________________________
struct PayloadSerialization
{
  //serialization method
  union {
    char     payloadSerialization[gSizePayloadSerializationString];
    uint64_t  payloadSerializationInt;
  };
  PayloadSerialization(const char* serialization)
    //: payloadSerializationInt(*(reinterpret_cast<const uint64_t*>(serialization))) {}
    : payloadSerialization() {
      memset(payloadSerialization, ' ', gSizePayloadSerializationString-1);
      if (serialization) {
        strncpy(payloadSerialization, serialization, gSizePayloadSerializationString-1);
      }
      payloadSerialization[gSizePayloadSerializationString-1] = '\0';
    }
  void print() const {printf("Serialization: %s\n", payloadSerialization);}
};

//____________________________________________________________________________
//possible data origins
extern const DataOrigin gDataOriginAny;
extern const DataOrigin gDataOriginInvalid;
extern const DataOrigin gDataOriginTPC;
extern const DataOrigin gDataOriginTRD;
extern const DataOrigin gDataOriginTOF;

//____________________________________________________________________________
//possible data types
extern const DataDescription gDataDescriptionAny;
extern const DataDescription gDataDescriptionInvalid;
extern const DataDescription gDataDescriptionRaw;
extern const DataDescription gDataDescriptionClusters;
extern const DataDescription gDataDescriptionTracks;

//____________________________________________________________________________
//possible serialization types
extern const PayloadSerialization gPayloadSerializationAny;
extern const PayloadSerialization gPayloadSerializationInvalid;
extern const PayloadSerialization gPayloadSerializationNone;
extern const PayloadSerialization gPayloadSerializationROOT;
extern const PayloadSerialization gPayloadSerializationFlatBuf;

} //namespace Base
} //namespace AliceO2

#endif
