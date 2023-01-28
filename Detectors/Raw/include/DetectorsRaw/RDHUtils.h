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

// @brief Class for operations with RawDataHeader
// @author ruben.shahoyan@cern.ch

#ifndef ALICEO2_RDHUTILS_H
#define ALICEO2_RDHUTILS_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"
#include "GPUCommonTypeTraits.h"
#if !defined(GPUCA_GPUCODE)
#include "CommonDataFormat/InteractionRecord.h"
#endif
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "Headers/DAQID.h"
#endif // GPUCA_GPUCODE / GPUCA_STANDALONE

namespace o2
{
namespace raw
{
using LinkSubSpec_t = uint32_t;

struct RDHUtils {

// disable is the type is a pointer
#define NOTPTR(T) typename std::enable_if<!std::is_pointer<GPUgeneric() T>::value>::type* = 0
// dereference SRC pointer as DST type reference
#define TOREF(DST, SRC) *reinterpret_cast<DST*>(SRC)
// dereference SRC pointer as DST type const reference
#define TOCREF(DST, SRC) *reinterpret_cast<const DST*>(SRC)

  using RDHDef = o2::header::RAWDataHeader; // wathever is default
  using RDHAny = o2::header::RDHAny;
  using RDHv4 = o2::header::RAWDataHeaderV4; // V3 == V4
  using RDHv5 = o2::header::RAWDataHeaderV5;
  using RDHv6 = o2::header::RAWDataHeaderV6;
  using RDHv7 = o2::header::RAWDataHeaderV7; // update this for every new version

  static constexpr int GBTWord128 = 16; // length of GBT word
  static constexpr int MAXCRUPage = 512 * GBTWord128;
  /// get numeric version of the RDH

  ///_______________________________
  template <typename H>
  static constexpr int getVersion()
  {
#ifndef GPUCA_GPUCODE_DEVICE
    RDHAny::sanityCheckStrict<H>();
    if (std::is_same<H, RDHv7>::value) {
      return 7;
    }
    if (std::is_same<H, RDHv6>::value) {
      return 6;
    }
    if (std::is_same<H, RDHv5>::value) {
      return 5;
    }
    if (std::is_same<H, RDHv4>::value) {
      return 4;
    }
#else
    return -1; // dummy value as this method will be used on the CPU only
#endif // GPUCA_GPUCODE_DEVICE
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint8_t getVersion(const H& rdh, NOTPTR(H))
  {
    return rdh.version;
  } // same for all
  GPUhdi() static uint8_t getVersion(const RDHAny& rdh) { return getVersion(rdh.voidify()); }
  GPUhdi() static uint8_t getVersion(const void* rdhP) { return getVersion(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setVersion(H& rdh, uint8_t v, NOTPTR(H))
  {
    rdh.word0 = (v < 5 ? 0x0000ffff00004000 : 0x00000000ffff4000) + v;
  } // same for all (almost)
  static void setVersion(RDHAny& rdh, uint8_t v) { setVersion(rdh.voidify(), v); }
  static void setVersion(void* rdhP, uint8_t v) { setVersion(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  template <typename H>
  GPUhdi() static int getHeaderSize(const H& rdh, NOTPTR(H))
  {
    return rdh.headerSize;
  } // same for all
  GPUhdi() static int getHeaderSize(const RDHAny& rdh) { return getHeaderSize(rdh.voidify()); }
  GPUhdi() static int getHeaderSize(const void* rdhP) { return getHeaderSize(TOCREF(RDHDef, rdhP)); }

  ///_______________________________
  GPUhdi() static uint16_t getBlockLength(const RDHv4& rdh) { return rdh.blockLength; } // exists in v4 only
  GPUhdi() static uint16_t getBlockLength(const RDHAny& rdh) { return getBlockLength(rdh.voidify()); }
  GPUhdi() static uint16_t getBlockLength(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version == 4) {
      return getBlockLength(TOCREF(RDHv4, rdhP));
    } else {
      processError(getVersion(rdhP), "blockLength");
      return 0;
    }
  }
  static void setBlockLength(RDHv4& rdh, uint16_t s) { rdh.blockLength = s; }
  static void setBlockLength(RDHAny& rdh, uint16_t s) { setBlockLength(rdh.voidify(), s); }
  static void setBlockLength(void* rdhP, uint16_t s)
  {
    int version = getVersion(rdhP);
    if (version == 4) {
      setBlockLength(TOREF(RDHv4, rdhP), s);
    } else {
      processError(getVersion(rdhP), "blockLength");
    }
  }

  ///_______________________________
  GPUhdi() static uint16_t getFEEID(const RDHv4& rdh) { return rdh.feeId; } // same name differen position in v3,4
  template <typename H>
  GPUhdi() static uint16_t getFEEID(const H& rdh, NOTPTR(H))
  {
    return rdh.feeId;
  }
  GPUhdi() static uint16_t getFEEID(const RDHAny& rdh) { return getFEEID(rdh.voidify()); }
  GPUhdi() static uint16_t getFEEID(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getFEEID(TOCREF(RDHv5, rdhP));
    } else {
      return getFEEID(TOCREF(RDHv4, rdhP));
    }
  }
  static void setFEEID(RDHv4& rdh, uint16_t v)
  {
    rdh.feeId = v;
  }
  template <typename H>
  static void setFEEID(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.feeId = v;
  } //
  static void setFEEID(RDHAny& rdh, uint16_t v) { setFEEID(rdh.voidify(), v); }
  static void setFEEID(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setFEEID(TOREF(RDHv5, rdhP), v);
    } else {
      setFEEID(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static bool getPriorityBit(const H& rdh, NOTPTR(H))
  {
    return rdh.priority;
  } // same for all
  GPUhdi() static bool getPriorityBit(const RDHAny& rdh) { return getPriorityBit(rdh.voidify()); }
  GPUhdi() static bool getPriorityBit(const void* rdhP) { return getPriorityBit(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setPriorityBit(H& rdh, bool v, NOTPTR(H))
  {
    rdh.priority = v;
  } // same for all
  static void setPriorityBit(RDHAny& rdh, bool v) { setPriorityBit(rdh.voidify(), v); }
  static void setPriorityBit(void* rdhP, bool v) { setPriorityBit(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint8_t getSourceID(const H& rdh, NOTPTR(H))
  { // does not exist before V6
    processError(getVersion(rdh), "sourceID");
    return 0xff;
  }
  GPUhdi() static uint8_t getSourceID(const RDHv7& rdh) { return rdh.sourceID; }
  GPUhdi() static uint8_t getSourceID(const RDHv6& rdh) { return rdh.sourceID; }
  GPUhdi() static uint8_t getSourceID(const RDHAny& rdh) { return getSourceID(rdh.voidify()); }
  GPUhdi() static uint8_t getSourceID(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 5) {
      return getSourceID(TOCREF(RDHv6, rdhP));
    } else {
      processError(version, "sourceID");
      return 0xff;
    }
  }
  static void setSourceID(RDHv7& rdh, uint8_t s) { rdh.sourceID = s; }
  static void setSourceID(RDHv6& rdh, uint8_t s) { rdh.sourceID = s; }
  static void setSourceID(RDHAny& rdh, uint8_t s) { setSourceID(rdh.voidify(), s); }
  static void setSourceID(void* rdhP, uint8_t s)
  {
    int version = getVersion(rdhP);
    if (version > 5) {
      setSourceID(TOREF(RDHv6, rdhP), s);
    } else {
      processError(version, "sourceID");
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint16_t getOffsetToNext(const H& rdh, NOTPTR(H))
  {
    return rdh.offsetToNext;
  } // same for all
  GPUhdi() static uint16_t getOffsetToNext(const RDHAny& rdh) { return getOffsetToNext(rdh.voidify()); }
  GPUhdi() static uint16_t getOffsetToNext(const void* rdhP) { return getOffsetToNext(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setOffsetToNext(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.offsetToNext = v;
  } // same for all
  static void setOffsetToNext(RDHAny& rdh, uint16_t v) { setOffsetToNext(rdh.voidify(), v); }
  static void setOffsetToNext(void* rdhP, uint16_t v) { setOffsetToNext(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint16_t getMemorySize(const H& rdh, NOTPTR(H))
  {
    return rdh.memorySize;
  } // same for all
  GPUhdi() static uint16_t getMemorySize(const RDHAny& rdh) { return getMemorySize(rdh.voidify()); }
  GPUhdi() static uint16_t getMemorySize(const void* rdhP) { return getMemorySize(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setMemorySize(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.memorySize = v;
  } // same for all
  static void setMemorySize(RDHAny& rdh, uint16_t v) { setMemorySize(rdh.voidify(), v); }
  static void setMemorySize(void* rdhP, uint16_t v) { setMemorySize(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint8_t getLinkID(const H& rdh, NOTPTR(H))
  {
    return rdh.linkID;
  } // same for all
  GPUhdi() static uint8_t getLinkID(const RDHAny& rdh) { return getLinkID(rdh.voidify()); }
  GPUhdi() static uint8_t getLinkID(const void* rdhP) { return getLinkID(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setLinkID(H& rdh, uint8_t v, NOTPTR(H))
  {
    rdh.linkID = v;
  } // same for all
  static void setLinkID(RDHAny& rdh, uint8_t v) { setLinkID(rdh.voidify(), v); }
  static void setLinkID(void* rdhP, uint8_t v) { setLinkID(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint8_t getPacketCounter(const H& rdh, NOTPTR(H))
  {
    return rdh.packetCounter;
  } // same for all
  GPUhdi() static uint8_t getPacketCounter(const RDHAny& rdh) { return getPacketCounter(rdh.voidify()); }
  GPUhdi() static uint8_t getPacketCounter(const void* rdhP) { return getPacketCounter(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setPacketCounter(H& rdh, uint8_t v, NOTPTR(H))
  {
    rdh.packetCounter = v;
  } // same for all
  static void setPacketCounter(RDHAny& rdh, uint8_t v) { setPacketCounter(rdh.voidify(), v); }
  static void setPacketCounter(void* rdhP, uint8_t v) { setPacketCounter(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint16_t getCRUID(const H& rdh, NOTPTR(H))
  {
    return rdh.cruID;
  } // same for all
  GPUhdi() static uint16_t getCRUID(const RDHAny& rdh) { return getCRUID(rdh.voidify()); }
  GPUhdi() static uint16_t getCRUID(const void* rdhP) { return getCRUID(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setCRUID(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.cruID = v;
  } // same for all
  static void setCRUID(RDHAny& rdh, uint16_t v) { setCRUID(rdh.voidify(), v); }
  static void setCRUID(void* rdhP, uint16_t v) { setCRUID(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint8_t getEndPointID(const H& rdh, NOTPTR(H))
  {
    return rdh.endPointID;
  } // same for all
  GPUhdi() static uint8_t getEndPointID(const RDHAny& rdh) { return getEndPointID(rdh.voidify()); }
  GPUhdi() static uint8_t getEndPointID(const void* rdhP) { return getEndPointID(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setEndPointID(H& rdh, uint8_t v, NOTPTR(H))
  {
    rdh.endPointID = v;
  } // same for all
  static void setEndPointID(RDHAny& rdh, uint8_t v) { setEndPointID(rdh.voidify(), v); }
  static void setEndPointID(void* rdhP, uint8_t v) { setEndPointID(TOREF(RDHDef, rdhP), v); }

  ///_______________________________
  GPUhdi() static uint16_t getHeartBeatBC(const RDHv4& rdh) { return rdh.heartbeatBC; }
  template <typename H>
  GPUhdi() static uint16_t getHeartBeatBC(const H& rdh, NOTPTR(H))
  {
    return rdh.bunchCrossing;
  }                                                                                                    // starting from V5 no distiction trigger or HB
  GPUhdi() static uint16_t getHeartBeatBC(const RDHAny& rdh) { return getHeartBeatBC(rdh.voidify()); } // starting from V5 no distiction trigger or HB
  GPUhdi() static uint16_t getHeartBeatBC(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getHeartBeatBC(TOCREF(RDHv5, rdhP));
    } else {
      return getHeartBeatBC(TOCREF(RDHv4, rdhP));
    }
  }
  GPUhdi() static void setHeartBeatBC(RDHv4& rdh, uint16_t v) { rdh.heartbeatBC = v; }
  template <typename H>
  GPUhdi() static void setHeartBeatBC(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.bunchCrossing = v;
  } // starting from V5 no distiction trigger or HB
  GPUhdi() static void setHeartBeatBC(RDHAny& rdh, uint16_t v) { setHeartBeatBC(rdh.voidify(), v); }
  GPUhdi() static void setHeartBeatBC(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setHeartBeatBC(TOREF(RDHv5, rdhP), v);
    } else {
      setHeartBeatBC(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  GPUhdi() static uint32_t getHeartBeatOrbit(const RDHv4& rdh) { return rdh.heartbeatOrbit; }
  template <typename H>
  GPUhdi() static uint32_t getHeartBeatOrbit(const H& rdh, NOTPTR(H))
  {
    return rdh.orbit;
  } // starting from V5 no distiction trigger or HB
  GPUhdi() static uint32_t getHeartBeatOrbit(const RDHAny& rdh) { return getHeartBeatOrbit(rdh.voidify()); }
  GPUhdi() static uint32_t getHeartBeatOrbit(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getHeartBeatOrbit(TOCREF(RDHv5, rdhP));
    } else {
      return getHeartBeatOrbit(TOCREF(RDHv4, rdhP));
    }
  }
  static void setHeartBeatOrbit(RDHv4& rdh, uint32_t v) { rdh.heartbeatOrbit = v; }
  template <typename H>
  static void setHeartBeatOrbit(H& rdh, uint32_t v, NOTPTR(H))
  {
    rdh.orbit = v;
  } // starting from V5 no distiction trigger or HB
  static void setHeartBeatOrbit(RDHAny& rdh, uint32_t v) { setHeartBeatOrbit(rdh.voidify(), v); }
  static void setHeartBeatOrbit(void* rdhP, uint32_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setHeartBeatOrbit(TOREF(RDHv5, rdhP), v);
    } else {
      setHeartBeatOrbit(TOREF(RDHv4, rdhP), v);
    }
  }

#ifndef GPUCA_GPUCODE
  using IR = o2::InteractionRecord;
  ///_______________________________
  template <typename H>
  GPUhdi() static IR getHeartBeatIR(const H& rdh, NOTPTR(H))
  {
    return {getHeartBeatBC(rdh), getHeartBeatOrbit(rdh)};
  } // custom extension
  GPUhdi() static IR getHeartBeatIR(const RDHAny& rdh) { return getHeartBeatIR(rdh.voidify()); }
  GPUhdi() static IR getHeartBeatIR(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getHeartBeatIR(TOCREF(RDHv5, rdhP));
    } else {
      return getHeartBeatIR(TOCREF(RDHv4, rdhP));
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static IR getTriggerIR(const H& rdh, NOTPTR(H))
  {
    return {getTriggerBC(rdh), getTriggerOrbit(rdh)};
  } // custom extension
  GPUhdi() static IR getTriggerIR(const RDHAny& rdh) { return getTriggerIR(rdh.voidify()); }
  GPUhdi() static IR getTriggerIR(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerIR(TOCREF(RDHv5, rdhP));
    } else {
      return getTriggerIR(TOCREF(RDHv4, rdhP));
    }
  }
#endif

  ///_______________________________
  GPUhdi() static uint16_t getTriggerBC(const RDHv4& rdh) { return rdh.triggerBC; }
  template <typename H>
  GPUhdi() static uint16_t getTriggerBC(const H& rdh, NOTPTR(H))
  {
    return rdh.bunchCrossing;
  } // starting from V5 no distiction trigger or HB
  GPUhdi() static uint16_t getTriggerBC(const RDHAny& rdh) { return getTriggerBC(rdh.voidify()); }
  GPUhdi() static uint16_t getTriggerBC(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerBC(TOCREF(RDHv5, rdhP));
    } else {
      return getTriggerBC(TOCREF(RDHv4, rdhP));
    }
  }
  static void setTriggerBC(RDHv4& rdh, uint16_t v) { rdh.triggerBC = v; }
  template <typename H>
  static void setTriggerBC(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.bunchCrossing = v;
  } // starting from V5 no distiction trigger or HB
  static void setTriggerBC(RDHAny& rdh, uint16_t v) { setTriggerBC(rdh.voidify(), v); }
  static void setTriggerBC(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setTriggerBC(TOREF(RDHv5, rdhP), v);
    } else {
      setTriggerBC(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  GPUhdi() static uint32_t getTriggerOrbit(const RDHv4& rdh) { return rdh.triggerOrbit; }
  template <typename H>
  GPUhdi() static uint32_t getTriggerOrbit(const H& rdh, NOTPTR(H))
  {
    return rdh.orbit;
  } // starting from V5 no distiction trigger or HB
  GPUhdi() static uint32_t getTriggerOrbit(const RDHAny& rdh) { return getTriggerOrbit(rdh.voidify()); }
  GPUhdi() static uint32_t getTriggerOrbit(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerOrbit(TOCREF(RDHv5, rdhP));
    } else {
      return getTriggerOrbit(TOCREF(RDHv4, rdhP));
    }
  }
  static void setTriggerOrbit(RDHv4& rdh, uint32_t v) { rdh.triggerOrbit = v; }
  template <typename H>
  static void setTriggerOrbit(H& rdh, uint32_t v, NOTPTR(H))
  {
    rdh.orbit = v;
  } // starting from V5 no distiction trigger or HB
  static void setTriggerOrbit(RDHAny& rdh, uint32_t v) { setTriggerOrbit(rdh.voidify(), v); }
  static void setTriggerOrbit(void* rdhP, uint32_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setTriggerOrbit(TOREF(RDHv5, rdhP), v);
    } else {
      setTriggerOrbit(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint8_t getDataFormat(const H& rdh, NOTPTR(H))
  { // does not exist before V7 (Jan. 2023), but <V7 headers are backward compatible to DataFormat=0 (padding)
    return 0xff;
  }
  GPUhdi() static uint8_t getDataFormat(const RDHv7& rdh) { return rdh.dataFormat; }
  GPUhdi() static uint8_t getDataFormat(const RDHAny& rdh) { return getDataFormat(rdh.voidify()); }
  GPUhdi() static uint8_t getDataFormat(const void* rdhP) { return (getVersion(rdhP) > 6) ? getDataFormat(TOCREF(RDHv7, rdhP)) : 0; }
  static void setDataFormat(RDHv7& rdh, uint8_t s) { rdh.dataFormat = s; }
  static void setDataFormat(RDHAny& rdh, uint8_t s) { setDataFormat(rdh.voidify(), s); }
  static void setDataFormat(void* rdhP, uint8_t s)
  {
    int version = getVersion(rdhP);
    if (version > 6) {
      setDataFormat(TOREF(RDHv7, rdhP), s);
    } else {
      processError(version, "dataFormat");
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint32_t getTriggerType(const H& rdh, NOTPTR(H))
  {
    return rdh.triggerType;
  }
  GPUhdi() static uint32_t getTriggerType(const RDHv5& rdh) { return rdh.triggerType; } // same name but different positions starting from v5
  GPUhdi() static uint32_t getTriggerType(const RDHAny& rdh) { return getTriggerType(rdh.voidify()); }
  GPUhdi() static uint32_t getTriggerType(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerType(TOCREF(RDHv5, rdhP));
    } else {
      return getTriggerType(TOCREF(RDHv4, rdhP));
    }
  }
  template <typename H>
  static void setTriggerType(H& rdh, uint32_t v, NOTPTR(H))
  {
    rdh.triggerType = v;
  }
  static void setTriggerType(RDHv5& rdh, uint32_t v) { rdh.triggerType = v; } // same name but different positions starting from v5
  static void setTriggerType(RDHAny& rdh, uint32_t v) { setTriggerType(rdh.voidify(), v); }
  static void setTriggerType(void* rdhP, uint32_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setTriggerType(TOREF(RDHv5, rdhP), v);
    } else {
      setTriggerType(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint16_t getPageCounter(const H& rdh, NOTPTR(H))
  {
    return rdh.pageCnt;
  } // same name but different positions from V4
  GPUhdi() static uint16_t getPageCounter(const RDHv5& rdh) { return rdh.pageCnt; }
  GPUhdi() static uint16_t getPageCounter(const RDHAny& rdh) { return getPageCounter(rdh.voidify()); }
  GPUhdi() static uint16_t getPageCounter(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getPageCounter(TOCREF(RDHv5, rdhP));
    } else {
      return getPageCounter(TOCREF(RDHv4, rdhP));
    }
  }
  template <typename H>
  static void setPageCounter(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.pageCnt = v;
  } // same name but different positions from V4
  static void setPageCounter(RDHv5& rdh, uint16_t v) { rdh.pageCnt = v; }
  static void setPageCounter(RDHAny& rdh, uint16_t v) { setPageCounter(rdh.voidify(), v); }
  static void setPageCounter(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setPageCounter(TOREF(RDHv5, rdhP), v);
    } else {
      setPageCounter(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint32_t getDetectorField(const H& rdh, NOTPTR(H))
  {
    return rdh.detectorField;
  }                                                                                                                       // same for all
  GPUhdi() static uint32_t getDetectorField(const RDHAny& rdh) { return getDetectorField(rdh.voidify()); }
  GPUhdi() static uint32_t getDetectorField(const void* rdhP) { return getDetectorField(TOCREF(RDHDef, rdhP)); }
  template <typename H>
  static void setDetectorField(H& rdh, uint32_t v, NOTPTR(H))
  {
    rdh.detectorField = v;
  }                                                                                                               // same for all
  static void setDetectorField(RDHAny& rdh, uint32_t v) { setDetectorField(rdh.voidify(), v); }
  static void setDetectorField(void* rdhP, uint32_t v) { setDetectorField(TOREF(RDHDef, rdhP), v); } // same for all

  ///_______________________________
  GPUhdi() static uint16_t getDetectorPAR(const RDHv4& rdh) { return rdh.par; }
  template <typename H>
  GPUhdi() static uint16_t getDetectorPAR(const H& rdh, NOTPTR(H))
  {
    return rdh.detectorPAR;
  } // different names starting from V5
  GPUhdi() static uint16_t getDetectorPAR(const RDHAny& rdh) { return getDetectorPAR(rdh.voidify()); }
  GPUhdi() static uint16_t getDetectorPAR(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getDetectorPAR(TOCREF(RDHv5, rdhP));
    } else {
      return getDetectorPAR(TOCREF(RDHv4, rdhP));
    }
  }
  static void setDetectorPAR(RDHv4& rdh, uint16_t v) { rdh.par = v; }
  template <typename H>
  static void setDetectorPAR(H& rdh, uint16_t v, NOTPTR(H))
  {
    rdh.detectorPAR = v;
  } // different names starting from V5
  static void setDetectorPAR(RDHAny& rdh, uint16_t v) { setDetectorPAR(rdh.voidify(), v); }
  static void setDetectorPAR(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setDetectorPAR(TOREF(RDHv5, rdhP), v);
    } else {
      setDetectorPAR(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  template <typename H>
  GPUhdi() static uint8_t getStop(const H& rdh, NOTPTR(H))
  {
    return rdh.stop;
  }
  GPUhdi() static uint8_t getStop(const RDHv5& rdh)
  {
    return rdh.stop;
  } // same name but different positions starting from v5
  GPUhdi() static uint8_t getStop(const RDHAny& rdh) { return getStop(rdh.voidify()); }
  GPUhdi() static uint8_t getStop(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getStop(TOCREF(RDHv5, rdhP));
    } else {
      return getStop(TOCREF(RDHv4, rdhP));
    }
  }
  template <typename H>
  static void setStop(H& rdh, uint8_t v, NOTPTR(H))
  {
    rdh.stop = v;
  } // same name but different positions starting from v5
  static void setStop(RDHAny& rdh, uint8_t v) { setStop(rdh.voidify(), v); }
  static void setStop(void* rdhP, uint8_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setStop(TOREF(RDHv5, rdhP), v);
    } else {
      setStop(TOREF(RDHv4, rdhP), v);
    }
  }

  ///_______________________________
  static void printRDH(const RDHv4& rdh);
  static void printRDH(const RDHv5& rdh);
  static void printRDH(const RDHv6& rdh);
  static void printRDH(const RDHv7& rdh);
  static void printRDH(const RDHAny& rdh) { printRDH(rdh.voidify()); }
  static void printRDH(const void* rdhP);

  ///_______________________________
  template <typename H>
  static void dumpRDH(const H& rdh, NOTPTR(H))
  {
    dumpRDH(reinterpret_cast<const void*>(&rdh));
  }
  static void dumpRDH(const void* rdhP);

  ///_______________________________
  static bool checkRDH(const RDHv4& rdh, bool verbose = true);
  static bool checkRDH(const RDHv5& rdh, bool verbose = true);
  static bool checkRDH(const RDHv6& rdh, bool verbose = true);
  static bool checkRDH(const RDHv7& rdh, bool verbose = true);
  static bool checkRDH(const RDHAny rdh, bool verbose = true) { return checkRDH(rdh.voidify(), verbose); }
  static bool checkRDH(const void* rdhP, bool verbose = true);

  ///_______________________________
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  static LinkSubSpec_t getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint, uint16_t feeId, o2::header::DAQID::ID srcid = o2::header::DAQID::INVALID)
  {
    // Adapt the same definition as DataDistribution
    // meaningfull DAQ sourceID means that it comes from RDHv6, in this case we use feeID as a subspec
    if (srcid != o2::header::DAQID::INVALID) {
      return feeId;
    }
    //else { // this may lead to ambiguities
    //  int linkValue = (LinkSubSpec_t(link) + 1) << (endpoint == 1 ? 8 : 0);
    //  return (LinkSubSpec_t(cru) << 16) | linkValue;
    //}
    //
    // RS At the moment suppress getting the subspec as a hash
    uint16_t seq[3] = {cru, uint16_t((uint16_t(link) << 8) | endpoint), feeId};
    return fletcher32(seq, 3);
  }
  template <typename H>
  static LinkSubSpec_t getSubSpec(const H& rdh, NOTPTR(H)) // will be used for all RDH versions but >=6
  {
    return getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID, rdh.feeId, o2::header::DAQID::INVALID);
  }

  GPUhdi() static LinkSubSpec_t getSubSpec(const RDHv7& rdh) { return getFEEID(rdh); }
  GPUhdi() static LinkSubSpec_t getSubSpec(const RDHv6& rdh) { return getFEEID(rdh); }
  GPUhdi() static LinkSubSpec_t getSubSpec(const RDHAny& rdh) { return getSubSpec(rdh.voidify()); }
  GPUhdi() static LinkSubSpec_t getSubSpec(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 5) {
      return getSubSpec(TOCREF(RDHv6, rdhP));
    } else {
      return getSubSpec(TOCREF(RDHv4, rdhP));
    }
  }
#endif // GPUCA_GPUCODE / GPUCA_STANDALONE

 private:
  static uint32_t fletcher32(const uint16_t* data, int len);
#if defined(GPUCA_GPUCODE_DEVICE) || defined(GPUCA_STANDALONE)
  template <typename T>
  GPUhdi() static void processError(int v, const T* field)
  {
  }
#else
  GPUhdni() static void processError(int v, const char* field);
#endif

  ClassDefNV(RDHUtils, 1);
};

} // namespace raw
} // namespace o2

#endif //ALICEO2_RDHUTILS_H
