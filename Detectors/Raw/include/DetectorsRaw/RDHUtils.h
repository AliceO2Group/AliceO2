// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "CommonDataFormat/InteractionRecord.h"
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace raw
{
using LinkSubSpec_t = uint32_t;
using IR = o2::InteractionRecord;

struct RDHUtils {
  using RDHDef = o2::header::RAWDataHeader; // wathever is default
  //using RDHv3 = o2::header::RAWDataHeaderV3; // V3 == V4
  using RDHv4 = o2::header::RAWDataHeaderV4;
  using RDHv5 = o2::header::RAWDataHeaderV5;
  using RDHv6 = o2::header::RAWDataHeaderV6;
  static constexpr uint8_t MaxRDHVersion = 6;

  static constexpr int GBTWord = 16; // length of GBT word
  static constexpr int MAXCRUPage = 512 * GBTWord;

  GPUhdi() static uint8_t getVersion(const RDHv4& rdh) { return rdh.version; } // same for all // why template does not work here?
  GPUhdi() static uint8_t getVersion(const RDHv5& rdh) { return rdh.version; } // same for all
  GPUhdi() static uint8_t getVersion(const RDHv6& rdh) { return rdh.version; } // same for all
  GPUhdi() static uint8_t getVersion(const void* rdhP)
  {
    auto v = getVersion((*reinterpret_cast<const RDHDef*>(rdhP)));
    if (v > MaxRDHVersion) {
      processError(v, "version");
    }
    return v;
  }
  template <typename RDH>
  static void setVersion(RDH& rdh, uint8_t v)
  {
    rdh.version = v;
  } // same for all
  static void setVersion(void* rdhP, uint8_t v) { setVersion(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  static int getHeaderSize(const RDH& rdh)
  {
    return rdh.headerSize;
  } // same for all
  static int getHeaderSize(const void* rdhP) { return getHeaderSize(*reinterpret_cast<const RDHDef*>(rdhP)); }

  template <typename RDH>
  GPUhdi() static uint16_t getFEEID(const RDH& rdh)
  {
    return rdh.feeId;
  } // same for all
  GPUhdi() static uint16_t getFEEID(const void* rdhP) { return getFEEID(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setFEEID(RDH& rdh, uint16_t v)
  {
    rdh.feeId = v;
  } // same for all
  static void setFEEID(void* rdhP, uint16_t v) { setFEEID(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  static bool getPriorityBit(const RDH& rdh)
  {
    return rdh.priority;
  } // same for all
  static bool getPriorityBit(const void* rdhP) { return getPriorityBit(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setPriorityBit(RDH& rdh, bool v)
  {
    rdh.priority = v;
  } // same for all
  static void setPriorityBit(void* rdhP, bool v) { setPriorityBit(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  GPUhdi() static uint8_t getSourceID(const RDH& rdh)
  { // does not exist before V6
    processError(getVersion(rdh), "sourceID");
    return 0xff;
  }
  GPUhdi() static uint8_t getSourceID(const RDHv6& rdh) { return rdh.sourceID; }
  GPUhdi() static uint8_t getSourceID(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 5) {
      return getSourceID(*reinterpret_cast<const RDHv6*>(rdhP));
    } else {
      processError(version, "sourceID");
      return 0xff;
    }
  }
  static void setSourceID(RDHv6& rdh, uint8_t s) { rdh.sourceID = s; }
  static void setSourceID(void* rdhP, uint8_t s)
  {
    int version = getVersion(rdhP);
    if (version > 5) {
      setSourceID(*reinterpret_cast<RDHv6*>(rdhP), s);
    } else {
      processError(version, "sourceID");
    }
  }

  template <typename RDH>
  static uint16_t getOffsetToNext(const RDH& rdh)
  {
    return rdh.offsetToNext;
  } // same for all
  static uint16_t getOffsetToNext(const void* rdhP) { return getOffsetToNext(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setOffsetToNext(RDH& rdh, uint16_t v)
  {
    rdh.offsetToNext = v;
  } // same for all
  static void setOffsetToNext(void* rdhP, uint16_t v) { setOffsetToNext(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  static uint16_t getMemorySize(const RDH& rdh)
  {
    return rdh.memorySize;
  } // same for all
  static uint16_t getMemorySize(const void* rdhP) { return getMemorySize(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setMemorySize(RDH& rdh, uint16_t v)
  {
    rdh.offsetToNext = v;
  } // same for all
  static void setMemorySize(void* rdhP, uint16_t v) { setMemorySize(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  static uint8_t getLinkID(const RDH& rdh)
  {
    return rdh.linkID;
  } // same for all
  static uint8_t getLinkID(const void* rdhP) { return getLinkID(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setLinkID(RDH& rdh, uint8_t v)
  {
    rdh.linkID = v;
  } // same for all
  static void setLinkID(void* rdhP, uint8_t v) { setLinkID(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  static uint8_t getPacketCounter(const RDH& rdh)
  {
    return rdh.packetCounter;
  } // same for all
  static uint8_t getPacketCounter(const void* rdhP) { return getPacketCounter(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setPacketCounter(RDH& rdh, uint8_t v)
  {
    rdh.packetCounter = v;
  } // same for all
  static void setPacketCounter(void* rdhP, uint8_t v) { setPacketCounter(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  static uint16_t getCRUID(const RDH& rdh)
  {
    return rdh.cruID;
  } // same for all
  static uint16_t getCRUID(const void* rdhP) { return getCRUID(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setCRUID(RDH& rdh, uint16_t v)
  {
    rdh.cruID = v;
  } // same for all
  static void setCRUID(void* rdhP, uint16_t v) { setCRUID(*reinterpret_cast<RDHDef*>(rdhP), v); }

  template <typename RDH>
  static uint8_t getEndPointID(const RDH& rdh)
  {
    return rdh.endPointID;
  } // same for all
  static uint8_t getEndPointID(const void* rdhP) { return getEndPointID(*reinterpret_cast<const RDHDef*>(rdhP)); }
  template <typename RDH>
  static void setEndPointID(RDH& rdh, uint8_t v)
  {
    rdh.endPointID = v;
  } // same for all
  static void setEndPointID(void* rdhP, uint8_t v) { setEndPointID(*reinterpret_cast<RDHDef*>(rdhP), v); }

  GPUhdi() static uint16_t getHeartBeatBC(const RDHv4& rdh) { return rdh.heartbeatBC; }
  GPUhdi() static uint16_t getHeartBeatBC(const RDHv5& rdh) { return rdh.bunchCrossing; } // starting from V5 no distiction trigger or HB
  GPUhdi() static uint16_t getHeartBeatBC(const RDHv6& rdh) { return rdh.bunchCrossing; }
  GPUhdi() static uint16_t getHeartBeatBC(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getHeartBeatBC(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getHeartBeatBC(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  static void setHeartBeatBC(RDHv4& rdh, uint16_t v) { rdh.heartbeatBC = v; }
  static void setHeartBeatBC(RDHv5& rdh, uint16_t v) { rdh.bunchCrossing = v; } // starting from V5 no distiction trigger or HB
  static void setHeartBeatBC(RDHv6& rdh, uint16_t v) { rdh.bunchCrossing = v; }
  static void setHeartBeatBC(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setHeartBeatBC(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setHeartBeatBC(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  GPUhdi() static uint32_t getHeartBeatOrbit(const RDHv4& rdh) { return rdh.heartbeatOrbit; }
  GPUhdi() static uint32_t getHeartBeatOrbit(const RDHv5& rdh) { return rdh.orbit; } // starting from V5 no distiction trigger or HB
  GPUhdi() static uint32_t getHeartBeatOrbit(const RDHv6& rdh) { return rdh.orbit; }
  GPUhdi() static uint32_t getHeartBeatOrbit(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getHeartBeatOrbit(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getHeartBeatOrbit(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  static void setHeartBeatOrbit(RDHv4& rdh, uint32_t v) { rdh.heartbeatOrbit = v; }
  static void setHeartBeatOrbit(RDHv5& rdh, uint32_t v) { rdh.orbit = v; } // starting from V5 no distiction trigger or HB
  static void setHeartBeatOrbit(RDHv6& rdh, uint32_t v) { rdh.orbit = v; }
  static void setHeartBeatOrbit(void* rdhP, uint32_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setHeartBeatOrbit(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setHeartBeatOrbit(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  template <typename RDH>
  static IR getHeartBeatIR(const RDH& rdh)
  {
    return {getHeartBeatBC(rdh), getHeartBeatOrbit(rdh)};
  } // custom extension
  static IR getHeartBeattIR(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getHeartBeatIR(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getHeartBeatIR(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }

  template <typename RDH>
  static IR getTriggerIR(const RDH& rdh)
  {
    return {getTriggerBC(rdh), getTriggerOrbit(rdh)};
  } // custom extension
  static IR getTriggerIR(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerIR(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getTriggerIR(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }

  static uint16_t getTriggerBC(const RDHv4& rdh) { return rdh.triggerBC; }
  static uint16_t getTriggerBC(const RDHv5& rdh) { return rdh.bunchCrossing; } // starting from V5 no distiction trigger or HB
  static uint16_t getTriggerBC(const RDHv6& rdh) { return rdh.bunchCrossing; }
  static uint16_t getTriggerBC(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerBC(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getTriggerBC(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  static void setTriggerBC(RDHv4& rdh, uint16_t v) { rdh.triggerBC = v; }
  static void setTriggerBC(RDHv5& rdh, uint16_t v) { rdh.bunchCrossing = v; } // starting from V5 no distiction trigger or HB
  static void setTriggerBC(RDHv6& rdh, uint16_t v) { rdh.bunchCrossing = v; }
  static void setTriggerBC(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setTriggerBC(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setTriggerBC(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  static uint32_t getTriggerOrbit(const RDHv4& rdh) { return rdh.triggerOrbit; }
  static uint32_t getTriggerOrbit(const RDHv5& rdh) { return rdh.orbit; } // starting from V5 no distiction trigger or HB
  static uint32_t getTriggerOrbit(const RDHv6& rdh) { return rdh.orbit; }
  static uint32_t getTriggerOrbit(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerOrbit(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getTriggerOrbit(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  static void setTriggerOrbit(RDHv4& rdh, uint32_t v) { rdh.triggerOrbit = v; }
  static void setTriggerOrbit(RDHv5& rdh, uint32_t v) { rdh.orbit = v; } // starting from V5 no distiction trigger or HB
  static void setTriggerOrbit(RDHv6& rdh, uint32_t v) { rdh.orbit = v; }
  static void setTriggerOrbit(void* rdhP, uint32_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setTriggerOrbit(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setTriggerOrbit(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  template <typename RDH>
  static uint32_t getTriggerType(const RDH& rdh)
  {
    return rdh.triggerType;
  }
  static uint32_t getTriggerType(const RDHv5& rdh) { return rdh.triggerType; } // same name but different positions starting from v5
  static uint32_t getTriggerType(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getTriggerType(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getTriggerType(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  template <typename RDH>
  static void setTriggerType(RDH& rdh, uint32_t v)
  {
    rdh.triggerType = v;
  }
  static void setTriggerType(RDHv5& rdh, uint32_t v) { rdh.triggerType = v; } // same name but different positions starting from v5
  static void setTriggerType(void* rdhP, uint32_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setTriggerType(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setTriggerType(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  template <typename RDH>
  static uint16_t getPageCounter(const RDH& rdh)
  {
    return rdh.pageCnt;
  } // same name but different positions from V4
  static uint16_t getPageCounter(const RDHv5& rdh) { return rdh.pageCnt; }
  static uint16_t getPageCounter(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getPageCounter(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getPageCounter(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  template <typename RDH>
  static void setPageCounter(RDH& rdh, uint16_t v)
  {
    rdh.pageCnt = v;
  } // same name but different positions from V4
  static void setPageCounter(RDHv5& rdh, uint16_t v) { rdh.pageCnt = v; }
  static void setPageCounter(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setPageCounter(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setPageCounter(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  template <typename RDH>
  static uint32_t getDetectorField(const RDH& rdh)
  {
    return rdh.detectorField;
  }                                                                                                                       // same for all
  static uint32_t getDetectorField(const void* rdhP) { return getDetectorField(*reinterpret_cast<const RDHDef*>(rdhP)); } // same for all
  template <typename RDH>
  static void setDetectorField(RDH& rdh, uint32_t v)
  {
    rdh.detectorField = v;
  }                                                                                                               // same for all
  static void setDetectorField(void* rdhP, uint32_t v) { setDetectorField(*reinterpret_cast<RDHDef*>(rdhP), v); } // same for all

  static uint16_t getDetectorPAR(const RDHv4& rdh) { return rdh.par; }
  static uint16_t getDetectorPAR(const RDHv5& rdh) { return rdh.detectorPAR; } // different names starting from V5
  static uint16_t getDetectorPAR(const RDHv6& rdh) { return rdh.detectorPAR; }
  static uint16_t getDetectorPAR(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getDetectorPAR(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getDetectorPAR(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  static void setDetectorPAR(RDHv4& rdh, uint16_t v) { rdh.par = v; }
  static void setDetectorPAR(RDHv5& rdh, uint16_t v) { rdh.detectorPAR = v; } // different names starting from V5
  static void setDetectorPAR(RDHv6& rdh, uint16_t v) { rdh.detectorPAR = v; }
  static void setDetectorPAR(void* rdhP, uint16_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setDetectorPAR(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setDetectorPAR(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  template <typename RDH>
  static uint8_t getStop(const RDH& rdh)
  {
    return rdh.stop;
  } // same name but different positions starting from v5
  static uint8_t getStop(const void* rdhP)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      return getStop(*reinterpret_cast<const RDHv5*>(rdhP));
    } else {
      return getStop(*reinterpret_cast<const RDHv4*>(rdhP));
    }
  }
  template <typename RDH>
  static void setStop(RDH& rdh, uint8_t v)
  {
    rdh.stop = v;
  } // same name but different positions starting from v5
  static void setStop(void* rdhP, uint8_t v)
  {
    int version = getVersion(rdhP);
    if (version > 4) {
      setStop(*reinterpret_cast<RDHv5*>(rdhP), v);
    } else {
      setStop(*reinterpret_cast<RDHv4*>(rdhP), v);
    }
  }

  static void printRDH(const RDHv4& rdh);
  static void printRDH(const RDHv5& rdh);
  static void printRDH(const RDHv6& rdh);
  static void printRDH(const void* rdhP);

  template <typename RDH>
  static void dumpRDH(const RDH& rdh)
  {
    dumpRDH(reinterpret_cast<const void*>(&rdh));
  }
  static void dumpRDH(const void* rdhP);

  static bool checkRDH(const RDHv4& rdh, bool verbose = true);
  static bool checkRDH(const RDHv5& rdh, bool verbose = true);
  static bool checkRDH(const RDHv6& rdh, bool verbose = true);
  static bool checkRDH(const void* rdhP, bool verbose = true);

  static LinkSubSpec_t getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint, uint16_t feeId);
  static LinkSubSpec_t getSubSpec(const RDHv4& rdh) { return getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID, rdh.feeId); }
  static LinkSubSpec_t getSubSpec(const RDHv5& rdh) { return getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID, rdh.feeId); }
  static LinkSubSpec_t getSubSpec(const RDHv6& rdh) { return getFEEID(rdh); }

 private:
  static uint32_t fletcher32(const uint16_t* data, int len);
#if defined(GPUCA_GPUCODE_DEVICE) || defined(GPUCA_STANDALONE)
  GPUhdi() static void processError(int v, const char* field)
  {
  }
#else
  GPUhdni() static void processError(int v, const char* field);
#endif

  ClassDefNV(RDHUtils, 1);
};

//_____________________________________________________________________
inline LinkSubSpec_t RDHUtils::getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint, uint16_t feeId)
{
  /*
  // RS Temporarily suppress this way since such a subspec does not define the TOF/TPC links in a unique way
  // define subspecification as in DataDistribution
  int linkValue = (LinkSubSpec_t(link) + 1) << (endpoint == 1 ? 8 : 0);
  return (LinkSubSpec_t(cru) << 16) | linkValue;
  */
  // RS Temporarily suppress this way since such a link is ambiguous
  uint16_t seq[3] = {cru, uint16_t((uint16_t(link) << 8) | endpoint), feeId};
  return fletcher32(seq, 3);
}

} // namespace raw
} // namespace o2

#endif //ALICEO2_RDHUTILS_H
