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

#ifndef ALICEO2_FIT_LOOKUPTABLE_H_
#define ALICEO2_FIT_LOOKUPTABLE_H_
////////////////////////////////////////////////
// Look Up Table FIT
//////////////////////////////////////////////

#include "CCDB/BasicCCDBManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <Rtypes.h>
#include <iostream>
#include <tuple>
#include <TSystem.h>
#include <map>
#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <algorithm>
namespace o2
{
namespace fit
{
struct EntryCRU { // This is specific struct for CRU entry
  int mLinkID;
  int mEndPointID;
  int mCRUID;
  int mFEEID;
  friend std::ostream& operator<<(std::ostream& os, const EntryCRU& entryCRU)
  {
    os << "LinkID: " << entryCRU.mLinkID << "|";
    os << "EndPointID: " << entryCRU.mEndPointID << "|";
    os << "CRUID: " << entryCRU.mCRUID << "|";
    os << "FEEID: " << entryCRU.mFEEID;
    return os;
  }
  void parse(const boost::property_tree::ptree& propertyTree)
  {
    mLinkID = propertyTree.get<int>("LinkID");
    mEndPointID = propertyTree.get<int>("EndPointID");
    mCRUID = propertyTree.get<int>("CRUID");
    mFEEID = propertyTree.get<int>("FEEID");
  }
  ClassDefNV(EntryCRU, 1);
};

struct HasherCRU {
  // Hash-function without any collisions due to technical bit size of fields:
  // RDH::EndPointID :  4 bits
  // RDH::LinkID : 8 bits
  std::size_t operator()(const EntryCRU& entryCRU) const
  {
    return (entryCRU.mLinkID << 4) | entryCRU.mEndPointID;
  }
};

struct ComparerCRU {
  bool operator()(const EntryCRU& entry1, const EntryCRU& entry2) const
  {
    return ((entry1.mLinkID << 4) | entry1.mEndPointID) == ((entry2.mLinkID << 4) | entry2.mEndPointID);
  }
};

struct EntryPM {
  EntryCRU mEntryCRU;
  int mLocalChannelID = 0;
  friend std::ostream& operator<<(std::ostream& os, const EntryPM& entryPM)
  {
    os << entryPM.mEntryCRU << "|";
    os << "LocalChID: " << entryPM.mLocalChannelID;
    return os;
  }
  ClassDefNV(EntryPM, 1);
};

inline bool operator<(EntryPM const& entryPM1, EntryPM const& entryPM2)
{
  auto comparer = [](const EntryPM& entryPM) -> decltype(auto) { return std::tie(entryPM.mEntryCRU.mEndPointID, entryPM.mEntryCRU.mLinkID, entryPM.mLocalChannelID); };
  return comparer(entryPM1) < comparer(entryPM2);
}

struct HasherPM {
  // Hash-function without any collisions due to technical bit size of fields:
  // RDH::EndPointID :  4 bits
  // EventData::ChannelID : 4 bits
  // RDH::LinkID : 8 bits
  std::size_t operator()(const EntryPM& entryPM) const
  {
    return (entryPM.mEntryCRU.mLinkID << 8) | (entryPM.mLocalChannelID << 4) | (entryPM.mEntryCRU.mEndPointID);
  }
};

struct ComparerPM {
  // Always true due to perfect hasher
  bool operator()(const EntryPM& entry1, const EntryPM& entry2) const
  {
    return ((entry1.mEntryCRU.mLinkID << 8) | (entry1.mLocalChannelID << 4) | (entry1.mEntryCRU.mEndPointID)) == ((entry2.mEntryCRU.mLinkID << 8) | (entry2.mLocalChannelID << 4) | (entry2.mEntryCRU.mEndPointID));
  }
};

struct EntryFEE {
  EntryCRU mEntryCRU;
  std::string mChannelID;      // ChannelID, string type because some entries containes N/A
  std::string mLocalChannelID; // Local channelID, string type because some entries containes N/A
  std::string mModuleType;     // PM, PM-LCS, TCM
  std::string mModuleName;
  std::string mBoardHV;
  std::string mChannelHV;
  std::string mSerialNumberMCP;
  std::string mCableHV;
  std::string mCableSignal;
  friend std::ostream& operator<<(std::ostream& os, const EntryFEE& entryFEE)
  {
    os << entryFEE.mEntryCRU << "|";
    os << "ChannelID: " << entryFEE.mChannelID << "|";
    os << "LocalChannelID: " << entryFEE.mLocalChannelID << "|";
    os << "ModuleType: " << entryFEE.mModuleType << "|";
    os << "ModuleName: " << entryFEE.mModuleName << "|";
    os << "HV board: " << entryFEE.mBoardHV << "|";
    os << "HV channel: " << entryFEE.mChannelHV << "|";
    os << "MCP S/N: " << entryFEE.mSerialNumberMCP << "|";
    os << "HV cable: " << entryFEE.mCableHV << "|";
    os << "signal cable: " << entryFEE.mCableSignal << "|";
    return os;
  }

  void parse(const boost::property_tree::ptree& propertyTree)
  {
    mEntryCRU.parse(propertyTree);
    mChannelID = propertyTree.get<std::string>("channel #");
    mLocalChannelID = propertyTree.get<std::string>("LocalChannelID");
    mModuleType = propertyTree.get<std::string>("ModuleType");
    mModuleName = propertyTree.get<std::string>("Module");
    mBoardHV = propertyTree.get<std::string>("HV board");
    mChannelHV = propertyTree.get<std::string>("HV channel");
    mSerialNumberMCP = propertyTree.get<std::string>("MCP S/N");
    mCableHV = propertyTree.get<std::string>("HV cable");
    mCableSignal = propertyTree.get<std::string>("signal cable");
  }
  ClassDefNV(EntryFEE, 1);
};
enum class EModuleType : int { kUnknown,
                               kPM,
                               kPM_LCS,
                               kTCM };

template <typename MapEntryCRU2ModuleType = std::unordered_map<EntryCRU, EModuleType, HasherCRU, ComparerCRU>,
          typename MapEntryPM2ChannelID = std::unordered_map<EntryPM, int, HasherPM, ComparerPM>,
          typename = typename std::enable_if_t<std::is_integral<typename MapEntryPM2ChannelID::mapped_type>::value>>
class LookupTableBase
{
 public:
  typedef std::vector<EntryFEE> Table_t;
  typedef MapEntryPM2ChannelID MapEntryPM2ChannelID_t;
  typedef MapEntryCRU2ModuleType MapEntryCRU2ModuleType_t;
  typedef typename MapEntryPM2ChannelID_t::key_type EntryPM_t;
  typedef typename MapEntryCRU2ModuleType_t::key_type EntryCRU_t;
  typedef typename MapEntryPM2ChannelID_t::mapped_type ChannelID_t;
  typedef std::map<ChannelID_t, EntryPM_t> MapChannelID2EntryPM_t;  // for digit2raw
  typedef std::map<EModuleType, EntryCRU_t> MapModuleType2EntryCRU; // for digit2raw
  typedef EntryPM_t Topo_t;                                         // temporary for common interface

  LookupTableBase() = default;
  LookupTableBase(const Table_t& vecEntryFEE) { initFromTable(vecEntryFEE); }
  LookupTableBase(const std::string& pathToFile) { initFromFile(pathToFile); }
  LookupTableBase(const std::string& urlCCDB, const std::string& pathToStorageInCCDB, long timestamp = -1) { initCCDB(urlCCDB, pathToStorageInCCDB, timestamp); }
  // Map of str module names -> enum types
  const std::map<std::string, EModuleType> mMapModuleTypeStr2Enum = {{"PM", EModuleType::kPM}, {"PM-LCS", EModuleType::kPM_LCS}, {"TCM", EModuleType::kTCM}};
  // Warning! To exclude double mapping do not use isTCM and isPM in the same time
  bool isTCM(int linkID, int epID) const
  {
    return mEntryCRU_TCM.mLinkID == linkID && mEntryCRU_TCM.mEndPointID == epID;
  }

  bool isPM(int linkID, int epID) const
  {
    return isPM(EntryCRU_t{linkID, epID});
  }

  bool isTCM(const EntryCRU_t& entryCRU) const
  {
    if (getModuleType(entryCRU) == EModuleType::kTCM) {
      return true;
    } else {
      return false;
    }
  }
  bool isPM(const EntryCRU_t& entryCRU) const
  {
    if (getModuleType(entryCRU) == EModuleType::kPM || getModuleType(entryCRU) == EModuleType::kPM_LCS) {
      return true;
    } else {
      return false;
    }
  }
  EModuleType getModuleType(const EntryCRU_t& entryCRU) const
  {
    const auto& mapEntries = getMapEntryCRU2ModuleType();
    const auto& it = mapEntries.find(entryCRU);
    if (it != mapEntries.end()) {
      return it->second;
    } else {
      return EModuleType::kUnknown;
    }
  }
  EModuleType getModuleType(const std::string& moduleType)
  {
    const auto& it = mMapModuleTypeStr2Enum.find(moduleType);
    if (it != mMapModuleTypeStr2Enum.end()) {
      return it->second;
    } else {
      return EModuleType::kUnknown;
    }
  }
  void initFromFile(const std::string& pathToFile)
  {
    std::string filepath{};
    if (pathToFile == "") {
      std::string inputDir;
      const char* aliceO2env = std::getenv("O2_ROOT");
      if (aliceO2env) {
        inputDir = aliceO2env;
      }
      inputDir += "/share/Detectors/FT0/files/";
      filepath = inputDir + "LookupTable_FT0.json";
      filepath = gSystem->ExpandPathName(filepath.data()); // Expand $(ALICE_ROOT) into real system path
    } else {
      filepath = pathToFile;
    }
    prepareEntriesFEE(filepath);
    prepareLUT();
  }
  void initCCDB(const std::string& urlCCDB, const std::string& pathToStorageInCCDB, long timestamp = -1)
  {
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL(urlCCDB);
    mVecEntryFEE = *(mgr.getForTimeStamp<Table_t>(pathToStorageInCCDB, timestamp));
    prepareLUT();
  }
  void initFromTable(const Table_t* vecEntryFEE)
  {
    mVecEntryFEE = *vecEntryFEE;
    prepareLUT();
  }
  ChannelID_t getGlobalChannelID(const EntryPM_t& entryPM, bool& isValid) const
  {
    const auto& it = mMapEntryPM2ChannelID.find(entryPM);
    if (it != mMapEntryPM2ChannelID.end()) {
      isValid = true;
      return it->second;
    } else {
      isValid = false;
      return -1;
    }
  }
  ChannelID_t getChannel(int linkID, int chID, int ep = 0)
  {
    return mMapEntryPM2ChannelID.find(std::move(EntryPM_t{EntryCRU_t{linkID, ep, 0, 0}, chID}))->second;
  }
  ChannelID_t getChannel(int linkID, int ep, int chID, bool& isValid)
  {
    const auto& it = mMapEntryPM2ChannelID.find(std::move(EntryPM_t{EntryCRU_t{linkID, ep, 0, 0}, chID}));
    if (it != mMapEntryPM2ChannelID.end()) {
      isValid = true;
      return it->second;
    } else {
      isValid = false;
      return -1;
    }
  }
  void prepareEntriesFEE(const std::string& pathToConfigFile)
  {
    boost::property_tree::ptree propertyTree;
    boost::property_tree::read_json(pathToConfigFile.c_str(), propertyTree);
    mVecEntryFEE = prepareEntriesFEE(propertyTree);
  }
  Table_t prepareEntriesFEE(const boost::property_tree::ptree& propertyTree)
  {
    Table_t vecEntryFEE;
    for (const auto& pairEntry : propertyTree) {
      const auto& propertyTreeSingle = pairEntry.second;
      EntryFEE entryFEE{};
      entryFEE.parse(propertyTreeSingle);
      vecEntryFEE.push_back(entryFEE);
    }
    return vecEntryFEE;
  }

  void prepareLUT()
  {
    mMapEntryCRU2ModuleType.clear();
    mMapEntryPM2ChannelID.clear();
    const auto& vecEntryFEE = getVecMetadataFEE();
    for (const auto entryFEE : vecEntryFEE) {
      EntryCRU_t entryCRU = entryFEE.mEntryCRU;
      std::string strModuleType = entryFEE.mModuleType;
      EModuleType moduleType = getModuleType(strModuleType);
      if (moduleType != EModuleType::kUnknown) {
        mMapEntryCRU2ModuleType.insert({entryCRU, moduleType});
      }
      if (moduleType == EModuleType::kPM || moduleType == EModuleType::kPM_LCS) {
        const std::string& strChannelID = entryFEE.mChannelID;
        const std::string& strLocalChannelID = entryFEE.mLocalChannelID;
        EntryPM_t entryPM{entryCRU, std::stoi(strLocalChannelID)};
        mMapEntryPM2ChannelID.insert({entryPM, std::stoi(strChannelID)});
      }
      if (moduleType == EModuleType::kTCM) {
        mEntryCRU_TCM = entryCRU;
      }
    }
  }
  void printFullMap() const
  {
    for (const auto& entry : mVecEntryFEE) {
      LOG(info) << entry;
    }
    /*
    std::cout<<std::endl<<"------------------------------------------------------------------------------"<<std::endl;
    for(const auto &entry:mMapEntryPM2ChannelID) {
      std::cout<<entry.first<<"| GlChID: "<<entry.second<<std::endl;
    }
    std::cout<<std::endl<<"------------------------------------------------------------------------------"<<std::endl;
    for(const auto &entry:mMapEntryCRU2ModuleType) {
      std::cout<<entry.first<<"| ModuleType: "<<static_cast<int>(entry.second)<<std::endl;
    }
    */
  }
  const Table_t& getVecMetadataFEE() const { return mVecEntryFEE; }
  const MapEntryCRU2ModuleType_t& getMapEntryCRU2ModuleType() const { return mMapEntryCRU2ModuleType; }
  const MapEntryPM2ChannelID_t& getMapEntryPM2ChannelID() const { return mMapEntryPM2ChannelID; }
  const EntryCRU_t& getEntryCRU_TCM() const { return mEntryCRU_TCM; }
  // Temporary
  // Making topo for FEE recognizing(Local channelID is supressed)
  static Topo_t makeGlobalTopo(const Topo_t& topo)
  {
    return Topo_t{topo.mEntryCRU, 0};
  }
  static int getLocalChannelID(const Topo_t& topo)
  {
    return topo.mLocalChannelID;
  }
  Topo_t getTopoPM(int globalChannelID) const
  {
    const auto& mapChannels = getMapEntryPM2ChannelID();
    auto findResult = std::find_if(mapChannels.begin(), mapChannels.end(), [&](const auto& pairEntry) {
      return pairEntry.second == globalChannelID;
    });
    return findResult->first;
  }
  Topo_t getTopoTCM() const
  {
    const auto& mapModuleType = getMapEntryCRU2ModuleType();
    auto findResult = std::find_if(mapModuleType.begin(), mapModuleType.end(), [&](const auto& pairEntry) {
      return pairEntry.second == EModuleType::kTCM;
    });
    return Topo_t{findResult->first, 0};
  }
  // Prepare full map for FEE metadata(for digit2raw convertion)
  template <typename RDHtype, typename RDHhelper = void>
  auto makeMapFEEmetadata() -> std::map<Topo_t, RDHtype>
  {
    std::map<Topo_t, RDHtype> mapResult;
    const uint16_t cruID = 0; // constant
    uint64_t feeID = 0;       // increments
    const auto& mapEntryPM2ChannelID = getMapEntryPM2ChannelID();
    // Temporary for sorting FEEIDs without using them from LUT(for digit2raw convertion), and by using GlobalChannelID
    std::map<int, Topo_t> mapBuf;
    for (const auto& entry : mapEntryPM2ChannelID) {
      mapBuf.insert({entry.second, entry.first});
    }
    const auto& cru_tcm = getEntryCRU_TCM();

    // FIXME: quick fix for to get the TCM into the right channel
    // mapBuf.insert({static_cast<int>(mapBuf.size()), Topo_t{cru_tcm, 0}});
    mapBuf.insert({1 + static_cast<int>((--mapBuf.end())->first), Topo_t{cru_tcm, 0}});
    //
    for (const auto& pairEntry : mapBuf) {
      auto en = pairEntry.second;
      auto pairInserted = mapResult.insert({makeGlobalTopo(en), RDHtype{}});
      if (pairInserted.second) {
        auto& rdhObj = pairInserted.first->second;
        const auto& topoObj = pairInserted.first->first;
        if constexpr (std::is_same<RDHhelper, void>::value) {
          rdhObj.linkID = topoObj.mEntryCRU.mLinkID;
          rdhObj.endPointID = topoObj.mEntryCRU.mEndPointID;
          rdhObj.feeId = feeID;
          rdhObj.cruID = cruID;
        } else // Using RDHUtils
        {
          RDHhelper::setLinkID(&rdhObj, topoObj.mEntryCRU.mLinkID);
          RDHhelper::setEndPointID(&rdhObj, topoObj.mEntryCRU.mEndPointID);
          RDHhelper::setFEEID(&rdhObj, feeID);
          RDHhelper::setCRUID(&rdhObj, cruID);
        }
        feeID++;
      }
    }
    for (const auto& entry : mapResult) {
      std::cout << "\nTEST: " << entry.first << std::endl;
    }
    return mapResult;
  }

 private:
  EntryCRU_t mEntryCRU_TCM;
  Table_t mVecEntryFEE;
  MapEntryCRU2ModuleType_t mMapEntryCRU2ModuleType;
  MapEntryPM2ChannelID_t mMapEntryPM2ChannelID;
};

// Singleton for LookUpTable, coomon for all three FIT detectors
template <o2::detectors::DetID::ID DetID, typename LUT>
class SingleLUT : public LUT
{
 private:
  SingleLUT() = default;
  SingleLUT(const std::string& ccdbPath, const std::string& ccdbPathToLUT) : LUT(ccdbPath, ccdbPathToLUT) {}
  SingleLUT(const std::string& pathToFile) : LUT(pathToFile) {}
  SingleLUT(const SingleLUT&) = delete;
  SingleLUT& operator=(SingleLUT&) = delete;
  constexpr static bool isValidDet()
  {
    return (DetID == o2::detectors::DetID::FDD) || (DetID == o2::detectors::DetID::FT0) || (DetID == o2::detectors::DetID::FV0);
  }

 public:
  typedef LUT LookupTable_t;
  typedef typename LookupTable_t::Table_t Table_t;

  constexpr static const char* getObjectPath()
  {
    static_assert(isValidDet(), "Invalid detector type(o2::detectors::DetID::ID)! Should be one of the FIT detector!");
    if constexpr (DetID == o2::detectors::DetID::FDD) {
      return "FDD/Config/LookupTable";
    } else if constexpr (DetID == o2::detectors::DetID::FT0) {
      return "FT0/Config/LookupTable";
    } else if constexpr (DetID == o2::detectors::DetID::FV0) {
      return "FV0/Config/LookupTable";
    }
    return "";
  }
  static constexpr o2::detectors::DetID sDetID = o2::detectors::DetID(DetID);
  static constexpr const char* sDetectorName = o2::detectors::DetID::getName(DetID);
  static constexpr const char* sDefaultLUTpath = getObjectPath();
  static constexpr const char sObjectName[] = "LookupTable";
  inline static std::string sCurrentCCDBpath = "";
  inline static std::string sCurrentLUTpath = sDefaultLUTpath;
  // Before instance() call, setup url and path
  static void setCCDBurl(const std::string& url) { sCurrentCCDBpath = url; }
  static void setLUTpath(const std::string& path) { sCurrentLUTpath = path; }
  bool mFirstUpdate{true}; // option in case if LUT should be updated during workflow initialization
  static SingleLUT& Instance(const Table_t* table = nullptr, long timestamp = -1)
  {
    if (sCurrentCCDBpath == "") {
      sCurrentCCDBpath = o2::base::NameConf::getCCDBServer();
    }
    static SingleLUT instanceLUT;
    if (table != nullptr) {
      instanceLUT.initFromTable(table);
      instanceLUT.mFirstUpdate = false;
    } else if (instanceLUT.mFirstUpdate) {
      instanceLUT.initCCDB(sCurrentCCDBpath, sCurrentLUTpath, timestamp);
      instanceLUT.mFirstUpdate = false;
    }
    return instanceLUT;
  }
};

} // namespace fit
} // namespace o2

#endif
