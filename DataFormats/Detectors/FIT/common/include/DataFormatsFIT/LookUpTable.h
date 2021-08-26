// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
struct EntryCRU {//This is specific struct for CRU entry
  int mLinkID;
  int mEndPointID;
  int mCRUID;
  int mFEEID;
  friend std::ostream& operator<<(std::ostream& os, const EntryCRU &entryCRU) {
      os<<"LinkID: "<<entryCRU.mLinkID<<"|";
      os<<"EndPointID: "<<entryCRU.mEndPointID<<"|";
      os<<"CRUID: "<<entryCRU.mCRUID<<"|";
      os<<"FEEID: "<<entryCRU.mFEEID;
    return os;
  }
  void parse(const boost::property_tree::ptree& propertyTree) {
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
  std::size_t operator()(const EntryCRU& entryCRU) const {
    return (entryCRU.mLinkID << 4) | entryCRU.mEndPointID;
  }
};

struct ComparerCRU {
  bool operator()(const EntryCRU& entry1,const EntryCRU& entry2) const {
    return true;
//    return entry1.mLinkID<entry2.mLinkID || entry1.mEndPointID<entry2.mEndPointID;
  }
};

struct EntryPM {
  EntryCRU mEntryCRU;
  int mLocalChannelID = 0;
  friend std::ostream& operator<<(std::ostream& os, const EntryPM &entryPM) {
      os<<entryPM.mEntryCRU<<"|";
      os<<"LocalChID: "<<entryPM.mLocalChannelID;
    return os;
  }
  ClassDefNV(EntryPM, 1);
};

inline bool operator<(EntryPM const& entryPM1, EntryPM const& entryPM2)
{
  auto comparer = [](const EntryPM & entryPM) -> decltype(auto) { return std::tie(entryPM.mEntryCRU.mEndPointID, entryPM.mEntryCRU.mLinkID, entryPM.mLocalChannelID ); };
  return comparer(entryPM1) < comparer(entryPM2);
}

struct HasherPM {
  // Hash-function without any collisions due to technical bit size of fields:
  // RDH::EndPointID :  4 bits
  // EventData::ChannelID : 4 bits
  // RDH::LinkID : 8 bits
  std::size_t operator()(const EntryPM& entryPM) const {
    return (entryPM.mEntryCRU.mLinkID << 8) | (entryPM.mLocalChannelID << 4 )| (entryPM.mEntryCRU.mEndPointID);
  }
};

struct ComparerPM {
  //Always true due to perfect hasher
  bool operator()(const EntryPM& entry1,const EntryPM& entry2) const {
    return true;
  }
};


struct EntryFEE {
  EntryCRU mEntryCRU;
  std::string mChannelID;  //ChannelID, string type because some entries containes N/A
  std::string mLocalChannelID; //Local channelID, string type because some entries containes N/A
  std::string mModuleType; //PM, PM-LCS, TCM
  std::string mModuleName;
  std::string mBoardHV;
  std::string mChannelHV;
  std::string mSerialNumberMCP;
  std::string mCableHV;
  std::string mCableSignal;
  friend std::ostream& operator<<(std::ostream& os, const EntryFEE &entryFEE) {
      os<<entryFEE.mEntryCRU<<"|";
      os<<"ChannelID: "<<entryFEE.mChannelID<<"|";
      os<<"LocalChannelID: "<<entryFEE.mLocalChannelID<<"|";
      os<<"ModuleType: "<<entryFEE.mModuleType<<"|";
      os<<"ModuleName: "<<entryFEE.mModuleName<<"|";
      os<<"HV board: "<<entryFEE.mBoardHV<<"|";
      os<<"HV channel: "<<entryFEE.mChannelHV<<"|";
      os<<"MCP S/N: "<<entryFEE.mSerialNumberMCP<<"|";
      os<<"HV cable: "<<entryFEE.mCableHV<<"|";
      os<<"signal cable: "<<entryFEE.mCableSignal<<"|";
    return os;
  }

  void parse(const boost::property_tree::ptree& propertyTree) {
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
enum class EModuleType:int {kUnknown,kPM,kPM_LCS,kTCM};

template<typename MapEntryCRU2ModuleType = std::unordered_map<EntryCRU,EModuleType,HasherCRU,ComparerCRU>,
         typename MapEntryPM2ChannelID = std::unordered_map<EntryPM,int,HasherPM,ComparerPM>,
         typename ChannelID=int,
         typename = typename std::enable_if_t<std::is_integral<ChannelID>::value> >
class LookupTable
{
 public:
  LookupTable() {init();}
  LookupTable(const std::string& ccdbPath,const std::string& ccdbPathToLUT) {initCCDB(ccdbPath,ccdbPathToLUT);}
  typedef MapEntryPM2ChannelID MapEntryPM2ChannelID_t;
  typedef MapEntryCRU2ModuleType MapEntryCRU2ModuleType_t;
  typedef ChannelID ChannelID_t;
  //Map of str module names -> enum types
  const std::map<std::string,EModuleType> mMapModuleTypeStr2Enum = {{"PM",EModuleType::kPM},{"PM-LCS",EModuleType::kPM_LCS},{"TCM",EModuleType::kTCM}};
  //Warning! To exclude double mapping do not use isTCM and isPM in the same time
  bool isTCM(int linkID, int epID) const {
    return mEntryCRU_TCM.mLinkID==linkID && mEntryCRU_TCM.mEndPointID==epID;
  }
  bool isTCM(const EntryCRU & entryCRU) const {
    if(getModuleType(entryCRU)==EModuleType::kTCM) {
      return true;
    }
    else {
      return false;
    }
  }
  bool isPM(const EntryCRU & entryCRU) const {
    if(getModuleType(entryCRU)==EModuleType::kPM || getModuleType(entryCRU)==EModuleType::kPM_LCS) {
      return true;
    }
    else {
      return false;
    }
  }
  EModuleType getModuleType(const EntryCRU &entryCRU) const {
    const auto &mapEntries = getMapEntryCRU2ModuleType();
    const auto &it = mapEntries.find(entryCRU);
    if(it!=mapEntries.end()) {
      return it->second;
    }
    else {
      return EModuleType::kUnknown;
    }
  }
  EModuleType getModuleType(const std::string &moduleType) {
    const auto &it = mMapModuleTypeStr2Enum.find(moduleType);
    if(it!=mMapModuleTypeStr2Enum.end()) {
      return it->second;
    }
    else {
      return EModuleType::kUnknown;
    }
  }
  void init() {
    std::string inputDir;
    const char* aliceO2env = std::getenv("O2_ROOT");
    if (aliceO2env) {
      inputDir = aliceO2env;
    }
    inputDir += "/share/Detectors/FT0/files/";
    std::string filepath = inputDir + "LookupTable_FT0.json";
    filepath = gSystem->ExpandPathName(filepath.data()); // Expand $(ALICE_ROOT) into real system path
    prepareEntriesFEE(filepath);
    prepareLUT();
  }
  void initCCDB(const std::string& ccdbPath,const std::string& ccdbPathToLUT) {
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL(ccdbPath);
    mVecEntryFEE = *(mgr.get<std::vector<EntryFEE>>(ccdbPathToLUT));
    prepareLUT();
  }
  ChannelID_t getGlobalChannelID(const EntryPM &entryPM) const {
    const auto &it = mMapEntryPM2ChannelID.find(entryPM);
    if(it!=mMapEntryPM2ChannelID.end()) {
      return it->second;
    }
    else {
      return -1;
    }
  }
  int getChannel (int linkID, int chID, int ep=0) {
    return mMapEntryPM2ChannelID.find(std::move(EntryPM{EntryCRU{linkID,ep,0,0},chID}))->second;
  }
  void prepareEntriesFEE(const std::string& pathToConfigFile) {
    boost::property_tree::ptree propertyTree;
    boost::property_tree::read_json(pathToConfigFile.c_str(), propertyTree);
    mVecEntryFEE = prepareEntriesFEE(propertyTree);
  }
  std::vector<EntryFEE> prepareEntriesFEE(const boost::property_tree::ptree& propertyTree) {
    std::vector<EntryFEE> vecEntryFEE;
    for (const auto &pairEntry : propertyTree) {
      const auto &propertyTreeSingle = pairEntry.second;
      EntryFEE entryFEE{};
      entryFEE.parse(propertyTreeSingle);
      vecEntryFEE.push_back(entryFEE);
    }
    return vecEntryFEE;
  }
  
  void prepareLUT() {
    mMapEntryCRU2ModuleType.clear();
    mMapEntryPM2ChannelID.clear();
    const auto &vecEntryFEE = getVecMetadataFEE();
    for(const auto entryFEE: vecEntryFEE) {
      EntryCRU entryCRU = entryFEE.mEntryCRU;
      std::string strModuleType = entryFEE.mModuleType;
      EModuleType moduleType = getModuleType(strModuleType);
      if(moduleType!=EModuleType::kUnknown) {
        mMapEntryCRU2ModuleType.insert({entryCRU,moduleType});
      }
      if(moduleType==EModuleType::kPM||moduleType==EModuleType::kPM_LCS) {
        const std::string &strChannelID = entryFEE.mChannelID;
        const std::string &strLocalChannelID = entryFEE.mLocalChannelID;
        EntryPM entryPM{entryCRU,std::stoi(strLocalChannelID)};
        mMapEntryPM2ChannelID.insert({entryPM,std::stoi(strChannelID)});
      }
      if(moduleType==EModuleType::kTCM) {
        mEntryCRU_TCM = entryCRU;
      }
    }
  }
  void printFullMap() const {
    for(const auto& entry: mVecEntryFEE) {
      LOG(INFO)<<entry;
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
  const std::vector<EntryFEE>& getVecMetadataFEE() const {return mVecEntryFEE;}
  const MapEntryCRU2ModuleType_t& getMapEntryCRU2ModuleType() const {return mMapEntryCRU2ModuleType;}
  const MapEntryPM2ChannelID_t& getMapEntryPM2ChannelID() const {return mMapEntryPM2ChannelID;}
  const EntryCRU& getEntryCRU_TCM() const {return mEntryCRU_TCM;}
 private:
  EntryCRU mEntryCRU_TCM;
  std::vector<EntryFEE> mVecEntryFEE;
  MapEntryCRU2ModuleType_t mMapEntryCRU2ModuleType;
  MapEntryPM2ChannelID_t mMapEntryPM2ChannelID;
};
//Singleton for LookUpTable
template<typename LUT>
class SingleLUT : public LUT
{
 private:
  SingleLUT(const std::string& ccdbPath,const std::string& ccdbPathToLUT) : LUT(ccdbPath,ccdbPathToLUT) {}
  SingleLUT(const SingleLUT&) = delete;
  SingleLUT& operator=(SingleLUT&) = delete;

 public:
  typedef EntryPM Topo_t;
  static constexpr char sDetectorName[] = "FT0";
  static SingleLUT& Instance(std::string ccdbPath="http://ccdb-test.cern.ch:8080/",std::string ccdbPathToLUT="FT0/LookUpTableNew")
  {
    static SingleLUT instanceLUT(ccdbPath,ccdbPathToLUT);
    return instanceLUT;
  }
  //Temporary
  //Making topo for FEE recognizing(Local channelID is supressed)
  static Topo_t makeGlobalTopo(const Topo_t& topo)
  {
    return Topo_t{topo.mEntryCRU,0};
  }
  static int getLocalChannelID(const Topo_t& topo)
  {
    return topo.mLocalChannelID;
  }
  Topo_t getTopoPM(int globalChannelID) const {
    const auto &mapChannels = Instance().getMapEntryPM2ChannelID();
    auto findResult = std::find_if(mapChannels.begin(), mapChannels.end(), [&](const auto &pairEntry)
    {
      return pairEntry.second == globalChannelID;
    });
    return findResult->first;
  }
  Topo_t getTopoTCM() const {
    const auto &mapModuleType = Instance().getMapEntryCRU2ModuleType();
    auto findResult = std::find_if(mapModuleType.begin(), mapModuleType.end(), [&](const auto &pairEntry)
    {
      return pairEntry.second == EModuleType::kTCM;
    });
    return Topo_t{findResult->first, 0};
  }
  //Prepare full map for FEE metadata(for digit2raw convertion)
  template <typename RDHtype, typename RDHhelper = void>
  auto makeMapFEEmetadata() -> std::map<Topo_t, RDHtype>
  {
    std::map<Topo_t, RDHtype> mapResult;
    const uint16_t cruID = 0;      //constant
    uint64_t feeID = 0;            //increments
    const auto &mapEntryPM2ChannelID = Instance().getMapEntryPM2ChannelID();
    //Temporary for sorting FEEIDs without using them from LUT(for digit2raw convertion), and by using GlobalChannelID
    std::map<int,Topo_t> mapBuf;
    for(const auto &entry: mapEntryPM2ChannelID) {
      mapBuf.insert({entry.second,entry.first});
    }
    const auto &cru_tcm = Instance().getEntryCRU_TCM();
    mapBuf.insert({static_cast<int>(mapBuf.size()),Topo_t{cru_tcm,0}});
    //
    for(const auto &pairEntry: mapBuf) {
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
        } else //Using RDHUtils
        {
          RDHhelper::setLinkID(&rdhObj, topoObj.mEntryCRU.mLinkID);
          RDHhelper::setEndPointID(&rdhObj, topoObj.mEntryCRU.mEndPointID);
          RDHhelper::setFEEID(&rdhObj, feeID);
          RDHhelper::setCRUID(&rdhObj, cruID);
        }
        feeID++;
      }
    }
    for(const auto&entry: mapResult) {
      std::cout<<"\nTEST: "<<entry.first<<std::endl;
    }
    return mapResult;
  }
};
using LUT = SingleLUT<LookupTable<>>;
} // namespace fit
} // namespace o2
#endif
