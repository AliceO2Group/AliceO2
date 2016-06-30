#ifndef AliceO2_TPC_Mapper_H
#define AliceO2_TPC_Mapper_H

#include <map>
#include <vector>
#include <string>

#include "Defs.h"
#include "PadPos.h"
#include "FECInfo.h"
#include "PadRegionInfo.h"

namespace AliceO2 {
namespace TPC {

class Mapper {
public:
  static Mapper& instance() {
    static Mapper mapper;
    return mapper;
  }

  const PadPos&     padPos    (GlobalPadNumber padNumber) const { return mMapGlobalPadToPadPos[padNumber%mPadsInSector]; }
  const PadCentre&  padCentre (GlobalPadNumber padNumber) const { return mMapGlobalPadCentre  [padNumber%mPadsInSector]; }
  const FECInfo&    fecInfo   (GlobalPadNumber padNumber) const { return mMapGlobalPadFECInfo [padNumber%mPadsInSector]; }
  
  const GlobalPadNumber globalPadNumber(const PadPos& padPosition) const { return mMapPadPosGlobalPad.find(padPosition)->second; }

  const PadRegionInfo& padRegionInfo(const unsigned char region) const { return mMapPadRegionInfo[region]; }

  static const unsigned short getPadsInIROC  () { return mPadsInIROC  ; }
  static const unsigned short getPadsInOROC1 () { return mPadsInOROC1 ; }
  static const unsigned short getPadsInOROC2 () { return mPadsInOROC2 ; }
  static const unsigned short getPadsInOROC3 () { return mPadsInOROC3 ; }
  static const unsigned short getPadsInOROC  () { return mPadsInOROC  ; }
  static const unsigned short getPadsInSector() { return mPadsInSector; }

//   bool loadFECInfo();
//   bool loadTraceLengh();
//   bool loadPositions();

  // c++11 feature don't work with root dictionary :(
//   Mapper(const Mapper&) = delete;
//   void operator=(const Mapper&) = delete;
private:
  Mapper() : mMapGlobalPadToPadPos(mPadsInSector),  mMapGlobalPadCentre(mPadsInSector), mMapPadPosGlobalPad(), mMapGlobalPadFECInfo(mPadsInSector), mMapPadRegionInfo(10) {load();}
  // use old c++03 due to root
  Mapper(const Mapper&) {}
  void operator=(const Mapper&) {}

  void load();
  void initPadRegions();
  bool readMappingFile(std::string file);

  static const unsigned short mPadsInIROC  {5280};  /// number of pads in IROC
  static const unsigned short mPadsInOROC1 {2880};  /// number of pads in OROC1
  static const unsigned short mPadsInOROC2 {3200};  /// number of pads in OROC2
  static const unsigned short mPadsInOROC3 {3200};  /// number of pads in OROC3
  static const unsigned short mPadsInOROC  {9280};  /// number of pads in OROC
  static const unsigned short mPadsInSector{14560}; /// number of pads in one sector

  // ===| Pad Mappings |========================================================
  std::vector<PadPos>               mMapGlobalPadToPadPos; /// mapping of global pad number to row and pad
  std::vector<PadCentre>            mMapGlobalPadCentre;   /// pad coordinates
  std::map<PadPos, GlobalPadNumber> mMapPadPosGlobalPad;   /// mapping pad position to global pad number, most probably needs to be changed to speed up
  std::vector<FECInfo>              mMapGlobalPadFECInfo;  /// map global pad number to FEC info

  // ===| Pad region mappings |=================================================
  std::vector<PadRegionInfo>        mMapPadRegionInfo;     /// pad region information

};

}
}

#endif
