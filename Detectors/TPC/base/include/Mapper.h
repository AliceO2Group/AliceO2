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

  const PadPos&  padPos (GlobalPadNumber padNumber) const { return mMapGlobalPadToPadPos[padNumber%mPadsInSector]; }
  const FECInfo& fecInfo(GlobalPadNumber padNumber) const { return mMapGlobalPadFECInfo [padNumber%mPadsInSector]; }
  const GlobalPadNumber globalPadNumber(const PadPos& padPosition) const { return mMapPadPosGlobalPad.find(padPosition)->second; }

  const PadRegionInfo& padRegionInfo(const unsigned char region) const { return mMapPadRegionInfo[region]; }

  static const unsigned short GetPadsInIROC  () { return mPadsInIROC  ; }
  static const unsigned short GetPadsInOROC1 () { return mPadsInOROC1 ; }
  static const unsigned short GetPadsInOROC2 () { return mPadsInOROC2 ; }
  static const unsigned short GetPadsInOROC3 () { return mPadsInOROC3 ; }
  static const unsigned short GetPadsInOROC  () { return mPadsInOROC  ; }
  static const unsigned short GetPadsInSector() { return mPadsInSector; }

//   bool loadFECInfo();
//   bool loadTraceLengh();
//   bool loadPositions();

  // c++11 feature don't work with root dictionary :(
//   Mapper(const Mapper&) = delete;
//   void operator=(const Mapper&) = delete;
private:
  Mapper() : mMapGlobalPadToPadPos(mPadsInSector),  mMapPadPosGlobalPad(), mMapGlobalPadFECInfo(mPadsInSector), mMapPadRegionInfo(10) {load();}
  // use old c++03 due to root
  Mapper(const Mapper&) {}
  void operator=(const Mapper&) {}

  void load();
  void initPadRegions();
  bool readMappingFile(std::string file);

  static const unsigned short mPadsInIROC  {5280};                    /// number of pad in IROC
  static const unsigned short mPadsInOROC1 {2880};
  static const unsigned short mPadsInOROC2 {3200};
  static const unsigned short mPadsInOROC3 {3200};
  static const unsigned short mPadsInOROC  {9280};
  static const unsigned short mPadsInSector{14560};

  std::vector<PadPos>               mMapGlobalPadToPadPos; /// mapping of global pad number to row and pad
  std::map<PadPos, GlobalPadNumber> mMapPadPosGlobalPad;   /// mapping pad position to global pad number, most probably needs to be changed to speed up
  std::vector<FECInfo>              mMapGlobalPadFECInfo;  /// map global pad number to FEC info
  std::vector<PadRegionInfo>        mMapPadRegionInfo;     /// pad region information

};

}
}

#endif
