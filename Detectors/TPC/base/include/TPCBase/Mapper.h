#ifndef AliceO2_TPC_Mapper_H
#define AliceO2_TPC_Mapper_H

#include <map>
#include <vector>
#include <array>
#include <string>
#include <cmath>

#include "TPCBase/Defs.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/DigitPos.h"
#include "TPCBase/FECInfo.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/Sector.h"

using AliceO2::TPC::PadRegionInfo;

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

  const PadRegionInfo& getPadRegionInfo(const unsigned char region) const { return mMapPadRegionInfo[region]; }

  const DigitPos findDigitPosFromLocalPosition(const LocalPosition3D& pos, const Sector& sec) const;
  const DigitPos findDigitPosFromGlobalPosition(const GlobalPosition3D& pos) const;

  const std::vector<PadRegionInfo>& getMapPadRegionInfo() const { return mMapPadRegionInfo; }

  const int getNumberOfPadRegions() { return int(mMapPadRegionInfo.size()); }

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

  static GlobalPosition3D LocalToGlobal(const LocalPosition3D pos, const double alpha)
  {
    const double cs=std::cos(alpha), sn=std::sin(alpha);
    return GlobalPosition3D(float(double(pos.getX())*cs-double(pos.getY())*sn),
                            float(double(pos.getX())*sn+double(pos.getY()*cs)),
                            pos.getZ());
  }

  static LocalPosition3D GlobalToLocal(const GlobalPosition3D& pos, const double alpha)
  {
    ///@todo: Lookup over sector number
    const double cs=std::cos(-alpha), sn=std::sin(-alpha);
    return LocalPosition3D(float(double(pos.getX())*cs-double(pos.getY())*sn),
                          float(double(pos.getX())*sn+double(pos.getY()*cs)),
                          pos.getZ());
  }

  static GlobalPosition3D LocalToGlobal(const LocalPosition3D pos, const Sector sec)
  {
    const double cs=CosinsPerSector[sec.getSector()%SECTORSPERSIDE], sn=SinsPerSector[sec.getSector()%SECTORSPERSIDE];
    return GlobalPosition3D(float(double(pos.getX())*cs-double(pos.getY())*sn),
                            float(double(pos.getX())*sn+double(pos.getY()*cs)),
                            pos.getZ());
  }

  static LocalPosition3D GlobalToLocal(const GlobalPosition3D& pos, const Sector sec)
  {
    ///@todo: Lookup over sector number
    const double cs=CosinsPerSector[sec.getSector()%SECTORSPERSIDE], sn=-SinsPerSector[sec.getSector()%SECTORSPERSIDE];
    return LocalPosition3D(float(double(pos.getX())*cs-double(pos.getY())*sn),
                          float(double(pos.getX())*sn+double(pos.getY()*cs)),
                          pos.getZ());
  }

private:
  Mapper() : mMapGlobalPadToPadPos(mPadsInSector),  mMapGlobalPadCentre(mPadsInSector), mMapPadPosGlobalPad(), mMapGlobalPadFECInfo(mPadsInSector), mMapPadRegionInfo(10) {load();}
  // use old c++03 due to root
  Mapper(const Mapper&) {}
  void operator=(const Mapper&) {}

  void load();
  void initPadRegions();
  bool readMappingFile(std::string file);

  static constexpr unsigned short mPadsInIROC  {5280};  ///< number of pads in IROC
  static constexpr unsigned short mPadsInOROC1 {2880};  ///< number of pads in OROC1
  static constexpr unsigned short mPadsInOROC2 {3200};  ///< number of pads in OROC2
  static constexpr unsigned short mPadsInOROC3 {3200};  ///< number of pads in OROC3
  static constexpr unsigned short mPadsInOROC  {9280};  ///< number of pads in OROC
  static constexpr unsigned short mPadsInSector{14560}; ///< number of pads in one sector

  // ===| lookup tables |=======================================================
  //   static constexpr std::array<double, SECTORSPERSIDE> SinsPerSector;   ///< Sinus values of sectors
  //   static constexpr std::array<double, SECTORSPERSIDE> CosinsPerSector; ///< Cosinus values of sectors
  static constexpr std::array<double, SECTORSPERSIDE> SinsPerSector{{
    0,
    0.3420201433256687129080830800376133993268,
    0.6427876096865392518964199553010985255241,
    0.8660254037844385965883020617184229195118,
    0.9848077530122080203156542665965389460325,
    0.9848077530122080203156542665965389460325,
    0.866025403784438707610604524234076961875,
    0.6427876096865394739410248803324066102505,
    0.3420201433256688794415367738110944628716,
    0.,
    -0.3420201433256686573969318487797863781452,
    -0.6427876096865392518964199553010985255241,
    -0.8660254037844383745436971366871148347855,
    -0.9848077530122080203156542665965389460325,
    -0.9848077530122081313379567291121929883957,
    -0.8660254037844385965883020617184229195118,
    -0.6427876096865395849633273428480606526136,
    -0.3420201433256686018857806175219593569636
  }};

//     static constexpr std::array<int, 2> test{1,2};

  static constexpr std::array<double, SECTORSPERSIDE> CosinsPerSector{{
    1,
    0.9396926207859084279050421173451468348503,
    0.7660444431189780134516809084743726998568,
    0.5000000000000001110223024625156540423632,
    0.1736481776669304144533612088707741349936,
    -0.1736481776669303034310587463551200926304,
    -0.4999999999999997779553950749686919152737,
    -0.7660444431189779024293784459587186574936,
    -0.9396926207859083168827396548294927924871,
    -1,
    -0.9396926207859084279050421173451468348503,
    -0.7660444431189780134516809084743726998568,
    -0.5000000000000004440892098500626161694527,
    -0.1736481776669303311866343619840336032212,
    0.1736481776669299703641513588081579655409,
    0.5000000000000001110223024625156540423632,
    0.7660444431189777914070759834430646151304,
    0.9396926207859084279050421173451468348503
  }};

  // ===| Pad Mappings |========================================================
  std::vector<PadPos>               mMapGlobalPadToPadPos; ///< mapping of global pad number to row and pad
  std::vector<PadCentre>            mMapGlobalPadCentre;   ///< pad coordinates
  std::map<PadPos, GlobalPadNumber> mMapPadPosGlobalPad;   ///< mapping pad position to global pad number, most probably needs to be changed to speed up
  std::vector<FECInfo>              mMapGlobalPadFECInfo;  ///< map global pad number to FEC info

  // ===| Pad region mappings |=================================================
  std::vector<PadRegionInfo>        mMapPadRegionInfo;     ///< pad region information

};

// ===| inline functions |======================================================
inline const DigitPos Mapper::findDigitPosFromLocalPosition(const LocalPosition3D& pos, const Sector& sec) const
{
  PadPos pad;
  CRU    cru;
  for (const PadRegionInfo& padRegion : mMapPadRegionInfo) {
    cru=CRU(sec,padRegion.getPartition());
    pad=padRegion.findPad(pos);
    if (pad.isValid()) break;
  }

  return DigitPos(cru, pad);
}

inline const DigitPos Mapper::findDigitPosFromGlobalPosition(const GlobalPosition3D& pos) const
{
  // ===| find sector |=========================================================
  float phi=std::atan2(pos.getY(), pos.getX());
  if (phi<0.) phi+=TWOPI;
  const unsigned char secNum = std::floor(phi/SECPHIWIDTH);
  const float        secPhi = secNum*SECPHIWIDTH+SECPHIWIDTH/2.;
  Sector sec(secNum+(pos.getZ()<0)*SECTORSPERSIDE);

  // ===| rotated position |====================================================
//   LocalPosition3D posLoc=GlobalToLocal(pos, secPhi);
  LocalPosition3D posLoc=GlobalToLocal(pos, Sector(secNum));

  return findDigitPosFromLocalPosition(posLoc, sec);
}


}
}

#endif
