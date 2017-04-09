#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>

// #include <boost/format.hpp>
// using std::cout;
// using std::endl;
// using boost::format;

#include "TPCBase/Mapper.h"
namespace o2 {
namespace TPC {
  constexpr std::array<double, SECTORSPERSIDE> Mapper::SinsPerSector/*{{
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
  }}*/;

  //     static constexpr std::array<int, 2> test{1,2};

  constexpr std::array<double, SECTORSPERSIDE> Mapper::CosinsPerSector/*{{
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
  }}*/;

Mapper::Mapper(const std::string& mappingDir)
  : mMapGlobalPadToPadPos(mPadsInSector),
    mMapGlobalPadCentre(mPadsInSector),
    mMapPadPosGlobalPad(),
    mMapFECIDGlobalPad(FECInfo::globalSAMPAId(91,0,0)),
    mMapGlobalPadFECInfo(mPadsInSector),
    mMapPadRegionInfo(),
    mMapPartitionInfo()
{
  load(mappingDir);
}
bool Mapper::readMappingFile(std::string file)
{
  // ===| Mapping file layout |=================================================
  //   Col 0 -> INDEX
  //   Col 1 -> PADROW
  //   Col 2 -> PAD
  //   Col 3 -> X coordinate
  //   Col 4 -> y coordinate
  //   Col 5 -> Connector
  //   Col 6 -> Pin
  //   Col 7 -> Partion
  //   Col 8 -> Region
  //   Col 9 -> FEC
  //   Col 10 -> FEC Connector
  //   Col 11 -> FEC Channel
  //   Col 12 -> SAMPA Chip
  //   Col 13 -> SAMPA Channel

  // ===| Input variables |=====================================================
  // pad info
  GlobalPadNumber padIndex;
  unsigned int padRow;
  unsigned int pad;
  float         xPos;
  float         yPos;

  // pad plane info
  unsigned int connector;
  unsigned int pin;
  unsigned int partion;
  unsigned int region;

  // FEC info
  unsigned int fecIndex;
  unsigned int fecConnector;
  unsigned int fecChannel;
  unsigned int sampaChip;
  unsigned int sampaChannel;

  std::string       line;
  std::ifstream infile(file, std::ifstream::in);
  while (std::getline(infile, line)) {
    std::stringstream streamLine(line);
    streamLine
      // pad info
      >> padIndex
      >> padRow
      >> pad
      >> xPos
      >> yPos

      // pad plane info
      >> connector
      >> pin
      >> partion
      >> region

      // FEC info
      >> fecIndex
      >> fecConnector
      >> fecChannel
      >> sampaChip
      >> sampaChannel;

      // the x and y positions are in mm
      // in the mapping files, the values are given for sector C04 in the global ALICE coordinate system
      // however, we need it in the local tracking system. Therefore:
      const float localX=yPos/10.f;
      const float localY=-xPos/10.f;
      // with the pad counting (looking to C-Side pad 0,0 is bottom left -- pad-side front view)
      // these values are for the C-Side
      // For the A-Side the localY position must be mirrored

      mMapGlobalPadToPadPos[padIndex]         = PadPos(padRow,pad);
      mMapPadPosGlobalPad[PadPos(padRow,pad)] = padIndex;
      mMapGlobalPadFECInfo[padIndex]          = FECInfo(fecIndex, /*fecConnector, fecChannel,*/ sampaChip, sampaChannel);
      mMapFECIDGlobalPad[FECInfo::globalSAMPAId(fecIndex, sampaChip, sampaChannel)] = padIndex;
      mMapGlobalPadCentre[padIndex]           = PadCentre(localX, localY);

      //std::cout
      //<< padIndex<< " "
      //<< padRow<< " "
      //<< pad<< " "
      //<< xPos<< " "
      //<< yPos<< " "
      //<< " "
      //// pad plane info<< " "
      //<< connector<< " "
      //<< pin<< " "
      //<< partion<< " "
      //<< region<< " "
      //<< " "
      //// FEC info<< " "
      //<< fecIndex<< " "
      //<< fecConnector<< " "
      //<< fecChannel<< " "
      //<< sampaChip<< " "
      //<< sampaChannel << std::endl;
  }
}

void Mapper::load(const std::string& mappingDir)
{

//   std::string inputDir(std::getenv("ALICEO2"));
  std::string inputDir=mappingDir;
  if (!inputDir.size()) {
    //const char* aliceO2env=std::getenv("ALICEO2");
    //if (aliceO2env) inputDir=aliceO2env;
    //readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-IROC.txt");
    //readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-OROC1.txt");
    //readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-OROC2.txt");
    //readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-OROC3.txt");

    const char* aliceO2env=std::getenv("O2_ROOT");
    if (aliceO2env) inputDir=aliceO2env;
    inputDir+="/share/Detectors/TPC/files";
  }
  readMappingFile(inputDir+"/TABLE-IROC.txt");
  readMappingFile(inputDir+"/TABLE-OROC1.txt");
  readMappingFile(inputDir+"/TABLE-OROC2.txt");
  readMappingFile(inputDir+"/TABLE-OROC3.txt");

  initPadRegionsAndPartitions();
}

void Mapper::initPadRegionsAndPartitions()
{
  // original values for pad widht and height and pad row position are in mm
  // the ALICE coordinate system is in cm
  mMapPadRegionInfo[0]=PadRegionInfo(0, 0, 17, 7.5/10., 4.16/10.,  848.5/10.,  0, 33.20,   0);
  mMapPadRegionInfo[1]=PadRegionInfo(0, 1, 15, 7.5/10., 4.20/10.,  976.0/10., 17, 33.00,  17);
  mMapPadRegionInfo[2]=PadRegionInfo(1, 2, 16, 7.5/10., 4.20/10., 1088.5/10., 32, 33.08,  32);
  mMapPadRegionInfo[3]=PadRegionInfo(1, 3, 15, 7.5/10., 4.36/10., 1208.5/10., 48, 31.83,  48);
  mMapPadRegionInfo[4]=PadRegionInfo(2, 4, 18, 10/10. , 6.00/10., 1347.0/10.,  0, 38.00,  63);
  mMapPadRegionInfo[5]=PadRegionInfo(2, 5, 16, 10/10. , 6.00/10., 1527.0/10., 18, 38.00,  81);
  mMapPadRegionInfo[6]=PadRegionInfo(3, 6, 16, 12/10. , 6.08/10., 1708.0/10.,  0, 47.90,  97);
  mMapPadRegionInfo[7]=PadRegionInfo(3, 7, 14, 12/10. , 5.88/10., 1900.0/10., 16, 49.55, 113);
  mMapPadRegionInfo[8]=PadRegionInfo(4, 8, 13, 15/10. , 6.04/10., 2089.0/10.,  0, 59.39, 127);
  mMapPadRegionInfo[9]=PadRegionInfo(4, 9, 12, 15/10. , 6.07/10., 2284.0/10.,  0, 64.70, 140);

  mMapPartitionInfo[0]=PartitionInfo(15, 0          , 32, 0          , 2400 );
  mMapPartitionInfo[1]=PartitionInfo(18, 15         , 31, 32         , 2880 );
  mMapPartitionInfo[2]=PartitionInfo(18, 15+18      , 34, 32+31      , 2880 );
  mMapPartitionInfo[3]=PartitionInfo(20, 15+18+18   , 30, 32+31+34   , 3200 );
  mMapPartitionInfo[4]=PartitionInfo(20, 15+18+18+20, 25, 32+31+34+30, 3200 );
}

}
}
