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
namespace AliceO2 {
namespace TPC {

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
      mMapGlobalPadFECInfo[padIndex]          = FECInfo(fecIndex, fecConnector, fecChannel, sampaChip, sampaChannel);
      mMapGlobalPadCentre[padIndex]           = PadCentre(localX, localY);

//       std::cout
//       << padIndex<< " "
//       << padRow<< " "
//       << pad<< " "
//       << xPos<< " "
//       << yPos<< " "
//       << " "
//       // pad plane info<< " "
//       << connector<< " "
//       << pin<< " "
//       << partion<< " "
//       << region<< " "
//       << " "
//       // FEC info<< " "
//       << fecIndex<< " "
//       << fecConnector<< " "
//       << fecChannel<< " "
//       << sampaChip<< " "
//       << sampaChannel << std::endl;
  }
}

void Mapper::load()
{

//   std::string inputDir(std::getenv("ALICEO2"));
  std::string inputDir;
  const char* aliceO2env=std::getenv("ALICEO2");
  if (aliceO2env) inputDir=aliceO2env;
  readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-IROC.txt");
  readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-OROC1.txt");
  readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-OROC2.txt");
  readMappingFile(inputDir+"/Detectors/TPC/base/files/TABLE-OROC3.txt");

  initPadRegions();
}

void Mapper::initPadRegions()
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

}

const DigitPos Mapper::findDigitPosFromLocalPosition(const LocalPosition3D& pos, const Sector& sec) const
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

const DigitPos Mapper::findDigitPosFromGlobalPosition(const GlobalPosition3D& pos) const
{
  // ===| find sector |=========================================================
  double phi=atan2(pos.getY(), pos.getX());
  if (phi<0.) phi+=TWOPI;
  const unsigned char secNum = floor(phi/SECPHIWIDTH);
  const double        secPhi = secNum*SECPHIWIDTH+SECPHIWIDTH/2.;
  Sector sec(secNum+(pos.getZ()<0)*SECTORSPERSIDE);

  // ===| rotated position |====================================================
  LocalPosition3D posLoc=GlobalToLocal(pos, secPhi);

  return findDigitPosFromLocalPosition(posLoc, sec);
}

}
}
