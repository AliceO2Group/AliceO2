#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>

#include "Mapper.h"
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

      mMapGlobalPadToPadPos[padIndex]         = PadPos(padRow,pad);
      mMapPadPosGlobalPad[PadPos(padRow,pad)] = padIndex;
      mMapGlobalPadFECInfo[padIndex]          = FECInfo(fecIndex, fecConnector, fecChannel, sampaChip, sampaChannel);
      mMapGlobalPadCentre[padIndex]           = PadCentre(xPos, yPos);

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
  mMapPadRegionInfo[0]=PadRegionInfo(0, 17, 7.5, 4.16,  848.5,  0, 33.20,   0);
  mMapPadRegionInfo[1]=PadRegionInfo(1, 15, 7.5, 4.20,  976.0, 17, 33.00,  17);
  mMapPadRegionInfo[2]=PadRegionInfo(2, 16, 7.5, 4.20, 1088.5, 32, 33.08,  32);
  mMapPadRegionInfo[3]=PadRegionInfo(3, 15, 7.5, 4.36, 1208.5, 48, 31.83,  48);
  mMapPadRegionInfo[4]=PadRegionInfo(4, 18, 10 , 6.00, 1347.0,  0, 38.00,  63);
  mMapPadRegionInfo[5]=PadRegionInfo(5, 16, 10 , 6.00, 1527.0, 18, 38.00,  81);
  mMapPadRegionInfo[6]=PadRegionInfo(6, 16, 12 , 6.08, 1708.0,  0, 47.90,  97);
  mMapPadRegionInfo[7]=PadRegionInfo(7, 14, 12 , 5.88, 1900.0, 16, 49.55, 113);
  mMapPadRegionInfo[8]=PadRegionInfo(8, 13, 15 , 6.04, 2089.0,  0, 59.39, 127);
  mMapPadRegionInfo[9]=PadRegionInfo(9, 12, 15 , 6.07, 2284.0,  0, 64.70, 140);

}
}
}
