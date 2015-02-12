/// \file runHough.cxx
/// \brief Implementation of a cluster loader
/// \author Charis Kouzinopoulos

#include "AliHLTTPCTrackGeometry.h"
#include "AliHLTTPCClusterDataFormat.h"
#include "AliHLTTPCSpacePointContainer.h"
#include "AliHLTComponent.h"

#include "boost/filesystem.hpp"

#include <sstream>

int processData(std::string dataFileName, std::string dataType, std::string dataOrigin)
{
  std::ifstream inputData(dataFileName.c_str(), std::ifstream::binary);
  if (!inputData) {
    std::cerr << "Error, cluster data file " << dataFileName << " could not be accessed" << endl;
    std::exit(1);
  }

  // Get length of file
  inputData.seekg(0, inputData.end);
  int dataLength = inputData.tellg();
  inputData.seekg(0, inputData.beg);

  // Allocate memory and read file to memory
  char* inputBuffer = new char[dataLength];
  inputData.read(inputBuffer, dataLength);
  inputData.close();

  // Cast data to an AliHLTUInt8_t*, AliHLTComponentBlockData* and AliHLTTPCClusterData* data type
  AliHLTUInt8_t* pData = reinterpret_cast<AliHLTUInt8_t*>(&inputBuffer[0]);
  AliHLTComponentBlockData* bdTarget = reinterpret_cast<AliHLTComponentBlockData*>(&inputBuffer[0]);
  AliHLTTPCClusterData* pClusterData = reinterpret_cast<AliHLTTPCClusterData*>(inputBuffer);

  // Create a collection of all points
  std::auto_ptr<AliHLTTPCSpacePointContainer> spacepoints(new AliHLTTPCSpacePointContainer);
  if (!spacepoints.get()) {
    std::cerr << "Error, could not create a space point collection" << endl;
    std::exit(1);
  }

  // Determine the number of clusters from the header of the data file
  pClusterData->fSpacePointCnt = ((int)inputBuffer[1] << 8) + (int)inputBuffer[0];

  AliHLTComponent::SetDataType(bdTarget->fDataType, dataType.c_str(), dataOrigin.c_str());

  // AliHLTTPCDefinitions::EncodeDataSpecification(currentSlice, currentSlice, currentPartition, currentPartition);
  bdTarget->fSpecification = kAliHLTVoidDataSpec;
  bdTarget->fPtr = inputBuffer;
  bdTarget->fSize = sizeof(AliHLTTPCClusterData) + pClusterData->fSpacePointCnt * sizeof(AliHLTTPCSpacePointData);

  int numberOfClusters = spacepoints->AddInputBlock(bdTarget);

  // cout << *spacepoints << endl;

  if (inputBuffer) {
    delete[] inputBuffer;
  }
  inputBuffer = NULL;

  return numberOfClusters;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <event number>" << endl;
    std::exit(1);
  }

  // Create data path
  std::string dataPath = "emulated-tpc-clusters/event";
  dataPath += argv[1];

  boost::filesystem::path someDir(dataPath);
  boost::filesystem::directory_iterator end_iter;

  typedef std::multimap<std::time_t, boost::filesystem::path> result_set_t;
  result_set_t result_set;

  std::string dataType = "CLUSTERS", dataOrigin = "TPC ";

  int totalNumberOfClusters = 0, totalNumberOfDataFiles = 0;

  if (boost::filesystem::exists(someDir) && boost::filesystem::is_directory(someDir)) {
    for (boost::filesystem::directory_iterator dir_iter(someDir); dir_iter != end_iter; ++dir_iter) {
      if (boost::filesystem::is_regular_file(dir_iter->status())) {
        totalNumberOfClusters += processData(dir_iter->path().string(), dataType, dataOrigin);
        totalNumberOfDataFiles++;
      }
    }
  } else {
    std::cerr << "Path " << someDir.string() << "/ could not be found or does not contain any valid data files" << endl;
    exit(1);
  }

  cout << "Added " << totalNumberOfClusters << " clusters from " << totalNumberOfDataFiles << " data files" << endl;
  return 0;
}
