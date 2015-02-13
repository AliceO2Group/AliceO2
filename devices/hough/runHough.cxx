/// \file runHough.cxx
/// \brief Implementation of a cluster loader
/// \author Charis Kouzinopoulos

#include "AliHLTTPCTrackGeometry.h"
#include "AliHLTTPCClusterDataFormat.h"
#include "AliHLTTPCSpacePointContainer.h"
#include "AliHLTComponent.h"
#include "AliHLTTPCDefinitions.h"

#include "boost/filesystem.hpp"

#include <sstream>

int processData(std::string dataPath, std::string dataType, std::string dataOrigin)
{
  std::ifstream inputData(dataPath.c_str(), std::ifstream::binary);
  if (!inputData) {
    std::cerr << "Error, cluster data file " << dataPath << " could not be accessed" << endl;
    std::exit(1);
  }

  //Retrieve the TPC slice and partition from the filename
  std::string currentSliceString (dataPath, dataPath.length() - 6, 2);
  std::string currentPartitionString (dataPath, dataPath.length() - 2, 2);

  AliHLTUInt8_t currentSlice = std::stoul(currentSliceString, nullptr, 16);
  AliHLTUInt8_t currentPartition = std::stoul(currentPartitionString, nullptr, 16);

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

  AliHLTTPCDefinitions::EncodeDataSpecification(currentSlice, currentSlice, currentPartition, currentPartition);

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
  std::string dataFilename = "emulated-tpc-clusters/event";
  dataFilename += argv[1];

  boost::filesystem::path dataPath(dataFilename);
  boost::filesystem::directory_iterator endIterator;

  typedef std::multimap<std::time_t, boost::filesystem::path> result_set_t;
  result_set_t result_set;

  std::string dataType = "CLUSTERS", dataOrigin = "TPC ";

  int totalNumberOfClusters = 0, totalNumberOfDataFiles = 0;

  if (boost::filesystem::exists(dataPath) && boost::filesystem::is_directory(dataPath)) {
    for (boost::filesystem::directory_iterator directoryIterator(dataPath); directoryIterator != endIterator; ++directoryIterator) {
      if (boost::filesystem::is_regular_file(directoryIterator->status())) {
        totalNumberOfClusters += processData(directoryIterator->path().string(), dataType, dataOrigin);
        totalNumberOfDataFiles++;
      }
    }
  } else {
    std::cerr << "Path " << dataPath.string() << "/ could not be found or does not contain any valid data files" << endl;
    exit(1);
  }

  cout << "Added " << totalNumberOfClusters << " clusters from " << totalNumberOfDataFiles << " data files" << endl;
  return 0;
}
