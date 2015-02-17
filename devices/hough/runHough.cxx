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

std::unique_ptr<AliHLTTPCSpacePointContainer> spacepoints;
vector<float> clusterCoordinates;

void printData(int totalNumberOfClusters)
{
  cout << "Cluster ID" << setw(13) << "X coordinate" << setw(13) << "Y coordinate" << setw(13) << "Z coordinate"
       << endl;

  for (int i = 0; i < totalNumberOfClusters; i++) {
    cout << (AliHLTUInt32_t)clusterCoordinates[i * 4] << setw(13) << clusterCoordinates[i * 4 + 1] << setw(13)
         << clusterCoordinates[i * 4 + 2] << setw(13) << clusterCoordinates[i * 4 + 3] << endl;
  }
}

void addDataToCoordinatesVector(AliHLTUInt32_t clusterID, float XCoordinate, float YCoordinate, float ZCoordinate)
{
  clusterCoordinates.push_back((float)clusterID);
  clusterCoordinates.push_back(XCoordinate);
  clusterCoordinates.push_back(YCoordinate);
  clusterCoordinates.push_back(ZCoordinate);
}

int processData(std::string dataPath, std::string dataType, std::string dataOrigin)
{
  // Open data file for reading
  std::ifstream inputData(dataPath.c_str(), std::ifstream::binary);
  if (!inputData) {
    std::cerr << "Error, cluster data file " << dataPath << " could not be accessed" << endl;
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

  // Retrieve the TPC slice and partition from the filename
  std::string currentSliceString(dataPath, dataPath.length() - 6, 2);
  std::string currentPartitionString(dataPath, dataPath.length() - 2, 2);

  AliHLTUInt8_t currentSlice = std::stoul(currentSliceString, nullptr, 16);
  AliHLTUInt8_t currentPartition = std::stoul(currentPartitionString, nullptr, 16);

  // Initialize a cluster point collection
  spacepoints = std::unique_ptr<AliHLTTPCSpacePointContainer>(new AliHLTTPCSpacePointContainer);
  if (!spacepoints.get()) {
    std::cerr << "Error, could not create a space point collection" << endl;
    std::exit(1);
  }

  // Create an AliHLTComponentBlockData object, fill it with default values and then set its pointer to the data buffer
  AliHLTComponentBlockData bd;
  AliHLTComponent::FillBlockData(bd);
  bd.fPtr = inputBuffer;
  bd.fSize = dataLength;
  // bd.fDataType=kAliHLTVoidDataType;
  AliHLTComponent::SetDataType(bd.fDataType, dataType.c_str(), dataOrigin.c_str());
  bd.fSpecification = kAliHLTVoidDataSpec;

  // Set slice and partition
  AliHLTTPCDefinitions::EncodeDataSpecification(currentSlice, currentSlice, currentPartition, currentPartition);

  // Add the AliHLTComponentBlockData object to AliHLTTPCSpacePointContainer
  int numberOfClusters = spacepoints->AddInputBlock(&bd);

  // cout << *spacepoints << endl;

  // Retrieve the cluster information from AliHLTTPCSpacePointContainer
  std::vector<AliHLTUInt32_t> clusterIDs;
  spacepoints->GetClusterIDs(clusterIDs);

  // Append the cluster IDs and their X, Y and Z coordinates to the clusterCoordinates vector
  for (vector<AliHLTUInt32_t>::const_iterator element = clusterIDs.begin(); element != clusterIDs.end(); element++) {
    AliHLTUInt32_t clusterID = *element;

    addDataToCoordinatesVector(clusterID, spacepoints->GetX(clusterID), spacepoints->GetY(clusterID),
                               spacepoints->GetZ(clusterID));
  }

  // De-allocate memory space
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

  // Traverse the filesystem and execute processData for each cluster file found
  if (boost::filesystem::exists(dataPath) && boost::filesystem::is_directory(dataPath)) {
    for (boost::filesystem::directory_iterator directoryIterator(dataPath); directoryIterator != endIterator;
         ++directoryIterator) {
      if (boost::filesystem::is_regular_file(directoryIterator->status())) {
        totalNumberOfClusters += processData(directoryIterator->path().string(), dataType, dataOrigin);
        totalNumberOfDataFiles++;
      }
    }
  } else {
    std::cerr << "Path " << dataPath.string() << "/ could not be found or does not contain any valid data files"
              << endl;
    exit(1);
  }

  cout << "Added " << totalNumberOfClusters << " clusters from " << totalNumberOfDataFiles << " data files" << endl;

  printData(totalNumberOfClusters);

  return 0;
}
