/// @file   TimeframeValidatorDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-07
/// @brief  Validator device for a full time frame

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "DataFlow/TimeframeValidatorDevice.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/DataHeader.h"
#include <options/FairMQProgOptions.h>


using DataHeader = o2::Header::DataHeader;

// FIXME: this should really be in a central place
using PartPosition = int;
typedef std::pair<o2::Header::DataHeader, PartPosition> IndexElement;

o2::DataFlow::TimeframeValidatorDevice::TimeframeValidatorDevice()
  : O2Device()
  , mInChannelName()
{
}

void o2::DataFlow::TimeframeValidatorDevice::InitTask()
{
  mInChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
}

void o2::DataFlow::TimeframeValidatorDevice::Run()
{
  while (CheckCurrentState(RUNNING)) {
    FairMQParts timeframeParts;
    if (Receive(timeframeParts, mInChannelName, 0, 100) <= 0)
      continue;

    if (timeframeParts.Size() < 2)
      LOG(ERROR) << "Expecting at least 2 parts\n";

    auto indexHeader = o2::Header::get<Header::DataHeader>(timeframeParts.At(timeframeParts.Size() - 2)->GetData());
    // FIXME: Provide iterator pair API for the index
    //        Index should really be something which provides an
    //        iterator pair API so that we can sort / find / lower_bound
    //        easily. Right now we simply use it a C-style array.
    auto index = reinterpret_cast<IndexElement*>(timeframeParts.At(timeframeParts.Size() - 1)->GetData());

    // TODO: fill this with checks on time frame
    LOG(INFO) << "This time frame has " << timeframeParts.Size() << " parts.\n";
    auto indexEntries = indexHeader->payloadSize / sizeof(IndexElement);
    if (strncmp(indexHeader->dataDescription.str, "TIMEFRAMEINDEX", 14) != 0)
      LOG(ERROR) << "Could not find a valid index header\n";
    LOG(INFO) << indexHeader->dataDescription.str << "\n";
    LOG(INFO) << "This time frame has " << indexEntries << "entries in the index.\n";
    if ((indexEntries * 2 + 2) != (timeframeParts.Size()))
      LOG(ERROR) << "Mismatched index and received parts\n";

    // - Use the index to find out if we have TPC data
    // - Get the part with the TPC data
    // - Validate TPCCluster dummy data
    // - Validate ITSRaw dummy data
    int tpcIndex = -1;
    int itsIndex = -1;

    for (int ii = 0; ii < indexEntries; ++ii) {
      IndexElement &ie = index[ii];
      assert(ie.second >= 0);
      LOG(DEBUG) << ie.first.dataDescription.str << std::endl;
      if (ie.first.dataDescription == "TPCCLUSTER")
        tpcIndex = ie.second;
      if (ie.first.dataDescription == "ITSRAW")
        itsIndex = ie.second;
    }

    if (tpcIndex < 0)
    {
      LOG(ERROR) << "Could not find expected TPC payload\n";
      continue;
    }
    if (itsIndex < 0)
    {
      LOG(ERROR) << "Could not find expected ITS payload\n";
      continue;
    }
    LOG(DEBUG) << "TPC Index " << tpcIndex << "\n";
    LOG(DEBUG) << "ITS Index " << itsIndex << "\n";

    // Data header it at position - 1
    auto tpcHeader = reinterpret_cast<DataHeader *>(timeframeParts.At(tpcIndex)->GetData());
    if (tpcHeader->dataDescription != "TPCCLUSTER")
    {
      LOG(ERROR) << "Wrong data description. Expecting TPCCLUSTER, found " << tpcHeader->dataDescription.str << "\n";
      continue;
    }
    auto tpcPayload = reinterpret_cast<TPCTestCluster *>(timeframeParts.At(tpcIndex + 1)->GetData());
    if (tpcHeader->payloadSize % sizeof(TPCTestCluster))
      LOG(ERROR) << "TPCCLUSTER Size Mismatch\n";
    auto numOfClusters = tpcHeader->payloadSize / sizeof(TPCTestCluster);
    for (size_t ci = 0 ; ci < numOfClusters; ++ci)
    {
      TPCTestCluster &cluster = tpcPayload[ci];
      if (cluster.z != 1.5)
      {
        LOG(ERROR) << "TPC Data mismatch. Expecting z = 1.5 got " << cluster.z << "\n";
        break;
      }
      if (cluster.timeStamp != ci)
      {
        LOG(ERROR) << "TPC Data mismatch. Expecting " << ci << " got " << cluster.timeStamp << "\n";
        break;
      }
    }

    // Data header it at position - 1
    auto itsHeader = reinterpret_cast<DataHeader *>(timeframeParts.At(itsIndex)->GetData());
    if (strcmp(itsHeader->dataDescription.str,"ITSRAW")!=0)
    {
      LOG(ERROR) << "Wrong data description. Expecting ITSRAW, found " << itsHeader->dataDescription.str << "\n";
      continue;
    }
    auto itsPayload = reinterpret_cast<ITSRawData*>(timeframeParts.At(itsIndex + 1)->GetData());
    if (itsHeader->payloadSize % sizeof(ITSRawData))
      LOG(ERROR) << "ITSRawData Size Mismatch.\n";
    numOfClusters = itsHeader->payloadSize / sizeof(ITSRawData);
    for (size_t ci = 0 ; ci < numOfClusters; ++ci)
    {
      ITSRawData &cluster = itsPayload[ci];
      if (cluster.timeStamp != ci)
      {
        LOG(ERROR) << "ITS Data mismatch. Expecting " << ci << " got " << cluster.timeStamp << "\n";
        break;
      }
    }
    LOG(INFO) << "Everything is fine with received timeframe\n";

  }
}
