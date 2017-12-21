// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TimeframeValidatorDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-07
/// @brief  Validator device for a full time frame

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "DataFlow/TimeframeValidatorDevice.h"
#include "TimeFrame/TimeFrame.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/DataHeader.h"

#include <options/FairMQProgOptions.h>

using DataHeader = o2::header::DataHeader;
using DataOrigin = o2::header::DataOrigin;
using DataDescription = o2::header::DataDescription;
using IndexElement = o2::DataFormat::IndexElement;

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

    auto indexHeader = o2::header::get<header::DataHeader>(timeframeParts.At(timeframeParts.Size() - 2)->GetData());
    // FIXME: Provide iterator pair API for the index
    //        Index should really be something which provides an
    //        iterator pair API so that we can sort / find / lower_bound
    //        easily. Right now we simply use it a C-style array.
    auto index = reinterpret_cast<IndexElement*>(timeframeParts.At(timeframeParts.Size() - 1)->GetData());

    // TODO: fill this with checks on time frame
    LOG(INFO) << "This time frame has " << timeframeParts.Size() << " parts.\n";
    auto indexEntries = indexHeader->payloadSize / sizeof(DataHeader);
    if (indexHeader->dataDescription != DataDescription("TIMEFRAMEINDEX"))
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
      LOG(DEBUG) << ie.first.dataDescription.str << " "
                 << ie.first.dataOrigin.str << std::endl;
      if ((ie.first.dataOrigin == header::gDataOriginTPC)
          && (ie.first.dataDescription == header::gDataDescriptionClusters)) {
        tpcIndex = ie.second;
      }
      if ((ie.first.dataOrigin == header::gDataOriginITS)
          && (ie.first.dataDescription == header::gDataDescriptionClusters)) {
        itsIndex = ie.second;
      }
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
    if ((tpcHeader->dataDescription != header::gDataDescriptionClusters) ||
        (tpcHeader->dataOrigin != header::gDataOriginTPC))
    {
      LOG(ERROR) << "Wrong data description. Expecting TPC - CLUSTERS, found "
                 << tpcHeader->dataOrigin.str << " - "
                 << tpcHeader->dataDescription.str << "\n";
      continue;
    }
    auto tpcPayload = reinterpret_cast<TPCTestCluster *>(timeframeParts.At(tpcIndex + 1)->GetData());
    if (tpcHeader->payloadSize % sizeof(TPCTestCluster))
      LOG(ERROR) << "TPC - CLUSTERS Size Mismatch\n";
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
    if ((itsHeader->dataDescription != header::gDataDescriptionClusters)
        || (itsHeader->dataOrigin != header::gDataOriginITS))
    {
      LOG(ERROR) << "Wrong data description. Expecting ITS - CLUSTERS, found "
                 << itsHeader->dataOrigin.str << " - " << itsHeader->dataDescription.str << "\n";
      continue;
    }
    auto itsPayload = reinterpret_cast<ITSRawData*>(timeframeParts.At(itsIndex + 1)->GetData());
    if (itsHeader->payloadSize % sizeof(ITSRawData))
      LOG(ERROR) << "ITS - CLUSTERS Size Mismatch.\n";
    numOfClusters = itsHeader->payloadSize / sizeof(ITSRawData);
    for (size_t ci = 0 ; ci < numOfClusters; ++ci)
    {
      ITSRawData &cluster = itsPayload[ci];
      if (cluster.timeStamp != ci)
      {
        LOG(ERROR) << "ITS Data mismatch. Expecting " << ci
        << " got " << cluster.timeStamp << "\n";
        break;
      }
    }
    LOG(INFO) << "Everything is fine with received timeframe\n";

  }
}
