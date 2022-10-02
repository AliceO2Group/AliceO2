// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HardwareClusterDecoder.cxx
/// \author David Rohr

#include "TPCReconstruction/HardwareClusterDecoder.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCBase/Mapper.h"
#include <algorithm>
#include <vector>
#include <numeric> // std::iota
#include <fairlogger/Logger.h>

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"

using namespace o2::tpc;
using namespace o2;
using namespace o2::dataformats;

int HardwareClusterDecoder::decodeClusters(std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>>& inputClusters,
                                           HardwareClusterDecoder::OutputAllocator outputAllocator,
                                           const std::vector<o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>>* inMCLabels,
                                           o2::dataformats::MCTruthContainer<o2::MCCompLabel>* outMCLabels)
{
  if (mIntegrator == nullptr) {
    mIntegrator.reset(new DigitalCurrentClusterIntegrator);
  }
  // MCLabelContainer does only allow appending new labels, so we need to write to separate
  // containers per {sector,padrow} and merge at the end;
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> outMCLabelContainers;
  if (!inMCLabels) {
    outMCLabels = nullptr;
  }
  ClusterNative* outputClusterBuffer = nullptr;
  // the number of clusters in a {sector,row}
  int nRowClusters[constants::MAXSECTOR][constants::MAXGLOBALPADROW] = {0};
  // offset of first cluster of {sector,row} in the output buffer
  size_t clusterOffsets[constants::MAXSECTOR][constants::MAXGLOBALPADROW] = {0};
  int containerRowCluster[constants::MAXSECTOR][constants::MAXGLOBALPADROW] = {0};
  Mapper& mapper = Mapper::instance();
  int numberOfOutputContainers = 0;
  for (int loop = 0; loop < 2; loop++) {
    int nTotalClusters = 0;
    for (int i = 0; i < inputClusters.size(); i++) {
      if (outMCLabels && inputClusters[i].second > 1) {
        LOG(error) << "Decoding of ClusterHardware to ClusterNative with MC labels is yet only support for single 8kb pages of ClusterHardwareContainer\n";
        return (1);
      }
      for (int j = 0; j < inputClusters[i].second; j++) {
        const char* tmpPtr = reinterpret_cast<const char*>(inputClusters[i].first);
        tmpPtr += j * 8192; //TODO: FIXME: Compute correct offset based on the size of the actual packet in the RDH
        const ClusterHardwareContainer& cont = *(reinterpret_cast<const ClusterHardwareContainer*>(tmpPtr));
        const CRU cru(cont.CRU);
        const Sector sector = cru.sector();
        const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
        const int rowOffset = region.getGlobalRowOffset();

        // TODO: make sure that input clusters are sorted in ascending row, so we
        // can write the MCLabels directly in consecutive order following the cluster sequence
        // Note: also the sorting below would need to be adjusted.
        for (int k = 0; k < cont.numberOfClusters; k++) {
          const int padRowGlobal = rowOffset + cont.clusters[k].getRow();
          int& nCls = nRowClusters[sector][padRowGlobal];
          if (loop == 1) {
            //Fill cluster in the respective output buffer
            const ClusterHardware& cIn = cont.clusters[k];
            ClusterNative& cOut = outputClusterBuffer[clusterOffsets[sector][padRowGlobal] + nCls];
            float pad = cIn.getPad();
            cOut.setPad(pad);
            cOut.setTimeFlags(cIn.getTimeLocal() + cont.timeBinOffset, cIn.getFlags());
            cOut.setSigmaPad(std::sqrt(cIn.getSigmaPad2()));
            cOut.setSigmaTime(std::sqrt(cIn.getSigmaTime2()));
            cOut.qMax = cIn.getQMax();
            cOut.qTot = cIn.getQTot();
            mIntegrator->integrateCluster(sector, padRowGlobal, pad, cIn.getQTot());
            if (outMCLabels) {
              auto& mcOut = outMCLabelContainers[containerRowCluster[sector][padRowGlobal]];
              for (const auto& element : (*inMCLabels)[i].getLabels(k)) {
                mcOut.addElement(nCls, element);
              }
            }
          } else {
            //Count how many output buffers we need (and how large they are below)
            if (nCls == 0) {
              numberOfOutputContainers++;
            }
          }
          nCls++;
          nTotalClusters++;
        }
      }
    }
    if (loop == 1) {
      //We are done with filling the buffers, sort all output buffers
      for (int i = 0; i < constants::MAXSECTOR; i++) {
        for (int j = 0; j < constants::MAXGLOBALPADROW; j++) {
          if (nRowClusters[i][j] == 0) {
            continue;
          }
          if (outMCLabels) {
            sortClustersAndMC(outputClusterBuffer + clusterOffsets[i][j], nRowClusters[i][j], outMCLabelContainers[containerRowCluster[i][j]]);
          } else {
            auto* cl = outputClusterBuffer + clusterOffsets[i][j];
            std::sort(cl, cl + nRowClusters[i][j]);
          }
        }
      }
    } else {
      //Now we know the size of all output buffers, allocate them
      if (outMCLabels) {
        outMCLabelContainers.resize(numberOfOutputContainers);
      }
      size_t rawOutputBufferSize = sizeof(ClusterCountIndex) + nTotalClusters * sizeof(ClusterNative);
      char* rawOutputBuffer = outputAllocator(rawOutputBufferSize);
      auto& clusterCounts = *(reinterpret_cast<ClusterCountIndex*>(rawOutputBuffer));
      outputClusterBuffer = reinterpret_cast<ClusterNative*>(rawOutputBuffer + sizeof(ClusterCountIndex));
      nTotalClusters = 0;
      numberOfOutputContainers = 0;
      for (int i = 0; i < constants::MAXSECTOR; i++) {
        for (int j = 0; j < constants::MAXGLOBALPADROW; j++) {
          clusterCounts.nClusters[i][j] = nRowClusters[i][j];
          if (nRowClusters[i][j] == 0) {
            continue;
          }
          containerRowCluster[i][j] = numberOfOutputContainers++;
          clusterOffsets[i][j] = nTotalClusters;
          nTotalClusters += nRowClusters[i][j];
          mIntegrator->initRow(i, j);
        }
      }
      memset(nRowClusters, 0, sizeof(nRowClusters));
    }
  }
  // Finally merge MC label containers into one container following the cluster sequence in the
  // output buffer
  if (outMCLabels) {
    auto& labels = *outMCLabels;
    int nCls = 0;
    for (int i = 0; i < constants::MAXSECTOR; i++) {
      for (int j = 0; j < constants::MAXGLOBALPADROW; j++) {
        if (nRowClusters[i][j] == 0) {
          continue;
        }
        for (int k = 0, end = outMCLabelContainers[containerRowCluster[i][j]].getIndexedSize(); k < end; k++, nCls++) {
          assert(end == nRowClusters[i][j]);
          assert(clusterOffsets[i][j] + k == nCls);
          for (const auto& element : outMCLabelContainers[containerRowCluster[i][j]].getLabels(k)) {
            labels.addElement(nCls, element);
          }
        }
      }
    }
  }
  return (0);
}

void HardwareClusterDecoder::sortClustersAndMC(ClusterNative* clusters, size_t nClusters,
                                               o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcTruth)
{
  std::vector<unsigned int> indizes(nClusters);
  std::iota(indizes.begin(), indizes.end(), 0);
  std::sort(indizes.begin(), indizes.end(), [&clusters](const auto a, const auto b) {
    return clusters[a] < clusters[b];
  });
  std::vector<ClusterNative> buffer(clusters, clusters + nClusters);
  ClusterNativeHelper::MCLabelContainer tmpMC = std::move(mcTruth);
  assert(mcTruth.getIndexedSize() == 0);
  for (int i = 0; i < nClusters; i++) {
    clusters[i] = buffer[indizes[i]];
    for (auto const& label : tmpMC.getLabels(indizes[i])) {
      mcTruth.addElement(i, label);
    }
  }
}
