// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "FairLogger.h"

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
                                           const std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* inMCLabels,
                                           std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* outMCLabels)
{
  if (mIntegrator == nullptr)
    mIntegrator.reset(new DigitalCurrentClusterIntegrator);
  if (!inMCLabels)
    outMCLabels = nullptr;
  std::vector<ClusterNativeBuffer*> outputBufferMap;
  int nRowClusters[Constants::MAXSECTOR][Constants::MAXGLOBALPADROW] = {0};
  int containerRowCluster[Constants::MAXSECTOR][Constants::MAXGLOBALPADROW] = {0};
  Mapper& mapper = Mapper::instance();
  int numberOfOutputContainers = 0;
  for (int loop = 0; loop < 2; loop++) {
    int nTotalClusters = 0;
    for (int i = 0; i < inputClusters.size(); i++) {
      if (outMCLabels && inputClusters[i].second > 1) {
        LOG(ERROR) << "Decoding of ClusterHardware to ClusterNative with MC labels is yet only support for single 8kb pages of ClusterHardwareContainer\n";
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

        for (int k = 0; k < cont.numberOfClusters; k++) {
          const int padRowGlobal = rowOffset + cont.clusters[k].getRow();
          int& nCls = nRowClusters[sector][padRowGlobal];
          if (loop == 1) {
            //Fill cluster in the respective output buffer
            const ClusterHardware& cIn = cont.clusters[k];
            ClusterNative& cOut = outputBufferMap[containerRowCluster[sector][padRowGlobal]]->clusters[nCls];
            float pad = cIn.getPad();
            cOut.setPad(pad);
            cOut.setTimeFlags(cIn.getTimeLocal() + cont.timeBinOffset, cIn.getFlags());
            cOut.setSigmaPad(std::sqrt(cIn.getSigmaPad2()));
            cOut.setSigmaTime(std::sqrt(cIn.getSigmaTime2()));
            cOut.qMax = cIn.getQMax();
            cOut.qTot = cIn.getQTot();
            mIntegrator->integrateCluster(sector, padRowGlobal, pad, cIn.getQTot());
            if (outMCLabels) {
              auto& mcOut = (*outMCLabels)[containerRowCluster[sector][padRowGlobal]];
              for (const auto& element : (*inMCLabels)[i].getLabels(k)) {
                mcOut.addElement(nCls, element);
              }
            }
          } else {
            //Count how many output buffers we need (and how large they are below)
            if (nCls == 0)
              numberOfOutputContainers++;
          }
          nCls++;
          nTotalClusters++;
        }
      }
    }
    if (loop == 1) {
      //We are done with filling the buffers, sort all output buffers
      for (int i = 0; i < outputBufferMap.size(); i++) {
        if (outMCLabels) {
          sortClustersAndMC(outputBufferMap[i]->clusters, outputBufferMap[i]->nClusters, (*outMCLabels)[i]);
        } else {
          auto* cl = outputBufferMap[i]->clusters;
          std::sort(cl, cl + outputBufferMap[i]->nClusters);
        }
      }
    } else {
      //Now we know the size of all output buffers, allocate them
      if (outMCLabels)
        outMCLabels->resize(numberOfOutputContainers);
      size_t rawOutputBufferSize = numberOfOutputContainers * sizeof(ClusterNativeBuffer) + nTotalClusters * sizeof(ClusterNative);
      char* rawOutputBuffer = outputAllocator(rawOutputBufferSize);
      char* rawOutputBufferIterator = rawOutputBuffer;
      numberOfOutputContainers = 0;
      for (int i = 0; i < Constants::MAXSECTOR; i++) {
        for (int j = 0; j < Constants::MAXGLOBALPADROW; j++) {
          if (nRowClusters[i][j] == 0)
            continue;
          outputBufferMap.push_back(reinterpret_cast<ClusterNativeBuffer*>(rawOutputBufferIterator));
          ClusterNativeBuffer& container = *outputBufferMap.back();
          container.sector = i;
          container.globalPadRow = j;
          container.nClusters = nRowClusters[i][j];
          containerRowCluster[i][j] = numberOfOutputContainers++;
          rawOutputBufferIterator += container.getFlatSize();
          mIntegrator->initRow(i, j);
        }
      }
      assert(rawOutputBufferIterator == rawOutputBuffer + rawOutputBufferSize);
      memset(nRowClusters, 0, sizeof(nRowClusters));
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
  std::vector<unsigned int> actual = indizes;
  MCLabelContainer tmpMC = std::move(mcTruth);
  assert(mcTruth.getIndexedSize() == 0);
  for (int i = 0; i < nClusters; i++) {
    ClusterNative backup = clusters[i];
    clusters[i] = clusters[actual[indizes[i]]];
    clusters[actual[indizes[i]]] = backup;
    auto tmp = actual[i];
    actual[i] = actual[indizes[i]];
    actual[indizes[i]] = tmp;
    for (auto const& label : tmpMC.getLabels(indizes[i])) {
      mcTruth.addElement(i, label);
    }
  }
}
