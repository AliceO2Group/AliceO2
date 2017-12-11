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
#include "TPCBase/Constants.h"
#include "TPCBase/Mapper.h"
#include <algorithm>
#include <vector>
#include "FairLogger.h"

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"

using namespace o2::TPC;
using namespace o2::DataFormat::TPC;
using namespace o2;
using namespace o2::dataformats;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

int HardwareClusterDecoder::decodeClusters(std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>>& inputClusters, std::vector<ClusterNativeContainer>& outputClusters, const std::vector<MCLabelContainer>* inMCLabels, std::vector<MCLabelContainer>* outMCLabels)
{
  if (!inMCLabels) outMCLabels = nullptr;
  int nRowClusters[Constants::MAXSECTOR][Constants::MAXGLOBALPADROW] = {0};
  int containerRowCluster[Constants::MAXSECTOR][Constants::MAXGLOBALPADROW] =  {0};
  Mapper& mapper = Mapper::instance();
  int numberOfOutputContainers = 0;
  for (int loop = 0;loop < 2;loop++)
  {
    for (int i = 0;i < inputClusters.size();i++)
    {
      if (outMCLabels && inputClusters[i].second > 1)
      {
        LOG(ERROR) << "Decoding of ClusterHardware to ClusterNative with MC labels is yet only support for single 8kb pages of ClusterHardwareContainer\n";
        return(1);
      }
      for (int j = 0;j < inputClusters[i].second;j++)
      {
        const char* tmpPtr = reinterpret_cast<const char*> (inputClusters[i].first);
        tmpPtr += j * 8192; //TODO: FIXME: Compute correct offset based on the size of the actual packet in the RDH
        const ClusterHardwareContainer& cont = *(reinterpret_cast<const ClusterHardwareContainer*> (tmpPtr));
        const CRU cru(cont.mCRU);
        const Sector sector = cru.sector();
        const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
        const int rowOffset = region.getGlobalRowOffset();

        for (int k = 0;k < cont.mNumberOfClusters;k++)
        {
          const int padRowGlobal = rowOffset + cont.mClusters[k].mRow;
          int& nCls = nRowClusters[sector][padRowGlobal];
          if (loop == 1)
          {
            //Fill cluster in the respective output buffer
            const ClusterHardware& cIn = cont.mClusters[k];
            ClusterNative& cOut = outputClusters[containerRowCluster[sector][padRowGlobal]].mClusters[nCls];
            MCLabelContainer& mcOut = (*outMCLabels)[containerRowCluster[sector][padRowGlobal]];
            float pad = cIn.getPad();
            cOut.setPad(pad);
            cOut.setTimeFlags(cIn.getTimeLocal() + cont.mTimeBinOffset, cIn.mFlags);
            cOut.setSigmaPad(std::sqrt(cIn.getSigmaPad2()));
            cOut.setSigmaTime(std::sqrt(cIn.getSigmaTime2()));
            cOut.mQMax = cIn.mQMax;
            cOut.mQTot = cIn.mQTot;
            mIntegrator.integrateCluster(sector, padRowGlobal, pad, cIn.mQTot);
            if (outMCLabels)
            {
                gsl::span<const MCCompLabel> mcArray = (*inMCLabels)[i].getLabels(k);
                for (int l = 0;l < mcArray.size();l++) {
                  mcOut.addElement(nCls, mcArray[l]);
                }
            }
          }
          else
          {
            //Count how many output buffers we need (and how large they are below)
            if (nCls == 0) numberOfOutputContainers++;
          }
          nCls++;
        }
      }
    }
    if (loop == 1)
    {
      //We are done with filling the buffers, sort all output buffers
      for (int i = 0;i < outputClusters.size();i++)
      {
        if (outMCLabels) {
          sortClustersAndMC(outputClusters[i].mClusters, (*outMCLabels)[i]);
        } else {
          auto& cl = outputClusters[i].mClusters;
          std::sort(cl.data(), cl.data() + cl.size(), ClusterNativeContainer::sortComparison);
        }
      }
    }
    else
    {
      //Now we know the size of all output buffers, allocate them
      outputClusters.resize(numberOfOutputContainers);
      if (outMCLabels) outMCLabels->resize(numberOfOutputContainers);
      numberOfOutputContainers = 0;
      for (int i = 0;i < Constants::MAXSECTOR;i++)
      {
        for (int j = 0;j < Constants::MAXGLOBALPADROW;j++)
        {
          if (nRowClusters[i][j] == 0) continue;
          outputClusters[numberOfOutputContainers].mClusters.resize(nRowClusters[i][j]);
          outputClusters[numberOfOutputContainers].mSector = i;
          outputClusters[numberOfOutputContainers].mGlobalPadRow = j;
          containerRowCluster[i][j] = numberOfOutputContainers++;
          mIntegrator.initRow(i, j);
        }
      }
      memset(nRowClusters, 0, sizeof(nRowClusters));
    }
  }
  return(0);
}

void HardwareClusterDecoder::sortClustersAndMC(std::vector<ClusterNative> clusters, MCLabelContainer mcTruth)
{
  std::vector<unsigned int> indizes(clusters.size());
  for (int i = 0;i < indizes.size();i++) indizes[i] = i;
  std::sort(indizes.data(), indizes.data() + indizes.size(), [&clusters](const auto a, const auto b) {
    return ClusterNativeContainer::sortComparison(clusters[a], clusters[b]);
  });
  std::vector<ClusterNative> tmpCl = std::move(clusters);
  MCLabelContainer tmpMC = std::move(mcTruth);
  for (int i = 0;i < clusters.size();i++)
  {
    clusters[i] = tmpCl[indizes[i]];
    gsl::span<const MCCompLabel> mcArray = tmpMC.getLabels(indizes[i]);
    for (int k = 0;k < mcArray.size();k++)
    {
      mcTruth.addElement(i, mcArray[k]);
    }
  }
}
