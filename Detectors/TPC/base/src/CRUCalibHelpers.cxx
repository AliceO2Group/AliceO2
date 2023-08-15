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

#include <fstream>
#include <iostream>
#include <numeric>

#include "TROOT.h"

#include "Framework/Logger.h"
#include "TPCBase/CRUCalibHelpers.h"

using namespace o2::tpc;

/// return the hardware channel number as mapped in the CRU
int cru_calib_helpers::getHWChannel(int sampa, int channel, int regionIter)
{
  const int sampaOffet[5] = {0, 4, 8, 0, 4};
  if (regionIter && sampa == 2) {
    channel -= 16;
  }
  int outch = sampaOffet[sampa] + ((channel % 16) % 2) + 2 * (channel / 16) + (channel % 16) / 2 * 10;
  return outch;
}

/// convert HW mapping to sampa and channel number
std::tuple<int, int> cru_calib_helpers::getSampaInfo(int hwChannel, int cruID)
{
  static constexpr int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
  static constexpr int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};
  const int regionIter = cruID % 2;

  const int istreamm = ((hwChannel % 10) / 2);
  const int partitionStream = istreamm + regionIter * 5;
  const int sampaOnFEC = sampaMapping[partitionStream];
  const int channel = (hwChannel % 2) + 2 * (hwChannel / 10);
  const int channelOnSAMPA = channel + channelOffset[partitionStream];

  return {sampaOnFEC, channelOnSAMPA};
}

/// Test input channel mapping vs output channel mapping
///
/// Consistency check of mapping
void cru_calib_helpers::testChannelMapping(int cruID)
{
  const int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
  const int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};
  const int regionIter = cruID % 2;

  for (std::size_t ichannel = 0; ichannel < 80; ++ichannel) {
    const int istreamm = ((ichannel % 10) / 2);
    const int partitionStream = istreamm + regionIter * 5;
    const int sampaOnFEC = sampaMapping[partitionStream];
    const int channel = (ichannel % 2) + 2 * (ichannel / 10);
    const int channelOnSAMPA = channel + channelOffset[partitionStream];

    const size_t outch = cru_calib_helpers::getHWChannel(sampaOnFEC, channelOnSAMPA, regionIter);
    printf("%4zu %4d %4d : %4zu %s\n", outch, sampaOnFEC, channelOnSAMPA, ichannel, (outch != ichannel) ? "============" : "");
  }
}

/// debug differences between two cal pad objects
void cru_calib_helpers::debugDiff(std::string_view file1, std::string_view file2, std::string_view objName)
{
  using namespace o2::tpc;
  CalPad dummy;
  CalPad* calPad1{nullptr};
  CalPad* calPad2{nullptr};

  std::unique_ptr<TFile> tFile1(TFile::Open(file1.data()));
  std::unique_ptr<TFile> tFile2(TFile::Open(file2.data()));
  gROOT->cd();

  tFile1->GetObject(objName.data(), calPad1);
  tFile2->GetObject(objName.data(), calPad2);

  for (size_t iroc = 0; iroc < calPad1->getData().size(); ++iroc) {
    const auto& calArray1 = calPad1->getCalArray(iroc);
    const auto& calArray2 = calPad2->getCalArray(iroc);
    // skip empty
    if (!(std::abs(calArray1.getSum() + calArray2.getSum()) > 0)) {
      continue;
    }

    for (size_t ipad = 0; ipad < calArray1.getData().size(); ++ipad) {
      const auto val1 = calArray1.getValue(ipad);
      const auto val2 = calArray2.getValue(ipad);

      if (std::abs(val2 - val1) >= 0.25) {
        printf("%2zu %5zu : %.5f - %.5f = %.2f\n", iroc, ipad, val2, val1, val2 - val1);
      }
    }
  }
}

std::unordered_map<std::string, CalPad> cru_calib_helpers::preparePedestalFiles(const CalPad& pedestals, const CalPad& noise, float sigmaNoise, float minADC, float pedestalOffset, bool onlyFilled, bool maskBad, float noisyChannelThreshold, float sigmaNoiseNoisyChannels, float badChannelThreshold, bool fixedSize)
{
  const auto& mapper = Mapper::instance();

  std::unordered_map<std::string, CalPad> pedestalsThreshold;
  pedestalsThreshold["Pedestals"] = CalPad("Pedestals");
  pedestalsThreshold["ThresholdMap"] = CalPad("ThresholdMap");
  pedestalsThreshold["PedestalsPhys"] = CalPad("Pedestals");
  pedestalsThreshold["ThresholdMapPhys"] = CalPad("ThresholdMap");

  auto& pedestalsCRU = pedestalsThreshold["Pedestals"];
  auto& thresholdCRU = pedestalsThreshold["ThresholdMap"];

  // ===| prepare values |===
  for (size_t iroc = 0; iroc < pedestals.getData().size(); ++iroc) {
    const ROC roc(iroc);

    const auto& rocPedestal = pedestals.getCalArray(iroc);
    const auto& rocNoise = noise.getCalArray(iroc);

    const int padOffset = roc.isOROC() ? mapper.getPadsInIROC() : 0;
    const auto& traceLengths = roc.isIROC() ? mapper.getTraceLengthsIROC() : mapper.getTraceLengthsOROC();

    // skip empty
    if (!(std::abs(rocPedestal.getSum() + rocNoise.getSum()) > 0)) {
      continue;
    }

    // loop over pads
    for (size_t ipad = 0; ipad < rocPedestal.getData().size(); ++ipad) {
      const int globalPad = ipad + padOffset;
      const FECInfo& fecInfo = mapper.fecInfo(globalPad);
      const CRU cru = mapper.getCRU(roc.getSector(), globalPad);
      const uint32_t region = cru.region();
      const int cruID = cru.number();
      const int sampa = fecInfo.getSampaChip();
      const int sampaChannel = fecInfo.getSampaChannel();
      // int globalLinkID = fecInfo.getIndex();

      const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
      const int nFECs = partInfo.getNumberOfFECs();
      const int fecOffset = (nFECs + 1) / 2;
      const int fecInPartition = fecInfo.getIndex() - partInfo.getSectorFECOffset();
      // const int dataWrapperID = fecInPartition >= fecOffset;
      // const int globalLinkID = (fecInPartition % fecOffset) + dataWrapperID * 12;

      const auto traceLength = traceLengths[ipad];

      float pedestal = rocPedestal.getValue(ipad);
      if ((pedestal > 0) && (pedestalOffset > pedestal)) {
        LOGP(warning, "ROC: {:2}, pad: {:3} -- pedestal offset {:.2f} larger than the pedestal value {:.2f}. Pedestal and noise will be set to 0", iroc, ipad, pedestalOffset, pedestal);
      } else {
        pedestal -= pedestalOffset;
      }

      float noise = std::abs(rocNoise.getValue(ipad)); // it seems with the new fitting procedure, the noise can also be negative, since in gaus sigma is quadratic
      float noiseCorr = noise - (0.847601 + 0.031514 * traceLength);
      if ((pedestal <= 0) || (pedestal > 150) || (noise <= 0) || (noise > 50)) {
        LOGP(info, "Bad pedestal or noise value in ROC {:2}, CRU {:3}, fec in CRU: {:2}, SAMPA: {}, channel: {:2}, pedestal: {:.4f}, noise {:.4f}", iroc, cruID, fecInPartition, sampa, sampaChannel, pedestal, noise);
        if (maskBad) {
          pedestal = 1023;
          noise = 1023;
          LOGP(info, ", they will be masked using pedestal value {:.0f} and noise {:.0f}", pedestal, noise);
        } else {
          LOGP(info, ", setting both to 0");
          pedestal = 0;
          noise = 0;
        }
      }
      float threshold = (noise > 0) ? std::max(sigmaNoise * noise, minADC) : 0;
      threshold = std::min(threshold, 1023.f);
      float thresholdHighNoise = (noiseCorr > noisyChannelThreshold) ? std::max(sigmaNoiseNoisyChannels * noise, minADC) : threshold;

      float pedestalHighNoise = pedestal;
      if (noiseCorr > badChannelThreshold) {
        pedestalHighNoise = 1023;
        thresholdHighNoise = 1023;
      }

      const int hwChannel = getHWChannel(sampa, sampaChannel, region % 2);
      // for debugging
      // printf("%4d %4d %4d %4d %4d: %u\n", cru.number(), globalLinkID, hwChannel, fecInfo.getSampaChip(), fecInfo.getSampaChannel(), getADCValue(pedestal));

      // default thresholds
      if (fixedSize) {
        pedestal = floatToFixedSize(pedestal);
        threshold = floatToFixedSize(threshold);

        // higher thresholds for physics data taking
        pedestalHighNoise = floatToFixedSize(pedestalHighNoise);
        thresholdHighNoise = floatToFixedSize(thresholdHighNoise);
      }

      pedestalsThreshold["Pedestals"].getCalArray(iroc).setValue(ipad, pedestal);
      pedestalsThreshold["ThresholdMap"].getCalArray(iroc).setValue(ipad, threshold);
      pedestalsThreshold["PedestalsPhys"].getCalArray(iroc).setValue(ipad, pedestalHighNoise);
      pedestalsThreshold["ThresholdMapPhys"].getCalArray(iroc).setValue(ipad, thresholdHighNoise);
      // for debugging
      // if(!(std::abs(pedestal - fixedSizeToFloat(adcPedestal)) <= 0.5 * 0.25)) {
      // printf("%4d %4d %4d %4d %4d: %u %.2f %.4f %.4f\n", cru.number(), globalLinkID, hwChannel, sampa, sampaChannel, adcPedestal, fixedSizeToFloat(adcPedestal), pedestal, pedestal - fixedSizeToFloat(adcPedestal));
      //}
    }
  }

  return pedestalsThreshold;
}
