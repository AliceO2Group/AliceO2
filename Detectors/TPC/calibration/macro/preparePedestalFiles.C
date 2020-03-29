// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <tuple>

#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"

#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Utils.h"
#endif

struct LinkInfo {
  LinkInfo(int cru, int link) : cru(cru), globalLinkID(link) {}
  int cru{0};
  int globalLinkID{0};

  bool operator<(const LinkInfo& other) const
  {
    if (cru < other.cru) {
      return true;
    }
    if ((cru == other.cru) && (globalLinkID < other.globalLinkID)) {
      return true;
    }
    return false;
  }
};

using ValueArray = std::array<uint32_t, 80>;
using DataMap = std::map<LinkInfo, ValueArray>;

void writeValues(const std::string_view fileName, const DataMap& map);
int getHWChannel(int sampa, int channel, int regionIter);

template <uint32_t DataBitSizeT = 12, uint32_t SignificantBitsT = 2>
uint32_t getADCValue(float value)
{
  static constexpr uint32_t DataBitSize = DataBitSizeT;                       ///< number of bits of the data representation
  static constexpr uint32_t SignificantBits = SignificantBitsT;               ///< number of bits used for floating point precision
  static constexpr uint64_t BitMask = ((uint64_t(1) << DataBitSize) - 1);     ///< mask for bits
  static constexpr float FloatConversion = 1.f / float(1 << SignificantBits); ///< conversion factor from integer representation to float

  const auto adc = uint32_t((value + 0.5f * FloatConversion) / FloatConversion) & BitMask;
  assert(std::abs(value - adc * FloatConversion) < FloatConversion);

  return adc;
}

void preparePedestalFiles(const std::string_view pedestalFileName, const std::string_view outputDir = "./", float sigmaNoise = 3, float minADC = 2, float pedestalOffset = 0)
{
  static constexpr float FloatConversion = 1.f / float(1 << 2);

  using namespace o2::tpc;
  const auto& mapper = Mapper::instance();

  TFile f(pedestalFileName.data());
  gROOT->cd();

  // ===| load noise and pedestal from file |===
  CalDet<float> output("Pedestals");
  CalDet<float>* calPedestal = nullptr;
  CalDet<float>* calNoise = nullptr;
  f.GetObject("Pedestals", calPedestal);
  f.GetObject("Noise", calNoise);

  DataMap pedestalValues;
  DataMap thresholdlValues;

  // ===| prepare values |===
  for (size_t iroc = 0; iroc < calPedestal->getData().size(); ++iroc) {
    const ROC roc(iroc);

    const auto& rocPedestal = calPedestal->getCalArray(iroc);
    const auto& rocNoise = calNoise->getCalArray(iroc);
    auto& rocOut = output.getCalArray(iroc);

    const int padOffset = (iroc > 35) ? mapper.getPadsInIROC() : 0;

    // skip empty
    if (!(std::abs(rocPedestal.getSum() + rocNoise.getSum()) > 0)) {
      continue;
    }

    //loop over pads
    for (size_t ipad = 0; ipad < rocPedestal.getData().size(); ++ipad) {
      const int globalPad = ipad + padOffset;
      const FECInfo& fecInfo = mapper.fecInfo(globalPad);
      const CRU cru = mapper.getCRU(roc.getSector(), globalPad);
      const uint32_t region = cru.region();
      const int cruID = cru.number();
      //int globalLinkID = fecInfo.getIndex();

      const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
      const int nFECs = partInfo.getNumberOfFECs();
      const int fecOffset = (nFECs + 1) / 2;
      const int fecInPartition = fecInfo.getIndex() - partInfo.getSectorFECOffset();
      const int dataWrapperID = fecInPartition >= fecOffset;
      const int globalLinkID = (fecInPartition % fecOffset) + dataWrapperID * 12;

      const float pedestal = rocPedestal.getValue(ipad);
      const float noise = rocNoise.getValue(ipad);
      const float threshold = std::max(sigmaNoise * noise, minADC);

      const int hwChannel = getHWChannel(fecInfo.getSampaChip(), fecInfo.getSampaChannel(), region % 2);
      //printf("%4d %4d %4d %4d %4d: %u\n", cru.number(), globalLinkID, hwChannel, fecInfo.getSampaChip(), fecInfo.getSampaChannel(), getADCValue(pedestal));

      const auto adcPedestal = getADCValue(pedestal);
      const auto adcThreshold = getADCValue(threshold);
      pedestalValues[LinkInfo(cruID, globalLinkID)][hwChannel] = adcPedestal;
      thresholdlValues[LinkInfo(cruID, globalLinkID)][hwChannel] = adcThreshold;
      rocOut.setValue(ipad, adcPedestal * FloatConversion);
      //if ((iroc == 1) && (ipad < 10)) {
        //const auto val = pedestalValues[LinkInfo(cruID, globalLinkID)][hwChannel];
        //printf("%4d %4d %4d %4d %4d %4d: %u (%u)\n", cru.number(), globalLinkID, ipad, hwChannel, fecInfo.getSampaChip(), fecInfo.getSampaChannel(), adcPedestal, val);
      //}
      //if ((cruID == 11) && (globalLinkID == 0)) {
      //const auto val = pedestalValues[LinkInfo(cruID, globalLinkID)][hwChannel];
      //printf("%4d %4d %4d %4d %4d: %u (%u)\n", cru.number(), globalLinkID, hwChannel, fecInfo.getSampaChip(), fecInfo.getSampaChannel(), getADCValue(pedestal), val);
      //}
    }
  }

  writeValues((outputDir + "/pedestal_values.txt").Data(), pedestalValues);
  writeValues((outputDir + "/threshold_values.txt").Data(), thresholdlValues);

  TFile fout("/tmp/outPed.direct.root", "recreate");
  fout.WriteObject(&output, "Pedestals");
}

/// return the hardware channel number as mapped in the CRU
//
int getHWChannel(int sampa, int channel, int regionIter)
{
  const int sampaOffet[5] = {0, 4, 8, 0, 4};
  if (regionIter && sampa == 2) {
    channel -= 16;
  }
  int outch = sampaOffet[sampa] + ((channel % 16) % 2) + 2 * (channel / 16) + (channel % 16) / 2 * 10;
  return outch;
}

/// write values of map to fileName
//
void writeValues(const std::string_view fileName, const DataMap& map)
{
  std::ofstream str(fileName.data(), std::ofstream::out);

  static constexpr float FloatConversion = 1.f / float(1 << 2);
  for (const auto& [linkInfo, data] : map) {
    int iter = 0;
    std::string values;
    for (const auto& val : data) {
      if (values.size()) {
        values += ",";
      }
      values += std::to_string(val);
      //printf("%3d: %4d (%.2f),\n", iter, val, val * FloatConversion);
      ++iter;
    }
    //printf("%4d %4d %s\n", linkInfo.cru, linkInfo.globalLinkID, values.data());
    //printf("\n\n");

    str << linkInfo.cru << " "
        << linkInfo.globalLinkID << " "
        << values << "\n";
  }
}

std::tuple<int, int> getSampaInfo(int hwChannel, int cruID)
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
void testChannelMapping(int cruID = 0)
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

    const size_t outch = getHWChannel(sampaOnFEC, channelOnSAMPA, regionIter);
    printf("%4zu %4d %4d : %4zu %s\n", outch, sampaOnFEC, channelOnSAMPA, ichannel, (outch != ichannel) ? "============" : "");
  }
}

o2::tpc::CalDet<float> getCalPad(const std::string_view fileName, const std::string_view outFile = "", std::string_view outName = "")
{
  using namespace o2::tpc;
  const auto& mapper = Mapper::instance();

  static constexpr float FloatConversion = 1.f / float(1 << 2);

  int cruID{0};
  int globalLinkID{0};
  int sampaOnFEC{0};
  int channelOnSAMPA{0};
  std::string values;
  CalDet<float> calPad(gSystem->BaseName(fileName.data()));

  std::string line;
  std::ifstream infile(fileName.data(), std::ifstream::in);
  if (!infile.is_open()) {
    std::cout << "could not open file " << fileName << "\n";
    return calPad;
  }

  while (std::getline(infile, line)) {
    std::stringstream streamLine(line);
    streamLine >> cruID >> globalLinkID >> values;

    const CRU cru(cruID);
    const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
    const int nFECs = partInfo.getNumberOfFECs();
    const int fecOffset = (nFECs + 1) / 2;
    const int fecInPartition = (globalLinkID < fecOffset) ? globalLinkID : fecOffset + globalLinkID % 12;
    //printf("%4d, %4d, %4d : %s\n", cruID, globalLinkID, fecInPartition, values.data());
    //printf("%4d, %4d, %4d\n", cruID, globalLinkID, fecInPartition);

    int hwChannel{0};
    for (const auto& val : utils::tokenize(values, ",")) {
      std::tie(sampaOnFEC, channelOnSAMPA) = getSampaInfo(hwChannel, cru);
      const PadROCPos padROCPos = mapper.padROCPos(cru, fecInPartition, sampaOnFEC, channelOnSAMPA);
      const float set = FloatConversion * std::stof(val);
      calPad.getCalArray(padROCPos.getROC()).setValue(padROCPos.getRow(), padROCPos.getPad(), set);
      //printf("%3d, %3d, %3d (%3d, %3d, %3d): %3d (%.2f)\n", hwChannel, sampaOnFEC, channelOnSAMPA, int(padROCPos.getROC()), padROCPos.getRow(), padROCPos.getPad(), std::stoi(val), set);
      ++hwChannel;
    }
  }

  if (outFile.size()) {
    TFile f(outFile.data(), "recreate");
    if (!outName.size()) {
      outName = calPad.getName();
    }
    f.WriteObject(&calPad, outName.data());
  }
  return calPad;
}

void debugDiff(std::string_view file1, std::string_view file2, std::string_view objName)
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

