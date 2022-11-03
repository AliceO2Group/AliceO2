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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string_view>
#include <string>

#include "TROOT.h"
#include "TMath.h"
#include "TFile.h"

#include "Framework/Logger.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Utils.h"
#include "TPCBase/CRUCalibHelpers.h"
#endif

using namespace o2::tpc::cru_calib_helpers;
using namespace o2::tpc;

void prepareCMFiles(const std::string_view pulserFile, std::string outputDir = "./")
{
  constexpr uint32_t DataBits = 8;
  constexpr uint32_t FractionalBits = 6;
  constexpr float MaxVal = float((1 << DataBits) - 1) / (1 << FractionalBits);

  const auto& mapper = Mapper::instance();

  // ===| load noise and pedestal from file |===
  CalDet<float> output("CMkValues");
  CalDet<float>* pulserQtot{nullptr};
  const CDBInterface::CalPadMapType* calPulser = nullptr;

  if (pulserFile.find("cdb") != std::string::npos) {
    auto& cdb = CDBInterface::instance();
    if (pulserFile.find("cdb-test") == 0) {
      cdb.setURL("http://ccdb-test.cern.ch:8080");
    } else if (pulserFile.find("cdb-prod") == 0) {
      cdb.setURL("https://alice-ccdb.cern.ch");
    }
    const auto timePos = pulserFile.find("@");
    if (timePos != std::string_view::npos) {
      std::cout << "set time stamp " << std::stol(pulserFile.substr(timePos + 1).data()) << "\n";
      cdb.setTimeStamp(std::stol(pulserFile.substr(timePos + 1).data()));
    }
    auto& pulserData = cdb.getObjectFromCDB<CDBInterface::CalPadMapType>(CDBTypeMap.at(CDBType::CalPulser));
    pulserQtot = &pulserData.at("Qtot");
  } else {
    auto calPads = utils::readCalPads(pulserFile, "Qtot");
    pulserQtot = calPads[0];
  }

  // normalize to <Qtot> per GEM stack
  for (size_t i = 0; i < pulserQtot->getData().size(); ++i) {
    auto& calROC = pulserQtot->getCalArray(i);
    auto& data = calROC.getData();
    if (i < 36) {
      const auto median = float(TMath::Mean(data.size(), data.data()));
      calROC /= median;
    } else {
      const std::vector<int> pads{Mapper::getPadsInOROC1(), Mapper::getPadsInOROC2(), Mapper::getPadsInOROC3()};
      auto median = TMath::Mean(pads[0], data.data());
      std::for_each(data.data(), data.data() + pads[0], [median](auto& val) { val /= (val > 0) ? median : 1; });

      median = TMath::Mean(pads[1], data.data() + pads[0]);
      std::for_each(data.data() + pads[0], data.data() + pads[0] + pads[1], [median](auto& val) { val /= (val > 0) ? median : 1; });

      median = TMath::Mean(pads[2], data.data() + pads[0] + pads[1]);
      std::for_each(data.data() + pads[0] + pads[1], data.data() + pads[0] + pads[1] + pads[2], [median](auto& val) { val /= (val > 0) ? median : 1; });
    }
  }

  DataMapU32 commonModeKValues;

  // ===| prepare values |===
  for (size_t iroc = 0; iroc < pulserQtot->getData().size(); ++iroc) {
    const ROC roc(iroc);

    const auto& rocPulserQtot = pulserQtot->getCalArray(iroc);
    auto& rocOut = output.getCalArray(iroc);

    const int padOffset = roc.isOROC() ? mapper.getPadsInIROC() : 0;

    // skip empty
    if (!(std::abs(rocPulserQtot.getSum()) > 0)) {
      continue;
    }

    // loop over pads
    for (size_t ipad = 0; ipad < rocPulserQtot.getData().size(); ++ipad) {
      const int globalPad = ipad + padOffset;
      const FECInfo& fecInfo = mapper.fecInfo(globalPad);
      const CRU cru = mapper.getCRU(roc.getSector(), globalPad);
      const uint32_t region = cru.region();
      const int cruID = cru.number();
      const int sampa = fecInfo.getSampaChip();
      const int sampaChannel = fecInfo.getSampaChannel();

      const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
      const int nFECs = partInfo.getNumberOfFECs();
      const int fecOffset = (nFECs + 1) / 2;
      const int fecInPartition = fecInfo.getIndex() - partInfo.getSectorFECOffset();
      const int dataWrapperID = fecInPartition >= fecOffset;
      const int globalLinkID = (fecInPartition % fecOffset) + dataWrapperID * 12;

      float pulserVal = rocPulserQtot.getValue(ipad);

      if ((pulserVal <= 0.5)) {
        LOGP(error, "Too small Pulser Qtot value in ROC {:2}, CRU {:3}, fec in CRU: {:2}, SAMPA: {}, channel: {:2}, pulserQtot norm: {:.4f}, setting value to 1", iroc, cruID, fecInPartition, sampa, sampaChannel, pulserVal);
        pulserVal = 1.f;
      }

      if (pulserVal > MaxVal) {
        LOGP(error, "Too large Pulser Qtot value in ROC {:2}, CRU {:3}, fec in CRU: {:2}, SAMPA: {}, channel: {:2}, pulserQtot norm: {:.4f}, setting value to max val: {}", iroc, cruID, fecInPartition, sampa, sampaChannel, pulserVal, MaxVal);
        pulserVal = MaxVal;
      }

      const int hwChannel = getHWChannel(sampa, sampaChannel, region % 2);
      // for debugging
      // printf("%4d %4d %4d %4d %4d: %u\n", cru.number(), globalLinkID, hwChannel, fecInfo.getSampaChip(), fecInfo.getSampaChannel(), getADCValue(pedestal));

      // default thresholds
      const auto pulserValFixed = floatToFixedSize<DataBits, FractionalBits>(pulserVal);
      commonModeKValues[LinkInfo(cruID, globalLinkID)][hwChannel] = pulserValFixed;
    }
  }

  const bool onlyFilled = false;
  const auto outFileTxt = (outputDir + "/commonMode_K_values.txt");
  const auto outFileRoot = (outputDir + "/commonMode_K_values.root");
  writeValues(outFileTxt, commonModeKValues, onlyFilled);

  getCalPad<FractionalBits>(outFileTxt, outFileRoot, "CMkValues");
}
