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
#include <vector>
#include <numeric>
#include <fmt/format.h>
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

void prepareCMFiles(const std::string_view itDataFile, std::string outputDir = "./")
{
  const auto& mapper = Mapper::instance();

  // ===| load noise and pedestal from file |===
  CalDet<float>* padFraction{nullptr};
  CalDet<float>* padExpLambda{nullptr};
  // const CDBInterface::CalPadMapType* feePad = nullptr;

  if (itDataFile.find("cdb") != std::string::npos) {
    auto& cdb = CDBInterface::instance();
    if (itDataFile.find("cdb-test") == 0) {
      cdb.setURL("http://ccdb-test.cern.ch:8080");
    } else if (itDataFile.find("cdb-prod") == 0) {
      cdb.setURL("https://alice-ccdb.cern.ch");
    }
    const auto timePos = itDataFile.find("@");
    if (timePos != std::string_view::npos) {
      std::cout << "set time stamp " << std::stol(itDataFile.substr(timePos + 1).data()) << "\n";
      cdb.setTimeStamp(std::stol(itDataFile.substr(timePos + 1).data()));
    }
    fmt::print("cdb reading not yet implemented!\n");
  } else {
    auto calPads = utils::readCalPads(itDataFile, "fraction,expLambda");
    padFraction = calPads[0];
    padExpLambda = calPads[1];
  }

  DataMapF mapFraction;
  DataMapF mapExpLambda;

  // ===| prepare values |===
  for (size_t iroc = 0; iroc < padFraction->getData().size(); ++iroc) {
    const ROC roc(iroc);

    const auto& rocFraction = padFraction->getCalArray(iroc);
    const auto& rocExpLambda = padExpLambda->getCalArray(iroc);

    const int padOffset = roc.isOROC() ? mapper.getPadsInIROC() : 0;
    const auto& traceLengths = roc.isIROC() ? mapper.getTraceLengthsIROC() : mapper.getTraceLengthsOROC();

    const float meanFraction = rocFraction.getMean();
    const float meanExpLambda = rocExpLambda.getMean();

    // skip empty
    if (!(std::abs(rocFraction.getSum()) > 0)) {
      continue;
    }

    // loop over pads
    for (size_t ipad = 0; ipad < rocFraction.getData().size(); ++ipad) {
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

      const auto traceLength = traceLengths[ipad];

      float fractionVal = rocFraction.getValue(ipad);
      float expLambdaVal = rocExpLambda.getValue(ipad);

      if ((fractionVal <= 0) || (fractionVal > 0.6)) {
        LOGP(error, "Too fraction value in ROC {:2}, CRU {:3}, fec in CRU: {:2}, SAMPA: {}, channel: {:2}: {:.4f}, setting value to roc mean {}", iroc, cruID, fecInPartition, sampa, sampaChannel, fractionVal, meanFraction);
        fractionVal = meanFraction;
      }

      if ((expLambdaVal < 0.5) || (expLambdaVal > 1)) {
        LOGP(error, "Too expLambda value in ROC {:2}, CRU {:3}, fec in CRU: {:2}, SAMPA: {}, channel: {:2}: {:.4f}, setting value to roc mean {}", iroc, cruID, fecInPartition, sampa, sampaChannel, expLambdaVal, meanExpLambda);
        expLambdaVal = meanExpLambda;
      }

      const int hwChannel = getHWChannel(sampa, sampaChannel, region % 2);
      // for debugging
      // printf("%4d %4d %4d %4d %4d: %u\n", cru.number(), globalLinkID, hwChannel, fecInfo.getSampaChip(), fecInfo.getSampaChannel(), getADCValue(pedestal));

      mapFraction[LinkInfo(cruID, globalLinkID)][hwChannel] = fractionVal;
      mapExpLambda[LinkInfo(cruID, globalLinkID)][hwChannel] = expLambdaVal;
    }
  }

  const bool onlyFilled = false;

  const auto outFileK1Txt = (outputDir + "/it_K1_values.txt");
  writeValues(outFileK1Txt, mapFraction, onlyFilled);

  const auto outFileK2Txt = (outputDir + "/it_K2_values.txt");
  writeValues(outFileK2Txt, mapExpLambda, onlyFilled);

  delete padFraction;
  delete padExpLambda;
}
