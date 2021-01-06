// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <string>
#include "FairLogger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/InputRecordWalker.h"
#include "CCDB/CcdbApi.h"
#include "PHOSBase/Mapping.h"
#include "PHOSBase/PHOSSimParams.h"
#include "PHOSReconstruction/Bunch.h"
#include "PHOSReconstruction/CaloRawFitter.h"
#include "PHOSReconstruction/AltroDecoder.h"
#include "PHOSReconstruction/RawDecodingError.h"
#include "PHOSWorkflow/RawToCellConverterSpec.h"

using namespace o2::phos::reco_workflow;

void RawToCellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "Initialize converter ";

  auto path = ctx.options().get<std::string>("mappingpath");
  if (!mMapping) {
    mMapping = std::unique_ptr<o2::phos::Mapping>(new o2::phos::Mapping(path));
    if (!mMapping) {
      LOG(ERROR) << "Failed to initialize mapping";
    }
    if (mMapping->setMapping() != o2::phos::Mapping::kOK) {
      LOG(ERROR) << "Failed to construct mapping";
    }
  }

  if (!mCalibParams) {
    if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mCalibParams = std::make_unique<CalibParams>(1); // test default calibration
      LOG(INFO) << "[RawToCellConverterSpec] No reading calibration from ccdb requested, set default";
    } else {
      LOG(INFO) << "[RawToCellConverterSpec] getting calibration object from ccdb";
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata;
      ccdb.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
      // auto tr = triggerbranch.begin();
      double eventTime = -1;
      // if(tr!=triggerbranch.end()){
      //   eventTime = (*tr).getBCData().getTimeNS() ;
      // }
      // mCalibParams = ccdb.retrieveFromTFileAny<o2::phos::CalibParams>("PHOS/Calib", metadata, eventTime);
      if (!mCalibParams) {
        LOG(FATAL) << "[RawToCellConverterSpec] can not get calibration object from ccdb";
      }
    }
  }

  auto fitmethod = ctx.options().get<std::string>("fitmethod");
  if (fitmethod == "default") {
    LOG(INFO) << "Using default raw fitter";
    mRawFitter = std::unique_ptr<o2::phos::CaloRawFitter>(new o2::phos::CaloRawFitter);
    //TODO: Configure parameters of fitter from options
    // mRawFitter->setAmpCut(mNoiseThreshold);
    // mRawFitter->setL1Phase(0.);
  }

  mFillChi2 = (ctx.options().get<std::string>("fillchi2").compare("on") == 0);
  if (mFillChi2) {
    LOG(INFO) << "Fit quality output will be filled";
  }

  mCombineGHLG = (ctx.options().get<std::string>("keepGHLG").compare("on") != 0);
  if (!mCombineGHLG) {
    LOG(INFO) << "Both HighGain and LowGain will be kept";
  }

  mPedestalRun = (ctx.options().get<std::string>("pedestal").find("on") != std::string::npos);
  if (mPedestalRun) {
    mRawFitter->setPedestal();
    LOG(INFO) << "Pedestal run will be processed";
  }
}

void RawToCellConverterSpec::run(framework::ProcessingContext& ctx)
{
  // Cache cells from bunch crossings as the component reads timeframes from many links consecutively
  std::map<o2::InteractionRecord, std::shared_ptr<std::vector<o2::phos::Cell>>> cellBuffer; // Internal cell buffer/
  int firstEntry = 0;
  mOutputHWErrors.clear();
  if (mFillChi2) {
    mOutputFitChi.clear();
  }

  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {

    o2::phos::RawReaderMemory rawreader(o2::framework::DataRefUtils::as<const char>(rawData));

    // loop over all the DMA pages
    while (rawreader.hasNext()) {
      try {
        rawreader.next();
      } catch (RawDecodingError::ErrorType_t e) {
        LOG(ERROR) << "Raw decoding error " << (int)e;
        //add error list
        mOutputHWErrors.emplace_back(14, (int)e, 1); //Put general errors to non-existing DDL14
        //if problem in header, abandon this page
        if (e == RawDecodingError::ErrorType_t::PAGE_NOTFOUND ||
            e == RawDecodingError::ErrorType_t::HEADER_DECODING ||
            e == RawDecodingError::ErrorType_t::HEADER_INVALID) {
          break;
        }
        //if problem in payload, try to continue
        continue;
      }
      auto& header = rawreader.getRawHeader();
      auto triggerBC = o2::raw::RDHUtils::getTriggerBC(header);
      auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(header);
      auto ddl = o2::raw::RDHUtils::getFEEID(header);

      o2::InteractionRecord currentIR(triggerBC, triggerOrbit);
      std::shared_ptr<std::vector<o2::phos::Cell>> currentCellContainer;
      auto found = cellBuffer.find(currentIR);
      if (found == cellBuffer.end()) {
        currentCellContainer = std::make_shared<std::vector<o2::phos::Cell>>();
        cellBuffer[currentIR] = currentCellContainer;
      } else {
        currentCellContainer = found->second;
      }
      if (ddl > o2::phos::Mapping::NDDL) { //only 14 correct DDLs
        LOG(ERROR) << "DDL=" << ddl;
        mOutputHWErrors.emplace_back(15, 16, char(ddl)); //Add non-existing DDL as DDL 15
        continue;                                        //skip STU ddl
      }
      // use the altro decoder to decode the raw data, and extract the RCU trailer
      o2::phos::AltroDecoder decoder(rawreader);
      AltroDecoderError::ErrorType_t err = decoder.decode();

      if (err != AltroDecoderError::kOK) {
        //TODO handle severe errors
        //TODO: probably careful conversion of decoder errors to Fitter errors?
        char e = (char)err;
        mOutputHWErrors.emplace_back(ddl, 16, e); //assign general header errors to non-existing FEE 16
      }
      auto& rcu = decoder.getRCUTrailer();
      auto& channellist = decoder.getChannels();
      // Loop over all the channels for this RCU
      for (auto& chan : channellist) {
        short absId;
        Mapping::CaloFlag caloFlag;
        short fee;
        char e2 = CheckHWAddress(ddl, chan.getHardwareAddress(), fee);
        if (e2) {
          mOutputHWErrors.emplace_back(ddl, fee, e2);
          continue;
        }
        Mapping::ErrorStatus s = mMapping->hwToAbsId(ddl, chan.getHardwareAddress(), absId, caloFlag);
        if (s != Mapping::ErrorStatus::kOK) {
          mOutputHWErrors.emplace_back(ddl, 15, (char)s); //use non-existing FEE 15 for general errors
          continue;
        }
        if (caloFlag != Mapping::kTRU) { //HighGain or LowGain
          CaloRawFitter::FitStatus fitResults = mRawFitter->evaluate(chan.getBunches());
          if (fitResults == CaloRawFitter::FitStatus::kNoTime) {
            mOutputHWErrors.emplace_back(ddl, fee, (char)5); //Time evaluation error occured
          }
          if (mFillChi2) {
            for (int is = 0; is < mRawFitter->getNsamples(); is++) {
              if (!mRawFitter->isOverflow(is)) { //Overflow is will show wrong chi2
                short chiAddr = absId;
                chiAddr |= caloFlag << 14;
                mOutputFitChi.emplace_back(chiAddr);
                mOutputFitChi.emplace_back(short(mRawFitter->getChi2(is)));
              }
            }
          }
          if (fitResults == CaloRawFitter::FitStatus::kOK || fitResults == CaloRawFitter::FitStatus::kNoTime) {
            //TODO: which results should be accepted? full configurable list
            for (int is = 0; is < mRawFitter->getNsamples(); is++) {
              if (caloFlag == Mapping::kHighGain && !mRawFitter->isOverflow(is)) {
                currentCellContainer->emplace_back(absId, mRawFitter->getAmp(is),
                                                   mRawFitter->getTime(is) * o2::phos::PHOSSimParams::Instance().mTimeTick, (ChannelType_t)caloFlag);
              }
              if (caloFlag == Mapping::kLowGain) {
                currentCellContainer->emplace_back(absId, mRawFitter->getAmp(is),
                                                   mRawFitter->getTime(is) * o2::phos::PHOSSimParams::Instance().mTimeTick, (ChannelType_t)caloFlag);
              }
            }
          }
        }
      }
    } //RawReader::hasNext
  }

  // Loop over BCs, sort cells with increasing cell ID and write to output containers
  mOutputCells.clear();
  mOutputTriggerRecords.clear();
  for (auto [bc, cells] : cellBuffer) {
    int prevCellSize = mOutputCells.size();
    if (cells->size()) {
      // Sort cells according to cell ID
      std::sort(cells->begin(), cells->end(), [](o2::phos::Cell& lhs, o2::phos::Cell& rhs) { return lhs.getAbsId() < rhs.getAbsId(); });

      if (mCombineGHLG && !mPedestalRun) { // combine for normal data, do not combine e.g. for LED run and pedestal
        //Combine HG and LG sells
        //Should be next to each other after sorting
        auto it1 = cells->begin();
        auto it2 = cells->begin();
        it2++;
        while (it1 != cells->end()) {
          if (it2 != cells->end()) {
            if ((*it1).getAbsId() == (*it2).getAbsId()) { //HG and LG channels, if both, copy only HG as more precise
              if ((*it1).getType() == o2::phos::HIGH_GAIN) {
                mOutputCells.push_back(*it1);
              } else {
                mOutputCells.push_back(*it2);
              }
              ++it1; //yes increase twice
              ++it2;
            } else { //no double cells, copy this one
              mOutputCells.push_back(*it1);
            }
          } else { //just copy last one
            mOutputCells.push_back(*it1);
          }
          ++it1;
          ++it2;
        }
      } else {
        for (auto cell : *cells) {
          mOutputCells.push_back(cell);
        }
      }
    }

    mOutputTriggerRecords.emplace_back(bc, mOutputCells.size(), mOutputCells.size() - prevCellSize);
  }
  cellBuffer.clear();

  LOG(INFO) << "[PHOSRawToCellConverter - run] Writing " << mOutputCells.size() << " cells ...";
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLS", 0, o2::framework::Lifetime::Timeframe}, mOutputCells);
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggerRecords);
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe}, mOutputHWErrors);
  if (mFillChi2) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLFITQA", 0, o2::framework::Lifetime::Timeframe}, mOutputFitChi);
  }
}

char RawToCellConverterSpec::CheckHWAddress(short ddl, short hwAddr, short& fee)
{

  if (ddl < 0 || ddl > o2::phos::Mapping::NDDL) {
    return (char)4;
  }
  //  short chan = hwAddr & 0xf;
  short chip = hwAddr >> 4 & 0x7;
  short fec = hwAddr >> 7 & 0xf;
  short branch = hwAddr >> 11 & 0x1;

  if (branch < 0 || branch > 1) {
    return (char)1;
  }
  if (fec < 0 || fec > 15) {
    return (char)2;
  }
  if (fec != 0 && (chip < 0 || chip > 4 || chip == 1)) { //Do not check for TRU (fec=0)
    return (char)3;
  }
  return (char)0;
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getRawToCellConverterSpec()
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("RAWDATA", o2::framework::ConcreteDataTypeMatcher{"PHS", "RAWDATA"}, o2::framework::Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("PHS", "CELLS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "CELLFITQA", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"PHOSRawToCellConverterSpec",
                                          inputs, // o2::framework::select("A:PHS/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::RawToCellConverterSpec>(),
                                          o2::framework::Options{
                                            {"fitmethod", o2::framework::VariantType::String, "default", {"Fit method (default or fast)"}},
                                            {"mappingpath", o2::framework::VariantType::String, "", {"Path to mapping files"}},
                                            {"fillchi2", o2::framework::VariantType::String, "off", {"Fill sample qualities on/off"}},
                                            {"keepGHLG", o2::framework::VariantType::String, "off", {"keep HighGain and Low Gain signals on/off"}},
                                            {"pedestal", o2::framework::VariantType::String, "off", {"Analyze as pedestal run on/off"}}}};
}
