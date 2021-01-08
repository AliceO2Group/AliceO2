// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FairLogger.h"

#include "Framework/RootSerializationSupport.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "PHOSWorkflow/CellConverterSpec.h"
#include "Framework/ControlService.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "PHOSBase/PHOSSimParams.h"
#include "CCDB/CcdbApi.h"

using namespace o2::phos::reco_workflow;

void CellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(INFO) << "[PHOSCellConverter - init] Initialize converter " << (mPropagateMC ? "with" : "without") << " MC truth container";
}

void CellConverterSpec::run(framework::ProcessingContext& ctx)
{
  //  LOG(DEBUG) << "[PHOSCellConverter - run] called";
  LOG(INFO) << "[PHOSCellConverter - run] called";
  auto dataref = ctx.inputs().get("digits");
  auto const* phosheader = o2::framework::DataRefUtils::getHeader<o2::phos::PHOSBlockHeader*>(dataref);
  if (!phosheader->mHasPayload) {
    LOG(INFO) << "[PHOSCellConverter - run] No more digits" << std::endl;
    ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
    return;
  }

  mOutputCells.clear();
  mOutputCellTrigRecs.clear();

  auto digits = ctx.inputs().get<std::vector<o2::phos::Digit>>("digits");
  auto digitsTR = ctx.inputs().get<std::vector<o2::phos::TriggerRecord>>("digitTriggerRecords");
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>> truthcont(nullptr);
  if (mPropagateMC) {
    truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>*>("digitsmctr");
    mOutputTruthCont.clear();
    mOutputTruthMap.clear();
  }
  LOG(INFO) << "[PHOSCellConverter - run]  Received " << digits.size() << " digits and " << digitsTR.size() << " TriggerRecords" << truthcont->getNElements() << " MC labels";

  //Get TimeStamp from TriggerRecord
  if (!mBadMap) {
    if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mBadMap = new BadChannelMap(1); // test default map
      LOG(INFO) << "[PHOSCellConverter - run] No reading BadMap from ccdb requested, set default";
    } else {
      LOG(INFO) << "[PHOSCellConverter - run] getting BadMap object from ccdb";
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata; // do we want to store any meta data?
      ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
      long bcTime = -1;                            //TODO!!! Convert BC time to time o2::InteractionRecord bcTime = digitsTR.front().getBCData() ;
      mBadMap = ccdb.retrieveFromTFileAny<o2::phos::BadChannelMap>("PHOS/BadMap", metadata, bcTime);
      if (!mBadMap) {
        LOG(FATAL) << "[PHOSCellConverter - run] can not get Bad Map";
      }
    }
  }
  //TODO!!! Should we check if BadMap should be updated/validity range still valid???
  mOutputCells.reserve(digits.size()); // most of digits will be copied
  int labelIndex = 0;
  for (const auto& tr : digitsTR) {
    int iFirstDigit = tr.getFirstEntry();
    int iLastDigit = iFirstDigit + tr.getNumberOfObjects();
    int indexStart = mOutputCells.size();
    int icell = 0;
    for (int i = iFirstDigit; i < iLastDigit; i++) {
      const auto& dig = digits.at(i);

      //apply filter
      if (!mBadMap->isChannelGood(dig.getAbsId())) {
        continue;
      }

      ChannelType_t chantype;
      if (dig.isHighGain()) {
        chantype = ChannelType_t::HIGH_GAIN;
      } else {
        chantype = ChannelType_t::LOW_GAIN;
      }

      //    TODO!!! TRU copying...
      //    if (dig.getTRU())
      //      chantype = ChannelType_t::TRU;

      mOutputCells.emplace_back(dig.getAbsId(), dig.getAmplitude(), dig.getTime(), chantype);
      if (mPropagateMC) { //copy MC info,
        int iLab = dig.getLabel();
        if (iLab > -1) {
          mOutputTruthCont.addElements(labelIndex, truthcont->getLabels(iLab));
          mOutputTruthMap.emplace_back(icell); //Relate cell index and label index
          labelIndex++;
        }
      }
      icell++;
    }
    mOutputCellTrigRecs.emplace_back(tr.getBCData(), indexStart, mOutputCells.size());
  }
  LOG(INFO) << "[PHOSCellConverter - run] Writing " << mOutputCells.size() << " cells, " << mOutputCellTrigRecs.size() << " Trig Records " << mOutputTruthCont.getNElements() << " PHOS labels ";
  ;
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLS", 0, o2::framework::Lifetime::Timeframe}, mOutputCells);
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputCellTrigRecs);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLSMCMAP", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthMap);
  }
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getCellConverterSpec(bool propagateMC)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginPHS, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("digitTriggerRecords", o2::header::gDataOriginPHS, "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "CELLS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("digitsmctr", "PHS", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("PHS", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("PHS", "CELLSMCMAP", 0, o2::framework::Lifetime::Timeframe);
  }
  return o2::framework::DataProcessorSpec{"PHOSCellConverterSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::CellConverterSpec>(propagateMC)};
}
