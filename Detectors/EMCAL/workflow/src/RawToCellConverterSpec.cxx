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
#include <string>
#include <iomanip>
#include <iostream>
#include <bitset>

#include <InfoLogger/InfoLogger.hxx>

#include "CommonConstants/Triggers.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "DataFormatsEMCAL/Constants.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "DataFormatsEMCAL/ErrorTypeFEE.h"
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALCalib/FeeDCS.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/CaloRawFitterGamma2.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/RawDecodingError.h"
#include "EMCALReconstruction/RecoParam.h"
#include "EMCALWorkflow/RawToCellConverterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonUtils/VerbosityConfig.h"

using namespace o2::emcal::reco_workflow;

RawToCellConverterSpec::~RawToCellConverterSpec()
{
  if (mErrorMessagesSuppressed) {
    LOG(warning) << "Suppressed further " << mErrorMessagesSuppressed << " error messages";
  }
}

void RawToCellConverterSpec::init(framework::InitContext& ctx)
{
  if (ctx.services().active<AliceO2::InfoLogger::InfoLoggerContext>()) {
    auto& ilctx = ctx.services().get<AliceO2::InfoLogger::InfoLoggerContext>();
    ilctx.setField(AliceO2::InfoLogger::InfoLoggerContext::FieldName::Detector, "EMC");
  }

  LOG(debug) << "[EMCALRawToCellConverter - init] Initialize converter ";
  if (!mGeometry) {
    mGeometry = Geometry::GetInstanceFromRunNumber(223409);
  }
  if (!mGeometry) {
    LOG(error) << "Failure accessing geometry";
  }

  if (!mMapper) {
    mMapper = std::unique_ptr<MappingHandler>(new o2::emcal::MappingHandler);
  }
  if (!mMapper) {
    LOG(error) << "Failed to initialize mapper";
  }

  auto fitmethod = ctx.options().get<std::string>("fitmethod");
  if (fitmethod == "standard") {
    LOG(info) << "Using standard raw fitter";
    mRawFitter = std::unique_ptr<CaloRawFitter>(new o2::emcal::CaloRawFitterStandard);
  } else if (fitmethod == "gamma2") {
    LOG(info) << "Using gamma2 raw fitter";
    mRawFitter = std::unique_ptr<CaloRawFitter>(new o2::emcal::CaloRawFitterGamma2);
  } else {
    LOG(fatal) << "Unknown fit method" << fitmethod;
  }
  LOG(info) << "Creating decoding errors: " << (mCreateRawDataErrors ? "yes" : "no");

  mPrintTrailer = ctx.options().get<bool>("printtrailer");

  mMaxErrorMessages = ctx.options().get<int>("maxmessage");
  LOG(info) << "Suppressing error messages after " << mMaxErrorMessages << " messages";

  mMergeLGHG = !ctx.options().get<bool>("no-mergeHGLG");
  mDisablePedestalEvaluation = ctx.options().get<bool>("no-evalpedestal");
  mActiveLinkCheck = !ctx.options().get<bool>("no-checkactivelinks");

  LOG(info) << "Running gain merging mode: " << (mMergeLGHG ? "yes" : "no");
  LOG(info) << "Checking for active links: " << (mActiveLinkCheck ? "yes" : "no");
  LOG(info) << "Calculate pedestals:       " << (mDisablePedestalEvaluation ? "no" : "yes");
  LOG(info) << "Using L0LM delay: " << o2::ctp::TriggerOffsetsParam::Instance().LM_L0 << " BCs";

  mRawFitter->setAmpCut(mNoiseThreshold);
  mRawFitter->setL1Phase(0.);
}

void RawToCellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(debug) << "[EMCALRawToCellConverter - run] called";
  mCalibHandler->checkUpdates(ctx);
  updateCalibrationObjects();

  // container with BCid and feeID
  std::unordered_map<int64_t, std::bitset<46>> bcFreq;

  double timeshift = RecoParam::Instance().getCellTimeShiftNanoSec();       // subtract offset in ns in order to center the time peak around the nominal delay
  auto maxBunchLengthRP = RecoParam::Instance().getMaxAllowedBunchLength(); // exclude bunches where either the start time or the bunch length is above the expected maximum
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  constexpr auto descRaw = o2::header::gDataDescriptionRawData;

  // reset message counter after 10 minutes
  auto currenttime = std::chrono::system_clock::now();
  std::chrono::duration<double> interval = mReferenceTime - currenttime;
  if (interval.count() > 600) {
    // Resetting error messages after 10 minutes
    mNumErrorMessages = 0;
    mReferenceTime = currenttime;
  }

  mOutputCells.clear();
  mOutputTriggerRecords.clear();
  mOutputDecoderErrors.clear();

  if (isLostTimeframe(ctx)) {
    sendData(ctx, mOutputCells, mOutputTriggerRecords, mOutputDecoderErrors);
    return;
  }

  mCellHandler.reset();

  // Get the first orbit of the timeframe later used to check whether the corrected
  // BC is within the timeframe
  const auto tfOrbitFirst = ctx.services().get<o2::framework::TimingInfo>().firstTForbit;
  auto lml0delay = o2::ctp::TriggerOffsetsParam::Instance().LM_L0;

  std::vector<framework::InputSpec> filter{{"filter", framework::ConcreteDataTypeMatcher(originEMC, descRaw)}};
  int firstEntry = 0;
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs(), filter)) {
    // Skip SOX headers
    auto rdhblock = reinterpret_cast<const o2::header::RDHAny*>(rawData.payload);
    if (o2::raw::RDHUtils::getHeaderSize(rdhblock) == static_cast<int>(o2::framework::DataRefUtils::getPayloadSize(rawData))) {
      continue;
    }

    o2::emcal::RawReaderMemory rawreader(framework::DataRefUtils::as<const char>(rawData));
    rawreader.setRangeSRUDDLs(0, 39);

    // loop over all the DMA pages
    while (rawreader.hasNext()) {
      try {
        rawreader.next();
      } catch (RawDecodingError& e) {
        handlePageError(e);
        if (e.getErrorType() == RawDecodingError::ErrorType_t::HEADER_DECODING || e.getErrorType() == RawDecodingError::ErrorType_t::HEADER_INVALID) {
          // We must break in case of header decoding as the offset to the next payload is lost
          // consequently the parser does not know where to continue leading to an infinity loop
          break;
        }
        // We must skip the page as payload is not consistent
        // otherwise the next functions will rethrow the exceptions as
        // the page format does not follow the expected format
        continue;
      }
      for (auto& e : rawreader.getMinorErrors()) {
        handleMinorPageError(e);
        // For minor errors we do not need to skip the page, just print and send the error to the QC
      }

      auto& header = rawreader.getRawHeader();
      auto triggerBC = raw::RDHUtils::getTriggerBC(header);
      auto triggerOrbit = raw::RDHUtils::getTriggerOrbit(header);
      auto feeID = raw::RDHUtils::getFEEID(header);
      auto triggerbits = raw::RDHUtils::getTriggerType(header);

      int correctionShiftBCmod4 = 0;
      o2::InteractionRecord currentIR(triggerBC, triggerOrbit);
      // Correct physics triggers for the shift of the BC due to the LM-L0 delay
      if (triggerbits & o2::trigger::PhT) {
        if (currentIR.differenceInBC({0, tfOrbitFirst}) >= lml0delay) {
          currentIR -= lml0delay; // guaranteed to stay in the TF containing the collision
          // in case we correct for the L0LM delay we need to adjust the BC mod 4, because if the L0LM delay % 4 != 0 it will change the permutation of trigger peaks
          // we need to add back the correction we applied % 4 to the corrected BC during the correction of the cell time in order to keep the same permutation
          correctionShiftBCmod4 = lml0delay % 4;
        } else {
          // discard the data associated with this IR as it was triggered before the start of timeframe
          continue;
        }
      }

      bcFreq[currentIR.toLong()].set(feeID, true);

      // Correct the cell time for the bc mod 4 (LHC: 40 MHz clock - ALTRO: 10 MHz clock)
      // Convention: All times shifted with respect to BC % 4 = 0 for trigger BC
      // Attention: Correction only works for the permutation (0 1 2 3) of the BC % 4, if the permutation is
      // different the BC for the correction has to be shifted by n BCs to obtain permutation (0 1 2 3)
      // We apply here the following shifts:
      // - correction for the L0-LM delay mod 4 in order to restore the original ordering of the BCs mod 4
      // - phase shift in order to adjust for permutations different from (0 1 2 3)
      int bcmod4 = (currentIR.bc + correctionShiftBCmod4 + RecoParam::Instance().getPhaseBCmod4()) % 4;
      LOG(debug) << "Original BC " << triggerBC << ", L0LM corrected " << currentIR.bc;
      LOG(debug) << "Applying correction for LM delay: " << correctionShiftBCmod4;
      LOG(debug) << "BC mod original: " << triggerBC % 4 << ", corrected " << bcmod4;
      LOG(debug) << "Applying time correction: " << -1 * 25 * bcmod4;
      auto& currentEvent = mCellHandler.getEventContainer(currentIR);
      if (!currentEvent.getTriggerBits()) {
        currentEvent.setTriggerBits(triggerbits);
      }

      if (feeID >= 40) {
        continue; // skip STU ddl
      }

      // std::cout<<rawreader.getRawHeader()<<std::endl;

      // use the altro decoder to decode the raw data, and extract the RCU trailer
      AltroDecoder decoder(rawreader);
      if (maxBunchLengthRP) {
        // apply user-defined max. bunch length
        decoder.setMaxBunchLength(maxBunchLengthRP);
      }
      // check the words of the payload exception in altrodecoder
      try {
        decoder.decode();
      } catch (AltroDecoderError& e) {
        handleAltroError(e, feeID);
        continue;
      }
      for (const auto& minorerror : decoder.getMinorDecodingErrors()) {
        handleMinorAltroError(minorerror, feeID);
      }

      if (mPrintTrailer) {
        // Can become very verbose, therefore must be switched on explicitly in addition
        // to high debug level
        LOG(debug4) << decoder.getRCUTrailer();
      }
      // Apply zero suppression only in case it was enabled
      if (decoder.getRCUTrailer().hasZeroSuppression()) {
        LOG(debug3) << "Zero suppression enabled";
      } else {
        LOG(debug3) << "Zero suppression disabled";
      }
      if (mDisablePedestalEvaluation) {
        // auto-disable pedestal evaluation in the raw fitter
        // treat all channels as zero-suppressed independent of
        // what is provided from the RCU trailer
        mRawFitter->setIsZeroSuppressed(true);
      } else {
        mRawFitter->setIsZeroSuppressed(decoder.getRCUTrailer().hasZeroSuppression());
      }

      try {

        const auto& map = mMapper->getMappingForDDL(feeID);
        uint16_t iSM = feeID / 2;

        // Loop over all the channels
        int nBunchesNotOK = 0;
        for (auto& chan : decoder.getChannels()) {
          int iRow, iCol;
          ChannelType_t chantype;
          try {
            iRow = map.getRow(chan.getHardwareAddress());
            iCol = map.getColumn(chan.getHardwareAddress());
            chantype = map.getChannelType(chan.getHardwareAddress());
          } catch (Mapper::AddressNotFoundException& ex) {
            handleAddressError(ex, feeID, chan.getHardwareAddress());
            continue;
          }

          if (!(chantype == o2::emcal::ChannelType_t::HIGH_GAIN || chantype == o2::emcal::ChannelType_t::LOW_GAIN || chantype == o2::emcal::ChannelType_t::LEDMON)) {
            continue;
          }

          // Drop LEDMON reconstruction in case of physics triggers
          if (chantype == o2::emcal::ChannelType_t::LEDMON && !(triggerbits & o2::trigger::Cal)) {
            continue;
          }

          int CellID = -1;
          bool isLowGain = false;
          try {
            if (chantype == o2::emcal::ChannelType_t::HIGH_GAIN || chantype == o2::emcal::ChannelType_t::LOW_GAIN) {
              // high- / low-gain cell
              CellID = getCellAbsID(iSM, iCol, iRow);
              isLowGain = chantype == o2::emcal::ChannelType_t::LOW_GAIN;
            } else {
              CellID = geLEDMONAbsID(iSM, iCol); // Module index encoded in colum for LEDMONs
              isLowGain = iRow == 0;             // For LEDMONs gain type is encoded in the row (0 - low gain, 1 - high gain)
            }
          } catch (ModuleIndexException& e) {
            handleGeometryError(e, iSM, CellID, chan.getHardwareAddress(), chantype);
            continue;
          }

          // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
          CaloFitResults fitResults;
          try {
            fitResults = mRawFitter->evaluate(chan.getBunches());
            // Prevent negative entries - we should no longer get here as the raw fit usually will end in an error state
            if (fitResults.getAmp() < 0) {
              fitResults.setAmp(0.);
            }
            if (fitResults.getTime() < 0) {
              fitResults.setTime(0.);
            }
            // apply correction for bc mod 4
            double celltime = fitResults.getTime() - timeshift - 25 * bcmod4;
            double amp = fitResults.getAmp() * o2::emcal::constants::EMCAL_ADCENERGY;
            if (isLowGain) {
              amp *= o2::emcal::constants::EMCAL_HGLGFACTOR;
            }
            if (chantype == o2::emcal::ChannelType_t::LEDMON) {
              // Mark LEDMONs as HIGH_GAIN/LOW_GAIN for gain type merging - will be flagged as LEDMON later when pushing to the output container
              currentEvent.setLEDMONCell(CellID, amp, celltime, isLowGain ? o2::emcal::ChannelType_t::LOW_GAIN : o2::emcal::ChannelType_t::HIGH_GAIN, chan.getHardwareAddress(), feeID, mMergeLGHG);
            } else {
              currentEvent.setCell(CellID, amp, celltime, chantype, chan.getHardwareAddress(), feeID, mMergeLGHG);
            }
          } catch (CaloRawFitter::RawFitterError_t& fiterror) {
            handleFitError(fiterror, feeID, CellID, chan.getHardwareAddress());
          }
        }
      } catch (o2::emcal::MappingHandler::DDLInvalid& ddlerror) {
        // Unable to catch mapping
        handleDDLError(ddlerror, feeID);
      }
    }
  }

  std::bitset<46> bitSetActiveLinks;
  if (mActiveLinkCheck) {
    // build expected active mask from DCS
    FeeDCS* feedcs = mCalibHandler->getFEEDCS();
    auto list0 = feedcs->getDDLlist0();
    auto list1 = feedcs->getDDLlist1();
    //  links 21 and 39 do not exist, but they are set active in DCS
    list0.set(21, false);
    list1.set(7, false);
    // must be 0x307FFFDFFFFF if all links are active
    bitSetActiveLinks = std::bitset<46>((list1.to_ullong() << 32) + list0.to_ullong());
  }

  // Loop over BCs, sort cells with increasing tower ID and write to output containers
  RecoContainerReader eventIterator(mCellHandler);
  while (eventIterator.hasNext()) {
    int ncellsEvent = 0, nLEDMONsEvent = 0;
    int eventstart = mOutputCells.size();
    auto& currentevent = eventIterator.nextEvent();
    const auto interaction = currentevent.getInteractionRecord();
    if (mActiveLinkCheck) {
      // check for current event if all links are present
      // discard event if not all links are present
      auto bcfreqFound = bcFreq.find(interaction.toLong());
      if (bcfreqFound != bcFreq.end()) {
        const auto& activelinks = bcfreqFound->second;
        if (activelinks != bitSetActiveLinks) {
          static int nErrors = 0;
          if (nErrors++ < 3) {
            LOG(error) << "Not all EMC active links contributed in global BCid=" << interaction.toLong() << ": mask=" << (activelinks ^ bitSetActiveLinks) << (nErrors == 3 ? " (not reporting further errors to avoid spamming)" : "");
          }
          if (mCreateRawDataErrors) {
            for (std::size_t ilink = 0; ilink < bitSetActiveLinks.size(); ilink++) {
              if (!bitSetActiveLinks.test(ilink)) {
                continue;
              }
              if (!activelinks.test(ilink)) {
                mOutputDecoderErrors.emplace_back(ilink, ErrorTypeFEE::ErrorSource_t::LINK_ERROR, 0, -1, -1);
              }
            }
          }
          // discard event
          // create empty trigger record with dedicated trigger bit marking as rejected
          mOutputTriggerRecords.emplace_back(interaction, currentevent.getTriggerBits() | o2::emcal::triggerbits::Inc, eventstart, 0);
          continue;
        }
      }
    }
    // Add cells
    if (currentevent.getNumberOfCells()) {
      LOG(debug) << "Event has " << currentevent.getNumberOfCells() << " cells";
      currentevent.sortCells(false);
      ncellsEvent = bookEventCells(currentevent.getCells(), false);
    }
    // Add LEDMONs (if present)
    if (currentevent.getNumberOfLEDMONs()) {
      LOG(debug) << "Event has " << currentevent.getNumberOfLEDMONs() << " LEDMONs";
      currentevent.sortCells(true);
      nLEDMONsEvent = bookEventCells(currentevent.getLEDMons(), true);
    }
    LOG(debug) << "Next event [Orbit " << interaction.orbit << ", BC (" << interaction.bc << "]: Accepted " << ncellsEvent << " cells and " << nLEDMONsEvent << " LEDMONS";
    mOutputTriggerRecords.emplace_back(interaction, currentevent.getTriggerBits(), eventstart, ncellsEvent + nLEDMONsEvent);
  }

  LOG(info) << "[EMCALRawToCellConverter - run] Writing " << mOutputCells.size() << " cells from " << mOutputTriggerRecords.size() << " events ...";
  sendData(ctx, mOutputCells, mOutputTriggerRecords, mOutputDecoderErrors);
}

void RawToCellConverterSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mCalibHandler->finalizeCCDB(matcher, obj)) {
    return;
  }
}

void RawToCellConverterSpec::updateCalibrationObjects()
{
  if (mCalibHandler->hasUpdateRecoParam()) {
    LOG(info) << "RecoParams updated";
    o2::emcal::RecoParam::Instance().printKeyValues(true, true);
  }
  if (mCalibHandler->hasUpdateFEEDCS()) {
    LOG(info) << "DCS params updated";
  }
}

bool RawToCellConverterSpec::isLostTimeframe(framework::ProcessingContext& ctx) const
{
  constexpr auto originEMC = header::gDataOriginEMC;
  o2::framework::InputSpec dummy{"dummy",
                                 framework::ConcreteDataMatcher{originEMC,
                                                                header::gDataDescriptionRawData,
                                                                0xDEADBEEF}};
  static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
  for (const auto& ref : o2::framework::InputRecordWalker(ctx.inputs(), {dummy})) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
    if (payloadSize == 0) {
      auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
      if (++contDeadBeef <= maxWarn) {
        LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
             contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
      }
      return true;
    }
  }
  contDeadBeef = 0; // if good data, reset the counter
  return false;
}

int RawToCellConverterSpec::bookEventCells(const gsl::span<const o2::emcal::RecCellInfo>& cells, bool isLELDMON)
{
  double noiseThresholLGnoHG = RecoParam::Instance().getNoiseThresholdLGnoHG();
  int ncellsSelected = 0;
  for (const auto& cell : cells) {
    if (cell.mIsLGnoHG) {
      // Treat error only in case the LG is above the noise threshold
      // no HG cell found, we can assume the cell amplitude is the LG amplitude
      int ampLG = cell.mCellData.getAmplitude() / (o2::emcal::constants::EMCAL_ADCENERGY * o2::emcal::constants::EMCAL_HGLGFACTOR);
      // use cut at 3 sigma where sigma for the LG digitizer is 0.4 ADC counts (EMCAL-502)
      if (ampLG > noiseThresholLGnoHG) {
        handleGainError(reconstructionerrors::GainError_t::LGNOHG, cell.mDDLID, cell.mHWAddressLG);
      }
      continue;
    }
    if (cell.mHGOutOfRange) {
      handleGainError(reconstructionerrors::GainError_t::HGNOLG, cell.mDDLID, cell.mHWAddressHG);
      continue;
    }
    ncellsSelected++;
    mOutputCells.push_back(cell.mCellData);
    if (isLELDMON) {
      // LEDMON was handled internally in the reco container as HG/LG cell for gain type merging - reflag as LEDMON
      mOutputCells.back().setLEDMon();
    }
  }
  return ncellsSelected;
}

int RawToCellConverterSpec::getCellAbsID(int supermoduleID, int column, int row)
{
  auto [phishift, etashift] = mGeometry->ShiftOnlineToOfflineCellIndexes(supermoduleID, row, column);
  int cellID = mGeometry->GetAbsCellIdFromCellIndexes(supermoduleID, phishift, etashift);
  if (cellID > 17664 || cellID < 0) {
    throw ModuleIndexException(cellID, column, row, etashift, phishift);
  }
  return cellID;
}

int RawToCellConverterSpec::geLEDMONAbsID(int supermoduleID, int moduleID)
{
  if (moduleID >= o2::emcal::EMCAL_LEDREFS || moduleID < 0) {
    throw ModuleIndexException(moduleID);
  }
  return supermoduleID * o2::emcal::EMCAL_LEDREFS + moduleID;
}

void RawToCellConverterSpec::handleAddressError(const Mapper::AddressNotFoundException& error, int feeID, int hwaddress)
{
  if (mNumErrorMessages < mMaxErrorMessages) {
    LOG(warning) << "Mapping error DDL " << feeID << ": " << error.what();
    mNumErrorMessages++;
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  } else {
    mErrorMessagesSuppressed++;
  }
  if (mCreateRawDataErrors) {
    ErrorTypeFEE mappingError{feeID, ErrorTypeFEE::ErrorSource_t::ALTRO_ERROR, AltroDecoderError::errorTypeToInt(AltroDecoderError::ErrorType_t::ALTRO_MAPPING_ERROR), -1, hwaddress};
    mOutputDecoderErrors.push_back(mappingError);
  }
}

void RawToCellConverterSpec::handleAltroError(const o2::emcal::AltroDecoderError& altroerror, int ddlID)
{
  if (mNumErrorMessages < mMaxErrorMessages) {
    std::string errormessage;
    using AltroErrType = AltroDecoderError::ErrorType_t;
    switch (altroerror.getErrorType()) {
      case AltroErrType::RCU_TRAILER_ERROR:
        errormessage = " RCU Trailer Error ";
        break;
      case AltroErrType::RCU_VERSION_ERROR:
        errormessage = " RCU Version Error ";
        break;
      case AltroErrType::RCU_TRAILER_SIZE_ERROR:
        errormessage = " RCU Trailer Size Error ";
        break;
      case AltroErrType::ALTRO_BUNCH_HEADER_ERROR:
        errormessage = " ALTRO Bunch Header Error ";
        break;
      case AltroErrType::ALTRO_BUNCH_LENGTH_ERROR:
        errormessage = " ALTRO Bunch Length Error ";
        break;
      case AltroErrType::ALTRO_PAYLOAD_ERROR:
        errormessage = " ALTRO Payload Error ";
        break;
      case AltroErrType::ALTRO_MAPPING_ERROR:
        errormessage = " ALTRO Mapping Error ";
        break;
      case AltroErrType::CHANNEL_ERROR:
        errormessage = " Channel Error ";
        break;
      default:
        break;
    };
    LOG(warning) << " EMCAL raw task: " << errormessage << " in DDL " << ddlID;
    mNumErrorMessages++;
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  } else {
    mErrorMessagesSuppressed++;
  }
  if (mCreateRawDataErrors) {
    // fill histograms  with error types
    ErrorTypeFEE errornum(ddlID, ErrorTypeFEE::ErrorSource_t::ALTRO_ERROR, AltroDecoderError::errorTypeToInt(altroerror.getErrorType()), -1, -1);
    mOutputDecoderErrors.push_back(errornum);
  }
}

void RawToCellConverterSpec::handleMinorAltroError(const o2::emcal::MinorAltroDecodingError& minorerror, int ddlID)
{
  if (mNumErrorMessages < mMaxErrorMessages) {
    LOG(warning) << " EMCAL raw task - Minor error in DDL " << ddlID << ": " << minorerror.what();
    mNumErrorMessages++;
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  } else {
    mErrorMessagesSuppressed++;
  }
  if (mCreateRawDataErrors) {
    int fecID = -1, hwaddress = -1;
    try {
      hwaddress = Channel::getHardwareAddressFromChannelHeader(minorerror.getChannelHeader());
      fecID = mMapper->getFEEForChannelInDDL(ddlID, Channel::getFecIndexFromHwAddress(hwaddress), Channel::getBranchIndexFromHwAddress(hwaddress));
    } catch (Mapper::AddressNotFoundException& e) {
      // Unfortunately corrupted FEC IDs will not have useful information, so we need to initalize with -1
    } catch (MappingHandler::DDLInvalid& e) {
      // Unfortunately corrupted FEC IDs will not have useful information, so we need to initalize with -1
    }
    ErrorTypeFEE errornum(ddlID, ErrorTypeFEE::ErrorSource_t::MINOR_ALTRO_ERROR, MinorAltroDecodingError::errorTypeToInt(minorerror.getErrorType()), fecID, hwaddress);
    mOutputDecoderErrors.push_back(errornum);
  }
}

void RawToCellConverterSpec::handleDDLError(const MappingHandler::DDLInvalid& error, int feeID)
{
  if (mNumErrorMessages < mMaxErrorMessages) {
    LOG(error) << "Failed obtaining mapping for DDL " << error.getDDDL();
    mNumErrorMessages++;
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(error) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  }
  if (mCreateRawDataErrors) {
    mOutputDecoderErrors.emplace_back(feeID, ErrorTypeFEE::ErrorSource_t::ALTRO_ERROR, AltroDecoderError::errorTypeToInt(AltroDecoderError::ErrorType_t::ALTRO_MAPPING_ERROR), -1, -1);
  }
}

void RawToCellConverterSpec::handleFitError(const o2::emcal::CaloRawFitter::RawFitterError_t& fiterror, int ddlID, int cellID, int hwaddress)
{
  if (fiterror != CaloRawFitter::RawFitterError_t::BUNCH_NOT_OK) {
    // Display
    if (mNumErrorMessages < mMaxErrorMessages) {
      LOG(warning) << "Failure in raw fitting: " << CaloRawFitter::createErrorMessage(fiterror);
      mNumErrorMessages++;
      if (mNumErrorMessages == mMaxErrorMessages) {
        LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
      }
    } else {
      mErrorMessagesSuppressed++;
    }
    // Exclude BUNCH_NOT_OK also from raw error objects
    if (mCreateRawDataErrors) {
      mOutputDecoderErrors.emplace_back(ddlID, ErrorTypeFEE::ErrorSource_t::FIT_ERROR, CaloRawFitter::getErrorNumber(fiterror), cellID, hwaddress);
    }
  } else {
    LOG(debug2) << "Failure in raw fitting: " << CaloRawFitter::createErrorMessage(fiterror);
  }
}

void RawToCellConverterSpec::handleGainError(const o2::emcal::reconstructionerrors::GainError_t& errortype, int ddlID, int hwaddress)
{
  int fecID = mMapper->getFEEForChannelInDDL(ddlID, Channel::getFecIndexFromHwAddress(hwaddress), Channel::getBranchIndexFromHwAddress(hwaddress));
  if (mNumErrorMessages < mMaxErrorMessages) {
    switch (errortype) {
      case reconstructionerrors::GainError_t::HGNOLG:
        LOG(warning) << "FEC " << fecID << ": 0x" << std::hex << hwaddress << std::dec << " (DDL " << ddlID << ") has only high-gain out-of-range";
        break;
      case reconstructionerrors::GainError_t::LGNOHG:
        LOG(warning) << "FEC " << fecID << ": 0x" << std::hex << hwaddress << std::dec << " (DDL " << ddlID << ") has low gain but no high-gain";
        break;
      default:
        LOG(warning) << "FEC " << fecID << ": 0x" << std::hex << hwaddress << std::dec << " (DDL " << ddlID << ") falsely flagged as gain error";
    };
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  } else {
    mErrorMessagesSuppressed++;
  }
  if (mCreateRawDataErrors) {
    mOutputDecoderErrors.emplace_back(ddlID, ErrorTypeFEE::GAIN_ERROR, reconstructionerrors::getErrorCodeFromGainError(errortype), fecID, hwaddress);
  }
}

void RawToCellConverterSpec::handleGeometryError(const ModuleIndexException& error, int feeID, int cellID, int hwaddress, ChannelType_t chantype)
{
  if (mNumErrorMessages < mMaxErrorMessages) {
    std::string celltypename;
    switch (chantype) {
      case o2::emcal::ChannelType_t::HIGH_GAIN:
        celltypename = "high gain";
        break;
      case o2::emcal::ChannelType_t::LOW_GAIN:
        celltypename = "low-gain";
        break;
      case o2::emcal::ChannelType_t::TRU:
        celltypename = "TRU";
        break;
      case o2::emcal::ChannelType_t::LEDMON:
        celltypename = "LEDMON";
        break;
    };
    int supermoduleID = feeID / 2;
    switch (error.getModuleType()) {
      case ModuleIndexException::ModuleType_t::CELL_MODULE:
        LOG(warning) << "Sending invalid or negative cell ID " << error.getIndex() << " (SM " << supermoduleID << ", row " << error.getRow() << " - shift " << error.getRowShifted() << ", col " << error.getColumn() << " - shift " << error.getColumnShifted() << ") of type " << celltypename;
        break;
      case ModuleIndexException::ModuleType_t::LEDMON_MODULE:
        LOG(warning) << "Sending invalid or negative LEDMON module ID " << error.getIndex() << "( SM" << supermoduleID << ")";
        break;
    };
    mNumErrorMessages++;
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  } else {
    mErrorMessagesSuppressed++;
  }
  if (mCreateRawDataErrors) {
    mOutputDecoderErrors.emplace_back(feeID, ErrorTypeFEE::ErrorSource_t::GEOMETRY_ERROR, reconstructionerrors::getErrorCodeFromGeometryError(cellID < 0 ? reconstructionerrors::GeometryError_t::CELL_INDEX_NEGATIVE : reconstructionerrors::GeometryError_t::CELL_RANGE_EXCEED), cellID, hwaddress); // 0 -> Cell ID out of range
  }
}

void RawToCellConverterSpec::handlePageError(const RawDecodingError& e)
{
  if (mCreateRawDataErrors) {
    mOutputDecoderErrors.emplace_back(e.getFECID(), ErrorTypeFEE::ErrorSource_t::PAGE_ERROR, RawDecodingError::ErrorTypeToInt(e.getErrorType()), -1, -1);
  }
  if (mNumErrorMessages < mMaxErrorMessages) {
    LOG(warning) << " Page decoding: " << e.what() << " in FEE ID " << e.getFECID();
    mNumErrorMessages++;
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  } else {
    mErrorMessagesSuppressed++;
  }
}

void RawToCellConverterSpec::handleMinorPageError(const RawReaderMemory::MinorError& e)
{
  if (mCreateRawDataErrors) {
    mOutputDecoderErrors.emplace_back(e.getFEEID(), ErrorTypeFEE::ErrorSource_t::PAGE_ERROR, RawDecodingError::ErrorTypeToInt(e.getErrorType()), -1, -1);
  }
  if (mNumErrorMessages < mMaxErrorMessages) {
    LOG(warning) << " Page decoding: " << RawDecodingError::getErrorCodeDescription(e.getErrorType()) << " in FEE ID " << e.getFEEID();
    mNumErrorMessages++;
    if (mNumErrorMessages == mMaxErrorMessages) {
      LOG(warning) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
    }
  } else {
    mErrorMessagesSuppressed++;
  }
}

void RawToCellConverterSpec::sendData(framework::ProcessingContext& ctx, const std::vector<o2::emcal::Cell>& cells, const std::vector<o2::emcal::TriggerRecord>& triggers, const std::vector<ErrorTypeFEE>& decodingErrors) const
{
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  ctx.outputs().snapshot(framework::Output{originEMC, "CELLS", mSubspecification}, cells);
  ctx.outputs().snapshot(framework::Output{originEMC, "CELLSTRGR", mSubspecification}, triggers);
  if (mCreateRawDataErrors) {
    LOG(debug) << "Sending " << decodingErrors.size() << " decoding errors";
    ctx.outputs().snapshot(framework::Output{originEMC, "DECODERERR", mSubspecification}, decodingErrors);
  }
}

RawToCellConverterSpec::ModuleIndexException::ModuleIndexException(int moduleIndex, int column, int row, int shiftedColumn, int shiftedRow) : mModuleType(ModuleType_t::CELL_MODULE),
                                                                                                                                              mIndex(moduleIndex),
                                                                                                                                              mColumn(column),
                                                                                                                                              mRow(row),
                                                                                                                                              mColumnShifted(shiftedColumn),
                                                                                                                                              mRowShifted(shiftedRow) {}

RawToCellConverterSpec::ModuleIndexException::ModuleIndexException(int moduleIndex) : mModuleType(ModuleType_t::LEDMON_MODULE), mIndex(moduleIndex) {}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getRawToCellConverterSpec(bool askDISTSTF, bool disableDecodingErrors, int subspecification)
{
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  std::vector<o2::framework::OutputSpec> outputs;

  outputs.emplace_back(originEMC, "CELLS", subspecification, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(originEMC, "CELLSTRGR", subspecification, o2::framework::Lifetime::Timeframe);
  if (!disableDecodingErrors) {
    outputs.emplace_back(originEMC, "DECODERERR", subspecification, o2::framework::Lifetime::Timeframe);
  }

  std::vector<o2::framework::InputSpec> inputs{{"stf", o2::framework::ConcreteDataTypeMatcher{originEMC, o2::header::gDataDescriptionRawData}, o2::framework::Lifetime::Timeframe}};
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }
  // CCDB objects
  auto calibhandler = std::make_shared<o2::emcal::CalibLoader>();
  calibhandler->enableRecoParams(true);
  calibhandler->enableFEEDCS(true);
  calibhandler->defineInputSpecs(inputs);

  return o2::framework::DataProcessorSpec{
    "EMCALRawToCellConverterSpec",
    inputs,
    outputs,
    o2::framework::adaptFromTask<o2::emcal::reco_workflow::RawToCellConverterSpec>(subspecification, !disableDecodingErrors, calibhandler),
    o2::framework::Options{
      {"fitmethod", o2::framework::VariantType::String, "gamma2", {"Fit method (standard or gamma2)"}},
      {"maxmessage", o2::framework::VariantType::Int, 100, {"Max. amout of error messages to be displayed"}},
      {"printtrailer", o2::framework::VariantType::Bool, false, {"Print RCU trailer (for debugging)"}},
      {"no-mergeHGLG", o2::framework::VariantType::Bool, false, {"Do not merge HG and LG channels for same tower"}},
      {"no-checkactivelinks", o2::framework::VariantType::Bool, false, {"Do not check for active links per BC"}},
      {"no-evalpedestal", o2::framework::VariantType::Bool, false, {"Disable pedestal evaluation"}}}};
}
