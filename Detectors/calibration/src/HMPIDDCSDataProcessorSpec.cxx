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


#include <unistd.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "HMPIDCalibration/HMPIDDCSProcessor.h"
//#include "HMPIDWorkFlows/HMPIDDataProcessorSpec.h" // headerfile currently
//#include "testWorkflow/HMPIDDCSDataProcessorSpec.h" // headerfile currently  
#include "HMPIDCalibration/HMPIDDCSDataProcessorSpec.h"		     // in same folder 	


#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"

using namespace o2::framework;

namespace o2
{
namespace hmpid
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using namespace o2::ccdb;
using CcdbManager = o2::ccdb::BasicCCDBManager;
using clbUtils = o2::calibration::Utils;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

//==========================================================================
  void HMPIDDCSDataProcessor::init(o2::framework::InitContext& ic)
  {
   
    std::vector<DPID> vect;
     

    // GV/CZ: is this necessary for HMPID?
	    mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
	    if (mDPsUpdateInterval == 0) {
	      LOG(error) << "HMPID DPs update interval set to zero seconds --> changed to 60";
	      mDPsUpdateInterval = 60;
	    }

   // GV/CZ:


    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");

    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = CcdbManager::instance();
      mgr.setURL(ccdbpath);
      CcdbApi api;
      api.init(mgr.getURL());
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("HMPID/Config/DCSDPconfig", ts); // correct??
      for (auto& i : *dpid2DataDesc) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";
      std::vector<std::string> aliases = {"tof_hv_vp_[00..89]", "tof_hv_vn_[00..89]", "tof_hv_ip_[00..89]", "tof_hv_in_[00..89]"};
      std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);
      
      for (const auto& i : expaliases) {
        vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
      }
      //std::vector<std::string> aliasesInt = {"TOF_FEACSTATUS_[00..71]"};
      //std::vector<std::string> expaliasesInt = o2::dcs::expandAliases(aliasesInt);
      //for (const auto& i : expaliasesInt) {  vect.emplace_back(i, o2::dcs::DPVAL_INT);}
    } // end else 

    LOG(info) << "Listing Data Points for HMPID:";
    for (auto& i : vect) {
      LOG(info) << i;
    }

    mProcessor = std::make_unique<o2::hmpid::HMPIDDCSProcessor>();
    bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
    LOG(info) << " ************************* Verbose?" << useVerboseMode;
    if (useVerboseMode) { // def in HMPIDCSProcessor.h
      mProcessor->useVerboseMode();
    }
    //mProcessor->init(vect); hmp??
    mTimer = HighResClock::now();
  }

//==========================================================================

  void HMPIDDCSDataProcessor::run(o2::framework::ProcessingContext& pc) 
  {
    auto startValidity = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    auto timeNow = HighResClock::now();
    
    if (startValidity == 0xffffffffffffffff) {                                                                   // it means it is not set
      startValidity = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }
    mProcessor->setStartValidity(startValidity); // in HMPIDDCSProcessor.h
    
    
    // process datapoints: 
    mProcessor->process(dps);
     // not necessary for HMPID??: 
    /*
    Duration elapsedTime = timeNow - mTimer; // in seconds
    if (elapsedTime.count() >= mDPsUpdateInterval) {
      sendDPsoutput(pc.outputs());
      mTimer = timeNow;
    }  */
    
    
    // as of now, the finalize-function processes both RefIndex and ChargeCut
    mProcessor->finalize(); // finalize, in HMPID-function
    
    //sendChargeThresOutput(pc.outputs()); // should be in run ??
    //sendRefIndexOutput(pc.outputs());
  }
  
//==========================================================================
  void HMPIDDCSDataProcessor::endOfStream(o2::framework::EndOfStreamContext& ec) 
  {
    sendChargeThresOutput(ec.outputs()); // should be in run ??
    sendRefIndexOutput(ec.outputs());
  }



  //=== send charge threshold(arQthre)=========================================
  void HMPIDDCSDataProcessor::sendChargeThresOutput(DataAllocator& output)
  {
    // fill CCDB with ChargeThres (arQthre)   
    const auto& payload = mProcessor->getChargeCutObj();   // or get  
    //const std::array<TF1,42> mChargeCut;
    
    auto& info = mProcessor->getHmpidChargeCutInfo();   // OK, but maybe change function and var names  
    
   
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ChargeCut", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ChargeCut", 0}, info);
  }

  //====send RefIndex (arrMean)=====================================================
  void HMPIDDCSDataProcessor::sendRefIndexOutput(DataAllocator& output)
  {
      // fill CCDB with RefIndex (arrMean)            
      const auto& payload = mProcessor->getRefIndexObj();   
      //const std::array<TF1,43> mRefIndex;
   
      auto& info = mProcessor->getccdbREF_INDEXsInfo(); // OK, but maybe change function and var names  

      
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "RefIndex", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "RefIndex", 0}, info);

  } // end class HMPIDDCSDataProcessor

} // namespace hmpid

namespace framework
{

DataProcessorSpec getHMPIDDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
 
  // NB! probably change Lifetime::Sporadic
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ChargeCut"}, Lifetime::Sporadic); outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ChargeCut"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "RefIndex"}, Lifetime::Sporadic); outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "RefIndex"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "HMPID-dcs-data-processor",
    Inputs{{"input", "DCS", "HMPIDDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::hmpid::HMPIDDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}//,
            // NB! commented out DPs update interval for HMPID:
            //{"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}
            }};
}

} // namespace framework
} // namespace o2


