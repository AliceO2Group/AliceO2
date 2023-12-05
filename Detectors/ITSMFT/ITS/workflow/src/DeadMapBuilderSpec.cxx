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

/// @file   DeadMapBuilderSpec.cxx

#include "ITSWorkflow/DeadMapBuilderSpec.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/CompCluster.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace its
{




//////////////////////////////////////////////////////////////////////////////
// Default constructor
  ITSDeadMapBuilder::ITSDeadMapBuilder(const ITSDMInpConf& inpConf, std::string datasource)
  : mDataSource(datasource)
{
  mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSDeadMapBuilder");
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
ITSDeadMapBuilder::~ITSDeadMapBuilder()
{
  // Clear dynamic memory

    
}

//////////////////////////////////////////////////////////////////////////////
void ITSDeadMapBuilder::init(InitContext& ic)
{
  LOGF(info, "ITSDeadMapBuilder init...", mSelfName);

  mTreeObject->Branch("orbit",&mFirstOrbitTF);
  mTreeObject->Branch("deadmap",&mDeadMapTF);

  mTFSampling = ic.options().get<int>("tf-sampling");
  DebugMode = ic.options().get<bool>("debug");
  mTFLength = ic.options().get<int>("tf-length");
  mLocalOutputDir = ic.options().get<std::string>("output-dir");

  LOG(info) << "Sampling one TF every "<<mTFSampling;


  return;
}





//////////////////////////////////////////////////////////////////////////////
// Grouping chips in lanes
short int ITSDeadMapBuilder::getLaneIDFromChip(short int chip)
{
  if (chip < N_CHIPS_IB) return chip;
  else return N_CHIPS_IB + (short int)((chip-N_CHIPS_IB)/7);
}


  ///////////////////////////////////////////////
  // to be imported by any code using the deadmap to traslate tree entries in lane lists
  // TODO: cross check once the encoding rule is finalized
std::vector<short int> ITSDeadMapBuilder::decodeITSMapWord(ULong64_t word){
  
std::vector<short int> lanelist{};

  if ((word & 0x1) == 0x0) { // _t0
    for (int l=0; l<7; l++){
      short int lanepp = (short int)(( word >> (9*l+1)) & 0x1FF);
      if (lanepp == 0) break;
      lanelist.push_back(lanepp-1);
    }
  }
  else if ((word & 0xF) == 0x1){ // _t1
    for (int l=0; l<5; l++){
      short int lanepp = (short int)(( word >> (12*l+4)) & 0xFFF);
      if (lanepp == 0) break;
      lanelist.push_back(lanepp-1);
    }
  }
  else if ((word & 0xF) == 0x3){ // _ t2
    short int lanelowpp = (short int)((word >> 4) & 0xFFF);
    short int laneuppp =  (short int)((word >> 16) & 0xFFF);
    for (short int lane = lanelowpp; lane <= laneuppp; lane++) lanelist.push_back(lane-1);
  }
  else{ // word not recognized
    lanelist.push_back(-1);
  }
  return lanelist;
}


//////////////////////////////////////////////////////////////////////////////

void ITSDeadMapBuilder::finalizeOutput()
{
  std::string localoutfilename = mLocalOutputDir+"/object.root";
  TFile outfile(localoutfilename.c_str(),"RECREATE");
  outfile.cd();
  mTreeObject->Write();
  outfile.Close();

  if (DebugMode){
    std::string localoutfilename2 = mLocalOutputDir+"/time.root";
    TFile outfile2(localoutfilename2.c_str(),"RECREATE");
    outfile2.cd();
    
    Htime->Write();
    outfile2.Close();
  }
  return;

} // finalizeOutput





//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get info from previous stf decoder workflow, then loop over readout frames
//     (ROFs) to count hits and extract thresholds
void ITSDeadMapBuilder::run(ProcessingContext& pc)
{
  if (mRunStopRequested) { // give up when run stop request arrived
    return;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  int difference;
  start = std::chrono::high_resolution_clock::now();

  mTFCounter++;

  
  mFirstOrbitTF = pc.services().get<o2::framework::TimingInfo>().firstTForbit;

  if ( (Long64_t)(mFirstOrbitTF / mTFLength) % mTFSampling != 0) return;

  mStepCounter++;
  LOG(info) << "Processing step #"<<mStepCounter<<" out of "<<mTFCounter<<" TF received. First orbit "<<mFirstOrbitTF;
  

  mChipsAlive.clear();
  mLanesAlive.clear();
  mDeadMapTF->clear();
  


  bool newchip = false;
  
  if (mDataSource == "digits"){
    const auto elements = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("elements");
    const  auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROFs");
    for (const auto& rof : ROFs) {
      auto elementsInTF = rof.getROFData(elements);
      for (const auto& el : elementsInTF){
	short int chipID = (short int)el.getChipIndex();
	newchip = mChipsAlive.insert(chipID).second;
	mLanesAlive.insert(getLaneIDFromChip(chipID));
      }
    }
  }
  else if (mDataSource == "clusters"){
    const  auto elements = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("elements");
    const  auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROFs");
    for (const auto& rof : ROFs) {
      auto elementsInTF = rof.getROFData(elements);
      for (const auto& el : elementsInTF){
	short int chipID = (short int)el.getSensorID();
	newchip = mChipsAlive.insert(chipID).second;
	mLanesAlive.insert(getLaneIDFromChip(chipID));
      }
    }
  }
  else if (mDataSource == "chipsstatus"){
    const auto elements = pc.inputs().get<std::vector<char>>("elements");
    for (short int chipID=0; chipID < elements.size(); chipID++){
      if (elements.at(chipID)) {
	newchip = mChipsAlive.insert(chipID).second;
	mLanesAlive.insert(getLaneIDFromChip(chipID));
      }
    }
  }

 
  mChipsAlive.insert(N_CHIPS);
  mLanesAlive.insert(getLaneIDFromChip(N_CHIPS));


  

  std::set<short int>::iterator laneIt = mLanesAlive.begin();

  short int laneLow = -1, laneUp = -1;

  int dcount_t0 = 0, dcount_t1 = 0, dcount_t2 = 0;
  mapelement_t0 = 0x0;
  mapelement_t1 = 0x1;
  mapelement_t2 = 0x3;
  
  for (int ilane = 0; ilane < mLanesAlive.size(); ilane++){

    laneLow = laneUp; 
    laneUp = *laneIt;
    laneIt++;

    if (laneUp -laneLow -1 > 6){  // more than 6 lanes (or full stave) --> better to save in a single word
      mapelement_t2 = mapelement_t2 | ( (ULong64_t)((laneLow+2) & 0xFFF) << 4);
      mapelement_t2 = mapelement_t2 | ( (ULong64_t)((laneUp-1+1) & 0xFFF) << 16);
      dcount_t2 += (laneUp -laneLow -1);
      mDeadMapTF->push_back(mapelement_t2);
      mapelement_t2 = 0x3;
    }

    else{
       for (short int idead = laneLow+1; idead < laneUp; idead++ ){
        
         if (idead < N_CHIPS_IB){ // IB;
       	if (DebugMode) LOG(info)<<"-DEBUG- low/up:"<<laneLow<<"/"<<laneUp<<" - Dead lane:"<<idead;
       	mapelement_t0 = mapelement_t0 | ( (ULong64_t)((idead+1) & 0x1FF) << ( 1 + 9*(dcount_t0 % 7)));
       	if (DebugMode) LOG(info)<<"-DEBUG- New map element type 0 "<<mapelement_t0;
       	dcount_t0++;
       	if (dcount_t0 % 7 == 0){
       	  mDeadMapTF->push_back(mapelement_t0);
       	  if (DebugMode) LOG(info)<<"-DEBUG- Pushing up map element type 0 "<<mapelement_t0;
       	  mapelement_t0 = 0x0;
       	}
         }
         else{ // OB
       	mapelement_t1 = mapelement_t1 | ( (ULong64_t)((idead+1) & 0xFFF) << ( 4 + 12*(dcount_t1 % 5)));
       	dcount_t1++;
       	if (dcount_t1 % 5 == 0){
       	  mDeadMapTF->push_back(mapelement_t1);
       	  mapelement_t1 = 0x1;
       	}
         }
         
       } // end loop over dead lanes
    }

    
  } // end loop over alive lanes set
      
  if (mapelement_t0 != 0x0) mDeadMapTF->push_back(mapelement_t0);
  if (mapelement_t1 != 0x1) mDeadMapTF->push_back(mapelement_t1);

  int dcountTot = dcount_t0 + dcount_t1 + dcount_t2;

  LOG(info) << "Dead lanes: "<<dcountTot<<", t0|t1|t2: "<<dcount_t0<<"|"<<dcount_t1<<"|"<<dcount_t2<<", saved in "<<mDeadMapTF->size()<<" words.";

  mTreeObject->Fill();

  end = std::chrono::high_resolution_clock::now();
  difference = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  LOG(info) << "Elapsed time: "<<difference/1000.<<" ms";
  if (DebugMode) Htime->Fill(difference/1000., 1.*(dcountTot));
   
  return;
}


 
void ITSDeadMapBuilder::finalize()
{
  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSDeadMapBuilder::PrepareOutputCcdb(DataAllocator& output)
{
  if (DebugMode) LOG(info)<<"-DEBUG- Entering PrepareOutputCcdb.";

  long tstart = o2::ccdb::getCurrentTimestamp();
  long secinyear = 365L* 24  * 3600 * 1000;
  long tend = o2::ccdb::getFutureTimestamp(secinyear);

  std::map<std::string, std::string> md = {
    {"test", "content"}};

  std::string path("ITS/Calib/");
  std::string name_str = "time_dead_map";

  o2::ccdb::CcdbObjectInfo info((path+name_str),"time_dead_map","timedeadmap.root",md,tstart,tend);
  
  std::vector<int> dummyobj ={1};
  auto image = o2::ccdb::CcdbApi::createObjectImage(mTreeObject, &info);
  
  info.setFileName("time_deadmap.root");

  LOG(info) << "Sending object "<< info.getPath() << "/" << info.getFileName()
	    << " of size "<< image->size() << "bytes, valid for" 
	    << info.getStartValidityTimestamp()<<" : "<<info.getEndValidityTimestamp();

  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_TimeDeadMap", 0}, *image);
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_TimeDeadMap", 0}, info);


  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSDeadMapBuilder::endOfStream(EndOfStreamContext& ec)
{
  if (!isEnded && !mRunStopRequested) {
    LOGF(info, "endOfStream report:", mSelfName);
    if (1 /*isFinalizeEos*/) {
      finalize();
    }
    finalizeOutput();
    PrepareOutputCcdb(ec.outputs());
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: simply close the latest tree
void ITSDeadMapBuilder::stop()
{
  if (!isEnded) {
    LOGF(info, "stop() report:", mSelfName);
    if (1 /*isFinalizeEos*/) {
      finalize();
    }
    this->finalizeOutput();
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSDeadMapBuilderSpec(const ITSDMInpConf& inpConf, std::string datasource)
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
  std::vector<InputSpec> inputs;

  if (datasource == "digits"){
    inputs.emplace_back("elements", detOrig, "DIGITS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROFs", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  }
  else if (datasource == "clusters"){
    inputs.emplace_back("elements",detOrig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROFs",detOrig,"CLUSTERSROF",0 ,Lifetime::Timeframe);
  }
  else if (datasource == "chipsstatus"){
    inputs.emplace_back("elements",detOrig,"CHIPSSTATUS",0 ,Lifetime::Timeframe);
  }
  else{
    return DataProcessorSpec{0x0}; // TODO: ADD PROTECTION
  }

  
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITS_TimeDeadMap"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITS_TimeDeadMap"});

  return DataProcessorSpec{
    "its-deadmap-builder_" + std::to_string(inpConf.chipModSel),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSDeadMapBuilder>(inpConf, datasource)},
    Options{{"debug", VariantType::Bool, false, {"Temporary debug mode"}},
            {"tf-sampling", VariantType::Int, 997, {"Process every Nth TF. Selection according to first TF Orbit."}},
	    {"tf-length", VariantType::Int, 32, {"Orbits per TFs."}}, 
            {"output-dir", VariantType::String, "./", {"ROOT trees output directory."}}}};
}
} // namespace its
} // namespace o2
