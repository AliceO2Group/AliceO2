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

// calibration/HMPIDCalibration header-files:
#include "HMPIDCalibration/HMPIDDCSProcessor.h"
#include "HMPIDCalibration/HMPIDDCSTime.h"
// HMPID Base  
#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"

// Root classes:
#include <TF1.h>                  
#include <TF2.h>                  
#include <TGraph.h>            

// miscallanous libraries
#include <memory>
#include <deque> 
#include <gsl/gsl> 
#include <string>

// O2 includes: 
#include "Framework/Logger.h"
#include "DetectorsDCS/DataPointCompositeObject.h" 
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h" 
#include "CCDB/CcdbObjectInfo.h" 
#include "CCDB/CcdbApi.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h" // o2::calibration::dcs,  o2::calibration::Utils
//using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
//using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;


using namespace o2::dcs;


namespace o2::hmpid {
	
/*
void HMPIDDCSProcessor::init(const std::vector<DPID>& pids)
{	
	for(const auto& it:pids) mPids[it] = false;
} */	

void HMPIDDCSProcessor::process(const gsl::span<const DPCOM> dps)
{
  if (dps.size() == 0) {
    return;
  }

  if (mVerbose) {
    LOG(info) << "\n\n\nProcessing new DCS DP map\n-----------------";
  }

  if (!mFirstTimeSet) {
    mFirstTime = mStartValidity;
    mFirstTimeSet = true;
  }

  for (const auto& dp : dps) {

    const auto& el = mPids.find(dp.id); // hmp?
    if (el == mPids.end()){		// hmp?	
	LOG(info) << "DP " << dp.id << "Not found, will not be processed";
	continue;			// hmp?
    }					// hmp?

    //mPids[it.id] = true;// hmp?
	
    const std::string_view alias(dp.id.get_alias());
    const auto detector_id = alias.substr(0, 7);
    const auto ir_id = alias.substr(0,16);

    // check if given dp is from HMPID
    // check first if IR:
    if (ir_id == IR_ID) {
	processIR(dp); 
    } 
// if not IR, check if other DP in HMPID (pressure, temp, HV): 
    else if (detector_id==HMPID_ID){
	processHMPID(dp); 
    }  else  LOG(debug) << "Unknown data point: {}"<< alias;
  } // end for
}	

// if the string of the dp contains the HMPID-specifier "HMP_DET",
// but not the IR-specifier "HMP_DET/HMP_INFR" : 
void HMPIDDCSProcessor::processHMPID(DPCOM dp)
{
   
  const std::string alias(dp.id.get_alias());

 if ( alias.substr(alias.length()-7) == TEMP_IN_ID ) {
      LOG(info) << "Temperature_in DP: {}"<< alias;
      fill_InTemperature(dp); 
    } else if (alias.substr(alias.length()-8) == TEMP_OUT_ID) {
      LOG(info) << "Temperature_out DP: {}"<< alias;
      fill_OutTemperature(dp);
    } else if (alias.substr(alias.length()-4) == HV_ID) {
      LOG(info) << "HV DP: {}"<< alias;
      fillHV(dp);
    } else if (alias.substr(alias.length()-17) == ENV_PRESS_ID ) {
      LOG(info) << "Environment Pressure DP: {}"<< alias;
      fillChamberPressures(dp);
    } else if (alias.substr(alias.length()-18) == CH_PRESS_ID) {
      LOG(info) << "Chamber Pressure DP: {}"<< alias;
      fillEnvironmentPressure(dp);
    } else {
      LOG(debug) << "Unknown data point: {}"<< alias;
    }	    
}

// if the string of the dp contains the HMPID-IR-specifier "HMP_DET/HMP_INFR"	
void HMPIDDCSProcessor::processIR(DPCOM dp)
{
    	const std::string alias(dp.id.get_alias());
     	auto specify_id = alias.substr(alias.length()-9);
	std::cout << alias << std::endl;
	std::cout << specify_id << std::endl;
	
	// is it desired to know the index of the IRs [0..29]?: 
	//auto numIR = subStringToInt(alias, indexOfIR );
	//auto numIR_2nd =  subStringToInt(alias, indexOfIR+1);
	
	
	
	// if fetched datapoints is out of range, exit function: 
	//if(numIR == -1) 
	//{ 
	//	LOG(debug) << "Datapoint index out of range: "<< numIR;
	//	return;
	//}
	
	// if there are two digits : 
	//if (numIR_2nd!=-1 ) 
	{
	//	numIR = numIR*10 +numIR_2nd;
	//}
	
       	//if(numIR < 30 && numIR >0) 				 
	//{
		if(alias.substr(alias.length()-10) == WAVE_LEN_ID) {
			LOG(info) << "WAVE_LEN_ID DP: "<< alias;
		}  else if(specify_id == FREON_CELL_ID) { 
			LOG(info) << "FREON_CELL_ID DP: "<< alias;
		}  else if(specify_id == ARGON_CELL_ID) { 
			LOG(info) << "ARGON_CELL_ID DP: "<< alias;
		}
		else if(specify_id == REF_ID) { 
			if( alias.substr(alias.length()-14) ==  ARGON_REF_ID){
				LOG(info) << "ARGON_REF_ID DP: "<< alias;
			} else if( alias.substr(alias.length()-14) ==  FREON_REF_ID){
				LOG(info) << "FREON_REF_ID DP: "<< alias;
			}      LOG(debug) << "Unknown data point: "<< alias;
		} else LOG(debug) << "Datapoint not found: "<< alias;
		std::cout << "==================" << std::endl;
	  	}
	//}   else LOG(debug) << "Datapoint index out of range: "<< numIR;
}

	
double HMPIDDCSProcessor::ProcTrans()
{   
    for(int i=0; i<30; i++) 
      {    		
            //==== evaluate wavelenght ===================================================================
            //("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght",i)); 
            if(waveLen[i].size() == 0) // if there is no entries 
            { 
		LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght  -----> Default E mean used!!!!!",i);
		return DefaultEMean();	// will break this entry in foor loop
            }  
	    //  pVal=(AliDCSValue*)pWaveLenght->At(0); // get first element, (i.e. pointer to TObject at index 0)
            	dpWaveLen =  (waveLen[i]).at(0);   
	    	
	    if(dpWaveLen.id.get_type() == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	    {
	        lambda = o2::dcs::getValue<double>(dpWaveLen); // Double_t lambda = pVal->GetFloat();
	    } else {
                LOG(debug) << Form("Not correct datatype for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght  -----> Default E mean used!!!!!",i);
                return DefaultEMean();
	    }
	    
	    
            if(lambda<150. || lambda>230.)
            { 
                LOG(debug) << Form("Wrong value for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); // will break this entry in foor loop
            } 
	    
            //find photon energy E in eV from radiation wavelength λ in nm
	    nm2eV = 1239.842609;	// 1239.842609 from nm to eV 
            photEn = nm2eV/lambda;     // photon energy	    
		    		    

            if(photEn<o2::hmpid::Param::ePhotMin() || photEn>o2::hmpid::Param::ePhotMax()) continue; // if photon energy is out of range
            
	    
	    
            // ===== evaluate phototube current for argon reference ==============================================================
            if(argonRef[i].size() == 0)
            { 
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonReference  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); 
            } 
      	
            dpArgonRef =  (argonRef[i]).at(0); // pVal=(AliDCSValue*)pArgonRef->At(0);    
	    
	    if(dpArgonRef.id.get_type() == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	    {
            	aRefArgon = o2::dcs::getValue<double>(dpArgonRef);// Double_t aRefArgon = pVal->GetFloat();
	    } else {
                LOG(debug) << Form("Not correct datatype for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonReference  -----> Default E mean used!!!!!",i);
                return DefaultEMean();
	    }
	    
        
	    
            //===== evaluate phototube current for argon cell  ==============================================================
            if(argonCell[i].size() == 0)
            { 
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonCell  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); 
            } 
	    
            dpArgonCell  =  (argonCell[i]).at(0); // pVal=(AliDCSValue*)pArgonRef->At(0);  
	    
	    if(dpArgonCell.id.get_type() == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	    {
            	aCellArgon = o2::dcs::getValue<double>(dpArgonCell);// Double_t aCellArgon = pVal->GetFloat(); 
	    } else {
                LOG(debug) << Form("Not correct datatype for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonCell  -----> Default E mean used!!!!!",i);
                return DefaultEMean();
	    }
		
            
	    
            //====evaluate phototube current for freon reference ==============================================================	
            if(freonRef[i].size() == 0)
            { 
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Reference  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); // to be checked
            } 
            
	    dpFreonRef  =  (freonRef[i]).at(0); //pVal=(AliDCSValue*)pFreonRef->At(0);  
      
	    if(dpFreonRef.id.get_type() == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	    {
            	aRefFreon = o2::dcs::getValue<double>(dpFreonRef);//   Double_t aRefFreon = pVal->GetFloat(); 
	    } else {
                LOG(debug) << Form("Not correct datatype for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Reference  -----> Default E mean used!!!!!",i);
                return DefaultEMean();
	    }
	    
		
	    
            // ==== evaluate phototube current for freon cell ==============================================================
            if(freonCell[i].size() == 0)
            {
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Cell  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); 
            }
            
	    dpFreonCell =  (freonCell[i]).at(0); // pVal=(AliDCSValue*)pFreonCell->At(0);
            
	    if(dpFreonCell.id.get_type() == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	    {
	     	aCellFreon = o2::dcs::getValue<double>(dpFreonCell);//   Double_t aCellFreon = pVal->GetFloat();
	    } else {
                LOG(debug) << Form("Not correct datatype for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Cell  -----> Default E mean used!!!!!",i);
                return DefaultEMean();
	    }
		    
	    
         
           //evaluate correction factor to calculate trasparency (Ref. NIMA 486 (2002) 590-609)
            
            //Double_t aN1 = AliHMPIDParam::NIdxRad(photEn,tRefCR5);
            //Double_t aN2 = AliHMPIDParam::NMgF2Idx(photEn);
            //Double_t aN3 = 1;                              // Argon Idx
        
           // Double_t aR1               = ((aN1 - aN2)*(aN1 - aN2))/((aN1 + aN2)*(aN1 + aN2));
          //  Double_t aR2               = ((aN2 - aN3)*(aN2 - aN3))/((aN2 + aN3)*(aN2 + aN3));
           // Double_t aT1               = (1 - aR1);
           // Double_t aT2               = (1 - aR2);
           // Double_t aCorrFactor       = (aT1*aT1)/(aT2*aT2);
        
            // evaluate 15 mm of thickness C6F14 Trans
            
            
            aConvFactor = 1.0 - 0.3/1.8;         
                
            if(aRefFreon*aRefArgon>0)
            {
              aTransRad  = TMath::Power((aCellFreon/aRefFreon)/(aCellArgon/aRefArgon)*aCorrFactor[i],aConvFactor); 
            }
            else
            {
              return DefaultEMean();
            }
            

            // evaluate 0.5 mm of thickness SiO2 Trans
	    	    
		    // TMath : Double_t	Exp(Double_t x)
		    aTransSiO2 = TMath::Exp(-0.5/o2::hmpid::Param::lAbsWin(photEn)); 

		    // evaluate 80 cm of thickness Gap (low density CH4) transparency
		    aTransGap  = TMath::Exp(-80./o2::hmpid::Param::lAbsGap(photEn)); 

		    // evaluate CsI quantum efficiency
		    aCsIQE            = o2::hmpid::Param::qEffCSI(photEn);  
        
            // evaluate total convolution of all material optical properties
            aTotConvolution   = aTransRad*aTransSiO2*aTransGap*aCsIQE;
        
            sEnergProb+=aTotConvolution*photEn;  
          
            sProb+=aTotConvolution;  

	    // Evaluate timestamps : 
	    //const std::vector<DPCOM> arr[5] = {waveLen[i], argonRef[i], argonCell[i], freonRef[i], freonCell[i]};
	    std::vector<DPCOM> irTSArray[5] = {waveLen[i], argonRef[i], argonCell[i], freonRef[i], freonCell[i]};
	   
	    auto minTime = HMPIDDCSTime::getMinTimeArr(irTSArray);
	    auto maxTime = HMPIDDCSTime::getMaxTimeArr(irTSArray);
	    if(minTime < mTimeArNmean.first) mTimeArNmean.first = minTime;    
	    if(maxTime > mTimeArNmean.last) mTimeArNmean.last = maxTime; 
		
        }
      if(sProb>0) 
      {
        eMean = sEnergProb/sProb;
      }
      else
      {
        return DefaultEMean();
      }
      //Log(Form(" Mean energy photon calculated ---> %f eV ",eMean));
    
      if(eMean<o2::hmpid::Param::ePhotMin() || eMean>o2::hmpid::Param::ePhotMax())	{return DefaultEMean(); } 
      
      return eMean;
      
} // end ProcTrans

double HMPIDDCSProcessor::DefaultEMean(){
	double eMean = 6.675;
        LOG(debug) << Form(" Mean energy photon calculated ---> %f eV ",eMean);
	return eMean;
}



// ==== Functions that are called after run is finished ======================== 

void HMPIDDCSProcessor::finalizeEnvPressure() 
{ 
	// if environment-pressure has entries 
	if(pEnv.size() != 0){ 

		auto minTime = HMPIDDCSTime::getMinTime(pEnv);
		if(minTime < mTimeQThresh.first) mTimeQThresh.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(pEnv);
		if(maxTime > mTimeQThresh.last) mTimeQThresh.last = maxTime;

		int cntEnvPressure = 0;

		//TGraph *pGrPenv = new TGraph;  BAD
		// should not use new/delete for raw-pointer
		// Workaround:
		std::unique_ptr<TGraph> pGrPenv;
		pGrPenv.reset(new TGraph); 

		for(DPCOM dp : pEnv){
			pGrPenv->SetPoint(cntEnvPressure++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		if(cntEnvPressure==1) { 
		 pGrPenv->GetPoint(0,xP,yP);       
		 new TF1("Penv",Form("%f",yP), minTime,maxTime);//fStartTime,fEndTime);
		} else {
		  pGrPenv->Fit(new TF1("Penv","1000+x*[0]",minTime,maxTime),"Q");
		} // delete pGrPenv;
	} else LOG(debug) << Form("No entries in environment pressure");
}


void HMPIDDCSProcessor::finalizeHV_Entry(Int_t iCh,Int_t iSec) 
{ 
	// check if given element has entries
	if(dpcomHV[3*iCh+iSec].size() != 0){    
		
		std::unique_ptr<TGraph> pGrHV;
		pGrHV.reset(new TGraph);
		cntHV=0; 

		auto minTime = HMPIDDCSTime::getMinTime(dpcomHV[3*iCh+iSec]);
		if(minTime < mTimeQThresh.first) mTimeQThresh.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(dpcomHV[3*iCh+iSec]);
		if(maxTime > mTimeQThresh.last) mTimeQThresh.last = maxTime;
		
		for(DPCOM dp : dpcomHV[3*iCh+iSec]){
			pGrHV->SetPoint(cntHV++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		if(cntHV==1) { 
		 pGrHV->GetPoint(0,xP,yP);           
		new TF1(Form("HV%i_%i",iCh,iSec),Form("%f",yP),minTime,maxTime);       
		} else {
		pGrHV->Fit(new TF1(Form("HV%i_%i",iCh,iSec),"[0]+x*[1]",minTime,maxTime,"Q"));      
		} 
	} else LOG(debug) << Form("No entries in HV for chamber %i, section %i",iCh,iSec);
}

	
void HMPIDDCSProcessor::finalizeChPressureEntry(Int_t iCh) 
{ 	
	// check if given element has entries
	if(pChamber[iCh].size() != 0){	

		std::unique_ptr<TGraph> pGrP;
		pGrP.reset(new TGraph);

		 cntChPressure=0;
		//std::unique_ptr<TGraph> pGrP
		auto minTime = HMPIDDCSTime::getMinTime(pChamber[iCh]);
		if(minTime < mTimeQThresh.first) mTimeQThresh.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(pChamber[iCh]);
		if(maxTime > mTimeQThresh.last) mTimeQThresh.last = maxTime;
		 		
		for(DPCOM dp : pChamber[iCh]){
			pGrP->SetPoint(cntChPressure++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		if(cntChPressure==1) { 
			pGrP->GetPoint(0,xP,yP);           
		 	new  TF1(Form("P%i",iCh),Form("%f",yP),minTime, maxTime);              
		} else {
		pGrP->Fit(new TF1(Form("P%i",iCh),"[0] + x*[1]",minTime,maxTime),"Q");       
		} 
	} else  LOG(debug) << Form("No entries in chamber-pressure for chamber %i",iCh); 	
}

			
void HMPIDDCSProcessor::finalizeTempOutEntry(Int_t iCh,Int_t iRad) 
{ 	
	// check if given element has entries
	if(tempOut[3*iCh+iRad].size() != 0){	
		auto minTime = HMPIDDCSTime::getMinTime(tempOut[3*iCh+iRad]);
		if(minTime < mTimeArNmean.first) mTimeArNmean.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(tempOut[3*iCh+iRad]);
		if(maxTime > mTimeArNmean.last) mTimeArNmean.last = maxTime;
		//TGraph *pGrTOut = new TGraph;  BAD
		std::unique_ptr<TGraph> pGrTOut;
		pGrTOut.reset(new TGraph);

		for(DPCOM dp : tempOut[3*iCh+iRad]){
			pGrTOut->SetPoint(cntTOut++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		std::unique_ptr<TF1> pTout;// std::make_unique<TF1>() ;
		pTout.reset(new TF1(Form("Tout%i%i",iCh,iRad),"[0]+[1]*x",minTime,maxTime));

		
		if(cntTOut==1) { 
		
		 pGrTOut->GetPoint(0,xP,yP);
		 pTout->SetParameter(0,yP);
		 pTout->SetParameter(1,0);
		} else {
	
		pGrTOut->Fit(pTout.get(),"Q");
		arNmean.at(6*iCh+2*iRad) = *(pTout.get());//pTout[3*iCh+iRad];
		} // delete pGrTOut; */
	} else  LOG(debug) << Form("No entries in temp-out for chamber %i, radiator %i",iCh,iRad); 
}

void HMPIDDCSProcessor::finalizeTempInEntry(Int_t iCh,Int_t iRad)
{ 	
	// check if given element has entries
	if(tempIn[3*iCh+iRad].size() != 0){	
		auto minTime = HMPIDDCSTime::getMinTime(tempIn[3*iCh+iRad]);
		if(minTime < mTimeArNmean.first) mTimeArNmean.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(tempOut[3*iCh+iRad]);
		if(maxTime > mTimeArNmean.last) mTimeArNmean.last = maxTime;

		std::unique_ptr<TGraph> pGrTIn;
		pGrTIn.reset(new TGraph);

		for(DPCOM dp : tempIn[3*iCh+iRad]){
			pGrTIn->SetPoint(cntTOut++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		std::unique_ptr<TF1> pTin;
		pTin.reset(new TF1(Form("Tin%i%i",iCh,iRad),"[0]+[1]*x",minTime,maxTime));

		
		if(cntTOut==1) { 
		
		 pGrTIn->GetPoint(0,xP,yP);
		 pTin->SetParameter(0,yP);
		 pTin->SetParameter(1,0);
		} else {
	
		pGrTIn->Fit(pTin.get(),"Q");
		arNmean.at(6*iCh+2*iRad) = *(pTin.get());
		} 
	} else  LOG(debug) << Form("No entries in temp-out for chamber %i, radiator %i",iCh,iRad); 
}



void HMPIDDCSProcessor::fillChamberPressures(const DPCOM& dpcom)
{
	
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  
  
  if(type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
  {

 	// find chamber number:  
	auto chNum = subStringToInt(aliasStr, startI_chamberPressure);
	  
	// verify chamber-number 
	if (chNum > 6 || chNum < 0) {
		pChamber[chNum].push_back(dpcom);
	}  else LOG(debug)<< "Chamber Number out of range for Environment-pressure DP: {}"<< chNum;
			  
  } else LOG(debug)<< "Not correct specification for Environment-pressure DP: {}"<< aliasStr;
}


void HMPIDDCSProcessor::fillEnvironmentPressure(const DPCOM& dpcom) 
{	
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  

  if(type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	{	
	  	pEnv.push_back(dpcom); 
  } else {
	LOG(debug)<< "Not correct specification for Environment-pressure DP: {}" << aliasStr;
  }
}

// HV in each chamber_section = 7*3 --> will result in Q_thre  
void HMPIDDCSProcessor::fillHV(const DPCOM& dpcom)
{
  
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  

  if(type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
  { 	
	auto chNum = subStringToInt(aliasStr, startI_chamberHV);
	auto secNum = subStringToInt(aliasStr,  startI_sectorHV);
	
	// verify chamber- and sector-numbers
	if (chNum > 6 || chNum < 0) { 
		if (secNum > 5 || secNum < 0) {
			dpcomHV[6*chNum+secNum].push_back(dpcom);
		}  else LOG(debug)<< "Sector Number out of range for HV DP: {}"<< secNum;  
	}  else LOG(debug)<< "Chamber Number out of range for HV DP: {}"<< chNum;
	
  } else LOG(debug)<< "Not correct datatype for HV DP: {}"<< aliasStr;	
}

	
// Temp in (T1)  in each chamber_radiator = 7*3  
void HMPIDDCSProcessor::fill_InTemperature(const DPCOM& dpcom) 
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  
	
	  if(type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	  {	
		auto chNum = subStringToInt(aliasStr,  startI_chamberTemp);
		auto radNum = subStringToInt(aliasStr,  startI_radiatorTemp);
		  
		// verify chamber- and raiator-numbers  
		if (chNum > 6 || chNum < 0 ) {
			if (radNum > 2 || radNum < 0) {
				tempIn[3*chNum+radNum].push_back( dpcom); 
			}  else LOG(debug)<< "Radiator Number out of range for TempIn DP: {}"<< radNum;  
		}  else LOG(debug)<< "Chamber Number out of range for TempIn DP: {}"<< chNum; 		  

	} else LOG(debug)<< "Not correct datatype for TempIn DP: {}"<< aliasStr;	
}	
	
	
// Temp out (T2), in each chamber_radiator = 7*3  
void HMPIDDCSProcessor::fill_OutTemperature(const DPCOM& dpcom) 
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  
	
  if(type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
  {		
	auto chNum = subStringToInt(aliasStr,  startI_chamberTemp);
	auto radNum = subStringToInt(aliasStr,  startI_radiatorTemp);
	  
	// verify chamber- and raiator-numbers   
	if (chNum > 6 || chNum < 0) {
		if (radNum > 2 || radNum < 0) {
       			tempOut[3*chNum+radNum].push_back(dpcom);
		}  else LOG(debug)<< "Radiator Number out of range for TempOut DP: {}"<< radNum;  
	}  else LOG(debug)<< "Chamber Number out of range for TempOut DP: {}"<< chNum; 	
	  
  } else LOG(debug)<< "Not correct datatype for TempOut DP: {}"<< aliasStr;
}


void HMPIDDCSProcessor::finalize() 
{ 
	finalizeEnvPressure();
	for(Int_t iCh=0;iCh<7;iCh++){
		finalizeChPressureEntry(iCh);
		for(Int_t iRad=0;iRad<3;iRad++){
			// fills up entries 0..41 of arNmean
			finalizeTempInEntry(iCh,iRad);
			finalizeTempOutEntry(iCh,iRad); 
		}

		for(Int_t iSec=0;iSec<6;iSec++){
			finalizeHV_Entry(iCh,iSec);	
			// evaluate Qthre
			hvFirstTime = HMPIDDCSTime::getMinTime(dpcomHV[6*iCh+iSec]);
			hvLastTime = HMPIDDCSTime::getMaxTime(dpcomHV[6*iCh+iSec]);
			
        		arQthre.at(6*iCh+iSec) = *(new TF1(Form("HMP_QthreC%iS%i",iCh,iSec),Form("3*10^(3.01e-3*HV%i_%i - 4.72)+170745848*exp(-(P%i+Penv)*0.0162012)",iCh,iSec,iCh),hvFirstTime,hvLastTime)); //Photon energy mean			



		}
	}

	
	double eMean = ProcTrans();	 
	

        arNmean.at(42) = *(new TF1("HMP_PhotEmean",Form("%f",eMean),mTimeArNmean.first,mTimeArNmean.last));//Photon energy mean



	
	 // prepare CCDB: =============================================================================
	// static void prepareCCDBobjectInfo(T& obj, o2::ccdb::CcdbObjectInfo& info, const std::string& path,
	  //                 const std::map<std::string, std::string>& md, long start, long end = -1);  

	 std::map<std::string, std::string> md;
	 md["responsible"] = "NB!! CHANGE RESPONSIBLE";
	
	 // Refractive index (T_out, T_in, mean photon energy); mRefIndex contains class-def
	 o2::calibration::Utils::prepareCCDBobjectInfo(mRefIndex, mccdbREF_INDEX_Info, "HMPID/Calib/RefIndex", md, mStartValidity, o2::calibration::Utils::INFINITE_TIME);

	 // charge threshold; mChargeCut contains class-definition
	 		    o2::calibration::Utils::prepareCCDBobjectInfo(mChargeCut,mccdbCHARGE_CUT_Info , "HMPID/Calib/ChargeCut", md, mStartValidity, o2::calibration::Utils::INFINITE_TIME);	
}

	
uint64_t HMPIDDCSProcessor::processFlags(const uint64_t flags, const char* pid)
{

  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  // for now, I don't know how to use the flags, so I do nothing

  if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
    LOG(debug) << "KEEP_ALIVE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::END_FLAG) {
    LOG(debug) << "END_FLAG active for DP " << pid;
  }
  return 0;
}	
	
int HMPIDDCSProcessor::subStringToInt(std::string inputString, std::size_t startIndex)
{ 
  	char stringPos = inputString.at(startIndex);
	int charInt = ((int)stringPos) - ((int)'0');
	if(charInt < 10 && charInt >= 0) return charInt;
  	else return -1;
}	
	
}	// end namespace




