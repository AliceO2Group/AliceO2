#include "HMPIDCalibration/HMPIDDCSProcessor.h"


#include "HMPIDCalibration/HMPIDDCSTime.h"

#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"


#include <TF1.h>                  //Process()
#include <TF2.h>                  //Process()
#include <TGraph.h>               //Process()


#include <memory>
#include <deque> 
#include <gsl/gsl> 

#include "Framework/Logger.h"
#include "DetectorsDCS/DataPointCompositeObject.h" 
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h" 
#include "CCDB/CcdbObjectInfo.h" 
#include "CCDB/CcdbApi.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h" // o2::calibration::dcs
//using DeliveryType = o2::dcs::DeliveryType;
//using DPID = o2::dcs::DataPointIdentifier;
//using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;


using namespace o2::dcs;



namespace o2::hmpid {
	
void HMPIDDCSProcessor::process(const gsl::span<const DPCOM> dps)
{
  if (dps.size() == 0) {
    return;
  }
  

  for (const auto& dp : dps) {
    const std::string_view alias(dp.id.get_alias());
    const auto detector_id = alias.substr(0, 7);
    const auto ir_id = alias.substr(0,16);

    // check if given dp is from HMPID
    	    // check first if IR:
	    if (ir_id == IR_ID) {
		processIR(dp); 
	    } // if not IR, check if other DP in HMPID (pressure, temp, HV): 
	    else if (detector_id==HMPID_ID){
		processHMPID(dp); 
	    }  else  LOG(debug) << "Unknown data point: {}"<< alias;
  }
}	

// if the string of the dp contains the HMPID-specifier "HMP_DET",
// but not the IR-specifier "HMP_DET/HMP_INFR" : 
void HMPIDDCSProcessor::processHMPID(DPCOM dp)
{
   
  const std::string alias(dp.id.get_alias());

 if ( alias.substr(alias.length()-7) == TEMP_IN_ID ) {
      LOG(info) << "Temperature_in DP: {}"<< alias;
      fillTemperature(dp, true); 
    } else if (alias.substr(alias.length()-8) == TEMP_OUT_ID) {
      LOG(info) << "Temperature_out DP: {}"<< alias;
      fillTemperature(dp, false);
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
    const auto specify_id = alias.substr(alias.length()-9);

	auto numIR = subStringToInt(alias, indexOfIR, indexOfIR);
	if(specify_id == WAVE_LEN_ID) {
		waveLen[numIR].push_back(dp);
	}  else if(specify_id == ARGON_CELL_ID) { 
		argonCell[numIR].push_back(dp);
	}  else if(specify_id == FREON_CELL_ID) { 
		freonCell[numIR].push_back(dp);
	}
	else if(specify_id == REF_ID) { 
		if( alias.substr(alias.length()-14) ==  ARGON_REF_ID){
			argonRef[numIR].push_back(dp);
		} else if( alias.substr(alias.length()-14) ==  FREON_REF_ID){
			freonRef[numIR].push_back(dp);
		}      LOG(debug) << "Unknown data point: {}"<< alias;
	} else LOG(debug) << "Datapoint not found: {}"<< alias;
}

	
double HMPIDDCSProcessor::ProcTrans()
{   
    for(int i=0; i<30; i++) 
      {
    		
            // evaluate wavelenght 
            //("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght",i)); 
            if(waveLen[i].size() == 0) // if there is no entries 
            { 
		LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght  -----> Default E mean used!!!!!",i);
		return DefaultEMean();	// will break this entry in foor loop
            }  
	    //  pVal=(AliDCSValue*)pWaveLenght->At(0); // get first element, (i.e. pointer to TObject at index 0)
            	dpWaveLen =  (waveLen[i]).at(0);   
	        lambda = o2::dcs::getValue<double>(dpWaveLen); // Double_t lambda = pVal->GetFloat();
            if(lambda<150. || lambda>230.)
            { 
                LOG(debug) << Form("Wrong value for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); // will break this entry in foor loop
            } 
	    
            //find photon energy E in eV from radiation wavelength λ in nm
	    nm2eV = 1239.842609;	// 1239.842609 from nm to eV 
            photEn = nm2eV/lambda;     // photon energy
	    
		    
		    
	    //Ali: static Double_t  https://github.com/alisw/AliRoot/blob/222b69b9f193abd33c5e7b71e91e21ae1816bcc5/HMPID/HMPIDbase/AliHMPIDParam.h#L87-L88
	    //O2: static double https://github.com/AliceO2Group/AliceO2/blob/3603ce32b2cddccfd94deb70b70628f3ff0846cd/Detectors/HMPID/base/include/HMPIDBase/Param.h#L136
            if(photEn<o2::hmpid::Param::ePhotMin() || photEn>o2::hmpid::Param::ePhotMax()) continue; // if photon energy is out of range
            
            // evaluate phototube current for argon reference
            if(argonRef[i].size() == 0)
            { 
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonReference  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); // to be checked
            } 
      	
            dpArgonRef =  (argonRef[i]).at(0); // pVal=(AliDCSValue*)pArgonRef->At(0);    
            aRefArgon = o2::dcs::getValue<double>(dpArgonRef);// Double_t aRefArgon = pVal->GetFloat();
        
            // evaluate phototube current for argon cell
            if(argonCell[i].size() == 0)
            { 
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonCell  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); // to be checked
            } 
	    
            dpArgonCell  =  (argonCell[i]).at(0); // pVal=(AliDCSValue*)pArgonRef->At(0);    
            aCellArgon = o2::dcs::getValue<double>(dpArgonCell);// Double_t aCellArgon = pVal->GetFloat(); 
            
        
            //evaluate phototube current for freon reference
            if(freonRef[i].size() == 0)
            { 
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Reference  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); // to be checked
            } 
            dpFreonRef  =  (freonRef[i]).at(0); //pVal=(AliDCSValue*)pFreonRef->At(0);  
            aRefFreon = o2::dcs::getValue<double>(dpFreonRef);//   Double_t aRefFreon = pVal->GetFloat(); 
      
			        
            //evaluate phototube current for freon cell
            if(freonCell[i].size() == 0)
            {
                LOG(debug) << Form("No Data Point values for HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Cell  -----> Default E mean used!!!!!",i);
                return DefaultEMean(); // to be checked
            }
             dpFreonCell =  (freonCell[i]).at(0); // pVal=(AliDCSValue*)pFreonCell->At(0);
             aCellFreon = o2::dcs::getValue<double>(dpFreonCell);//   Double_t aCellFreon = pVal->GetFloat();
           
         
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
              aTransRad  = TMath::Power((aCellFreon/aRefFreon)/(aCellArgon/aRefArgon)*aCorrFactor[i],aConvFactor); // Double_t = pow(Double_t,Double_t)
            }
            else
            {
              return DefaultEMean();
            }
            
	    //https://github.com/AliceO2Group/AliceO2/blob/3603ce32b2cddccfd94deb70b70628f3ff0846cd/Detectors/HMPID/base/include/HMPIDBase/Param.h#L146-L148
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
	    const std::vector<DPCOM> arr[5] = {waveLen[i], argonRef[i], argonCell[i], freonRef[i], freonCell[i]};
		
	    auto minTime = HMPIDDCSTime::getMinTimeArr(arr);
	    auto maxTime = HMPIDDCSTime::getMaxTimeArr(arr);
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

/*	
// A: initialize pTin in  finalizeTempOutEntry function,
void HMPIDTestComp::initTempArr()
{	for(auto iCh = 0;iCh<7;iCh++){                   
		for(auto iRad = 0;iRad<3;iRad++){
			pTin[3*iCh+iRad]  = new TF1(Form("Tin%i%i" ,iCh,iRad),"[0]+[1]*x",0,1);//,fStartTime,fEndTime);
			pTout[3*iCh+iRad] = new TF1(Form("Tout%i%i",iCh,iRad),"[0]+[1]*x",0,1);//fStartTime,fEndTime);
		}
	  }
} */ 
 

void HMPIDDCSProcessor::finalizeEnvPressure() // after run is finished, 
{ 
	if(pEnv.size() != 0){ // if environment-pressure has entries 
		TGraph *pGrPenv=new TGraph; //  cntEnvPressure=0;

		auto minTime = HMPIDDCSTime::getMinTime(pEnv);
		if(minTime < mTimeQThresh.first) mTimeQThresh.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(pEnv);
		if(maxTime > mTimeQThresh.last) mTimeQThresh.last = maxTime;

		for(DPCOM dp : pEnv){
			pGrPenv->SetPoint(cntEnvPressure++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		if(cntEnvPressure==1) { 
		 pGrPenv->GetPoint(0,xP,yP);       
		 new TF1("Penv",Form("%f",yP), minTime,maxTime);//fStartTime,fEndTime);
		} else {
		  pGrPenv->Fit(new TF1("Penv","1000+x*[0]",minTime,maxTime),"Q");
		}  delete pGrPenv;
	} else LOG(debug) << Form("No entries in environment pressure");
}


void HMPIDDCSProcessor::finalizeHV_Entry(Int_t iCh,Int_t iSec) // after run is finished, 
{ 
	// check if given element has entries
	if(dpcomHV[3*iCh+iSec].size() != 0){
		TGraph *pGrHV=new TGraph; cntHV=0; 

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
		}  delete pGrHV;
	} else LOG(debug) << Form("No entries in HV for chamber %i, section %i",iCh,iSec);
}

void HMPIDDCSProcessor::finalizeChPressureEntry(Int_t iCh) // after run is finished, 
{ 	
	if(pChamber[iCh].size() != 0){
		TGraph *pGrP=new TGraph; cntChPressure=0;

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
		}  delete pGrP;
	} else  LOG(debug) << Form("No entries in chamber-pressure for chamber %i",iCh); 	
}


			
void HMPIDDCSProcessor::finalizeTempOutEntry(Int_t iCh,Int_t iRad) // after run is finished, 
{ 
	if(tempOut[3*iCh+iRad].size() != 0){

		auto minTime = HMPIDDCSTime::getMinTime(tempOut[3*iCh+iRad]);
		if(minTime < mTimeArNmean.first) mTimeArNmean.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(tempOut[3*iCh+iRad]);
		if(maxTime > mTimeArNmean.last) mTimeArNmean.last = maxTime;

		TGraph *pGrTOut = new TGraph; cntTOut=0;
		for(DPCOM dp : tempOut[3*iCh+iRad]){
			pGrTOut->SetPoint(cntTOut++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		// might need to initialize pTout[3*iCh+iRad] here 
		pTout[3*iCh+iRad] = new TF1(Form("Tout%i%i",iCh,iRad),"[0]+[1]*x",minTime,maxTime);
		if(cntTOut==1) { 
		 pGrTOut->GetPoint(0,xP,yP);
		 pTout[3*iCh+iRad]->SetParameter(0,yP);
		 pTout[3*iCh+iRad]->SetParameter(1,0);
		} else {
		pGrTOut->Fit(pTout[3*iCh+iRad],"Q");
		}  delete pGrTOut;
	} else  LOG(debug) << Form("No entries in temp-out for chamber %i, radiator %i",iCh,iRad); 

}

void HMPIDDCSProcessor::finalizeTempInEntry(Int_t iCh,Int_t iRad) // after run is finished, 
{ 
	if(tempIn[3*iCh+iRad].size() != 0){

		auto minTime = HMPIDDCSTime::getMinTime(tempIn[3*iCh+iRad]);
		if(minTime < mTimeArNmean.first) mTimeArNmean.first = minTime;
		auto maxTime = HMPIDDCSTime::getMaxTime(tempIn[3*iCh+iRad]);
		if(maxTime > mTimeArNmean.last) mTimeArNmean.last = maxTime;

		TGraph *pGrTIn = new TGraph; cntTin=0;
		for(DPCOM dp : tempIn[3*iCh+iRad]){
			pGrTIn->SetPoint(cntTin++,dp.data.get_epoch_time(),o2::dcs::getValue<double>(dp));
		}
		// might need to initialize pTin[3*iCh+iRad] here 
		pTin[3*iCh+iRad]  = new TF1(Form("Tin%i%i" ,iCh,iRad),"[0]+[1]*x",minTime,maxTime);
		if(cntTin==1) { 
		 pGrTIn->GetPoint(0,xP,yP);
		 pTin[3*iCh+iRad]->SetParameter(0,yP);
		 pTin[3*iCh+iRad]->SetParameter(1,0);
		} else {
		pGrTIn->Fit(pTin[3*iCh+iRad],"Q");
		}  delete pGrTIn;
		
	} else { LOG(debug) << Form("No entries in temp-in for chamber %i, radiator %i",iCh,iRad); }

}

void HMPIDDCSProcessor::finalize() // after run is finished, 
{ 
	for(Int_t iCh=0;iCh<7;iCh++){
		finalizeChPressureEntry(iCh);
		for(Int_t iRad=0;iRad<3;iRad++){
			finalizeTempInEntry(iCh,iRad);
			finalizeTempOutEntry(iCh,iRad); 
			// evaluate Mean Refractive Index
			// de-reference pTin/pTout since they are pointers
		      	arNmean[6*iCh+2*iRad] = *pTin[3*iCh+iRad]; //Tin =f(t)
      			arNmean[6*iCh+2*iRad+1] = *pTout[3*iCh+iRad]; //Tout=f(t)
		}

		for(Int_t iSec=0;iSec<6;iSec++){
			finalizeHV_Entry(iCh,iSec);	
			// evaluate Qthre
			hvFirstTime = HMPIDDCSTime::getMinTime(dpcomHV[6*iCh+iSec]);
			hvLastTime = HMPIDDCSTime::getMaxTime(dpcomHV[6*iCh+iSec]);
			// de-reference TF1 since it is a pointer
			arQthre[6*iCh+iSec] = *(new TF1(Form("HMP_QthreC%iS%i",iCh,iSec),Form("3*10^(3.01e-3*HV%i_%i - 4.72)+170745848*exp(-(P%i+Penv)*0.0162012)",iCh,iSec,iCh),hvFirstTime,hvLastTime)); 
		}
	}


	


	double eMean = ProcTrans();	 
	
	// startTimeTemp and endTimeTemp: min and max in 1d array of vectors of Tin/Tout
	uint64_t startTimeTemp = std::max(HMPIDDCSTime::getMinTimeArr(tempOut),HMPIDDCSTime::getMinTimeArr(tempIn));
	
	uint64_t endTimeTemp = std::min(HMPIDDCSTime::getMaxTimeArr(tempOut),HMPIDDCSTime::getMaxTimeArr(tempIn)); // ?? 
	// startTime is from temperature, but endTime should be from last entry in  ProcTrans()? 
        arNmean[42] = *(new TF1("HMP_PhotEmean",Form("%f",eMean),startTimeTemp,endTimeTemp));//fStartTime,fEndTime); //Photon energy mean



	
	 // prepare CCDB: 
	 std::map<std::string, std::string> md;
	 md["responsible"] = "NB!! CHANGE RESPONSIBLE";
	
		 // Refractive index (T_out, T_in, mean photon energy)
		 o2::calibration::Utils::prepareCCDBobjectInfo(arNmean, mccdbREF_INDEX_Info, "HMPID/Calib/RefIndex", md, mStartValidity, o2::calibration::Utils::INFINITE_TIME);

		 // charge threshold 
		 o2::calibration::Utils::prepareCCDBobjectInfo(arQthre,mccdbCHARGE_CUT_Info , "HMPID/Calib/ChargeCut", md, mStartValidity, o2::calibration::Utils::INFINITE_TIME);

}






void HMPIDDCSProcessor::fillChamberPressures(const DPCOM& dpcom)
{
	
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  
  
  if(type == DeliveryType::DPVAL_INT || type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
  {

 	// find chamber number:  
	auto chNum = subStringToInt(aliasStr, startI_chamberPressure, startI_chamberPressure);
	pChamber[chNum].push_back(dpcom);
	//mChamberPressure.fill(chamberNumber,aliasStr, time, value);
  } else LOG(debug)<< "Not correct specification for Environment-pressure DP: {}"<< aliasStr;
}



void HMPIDDCSProcessor::fillEnvironmentPressure(const DPCOM& dpcom) // A :better to pass string ? 
{	
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  

  if(type == DeliveryType::DPVAL_INT || type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
	{
		
	  	pEnv.push_back(dpcom); 
		//mEnvironmentPressure.fill(aliasStr, time, value);
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

  if(type == DeliveryType::DPVAL_INT || type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected
  { 
	
	auto chNum = subStringToInt(aliasStr, startI_chamberHV,  startI_chamberHV);
	auto secNum = subStringToInt(aliasStr,  startI_sectorHV,  startI_sectorHV);
	dpcomHV[6*chNum+secNum].push_back(dpcom);
  } else LOG(debug)<< "Not correct datatype for HV DP: {}"<< aliasStr;
	
}
	
// Temp in (T1) and out (T2), in each chamber_radiator = 7*6  
void HMPIDDCSProcessor::fillTemperature(const DPCOM& dpcom, bool in) // A :better to pass string ? 
{
  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  const std::string aliasStr(dpid.get_alias());  
	
  if(type == DeliveryType::DPVAL_INT || type == DeliveryType::DPVAL_DOUBLE) // check if datatype is as expected 
  {	
	
	auto chNum = subStringToInt(aliasStr,  startI_chamberTemp,  startI_chamberTemp);
	auto radNum = subStringToInt(aliasStr,  startI_radiatorTemp,  startI_radiatorTemp);

	
	if(in){
		tempIn[3*chNum+radNum].push_back( dpcom); 
		//mTemperature.fill(aliasStr,chamberNumber,radiatorNumber, time, value);

	} else{
		tempOut[3*chNum+radNum].push_back(dpcom); 
	} 
  } else LOG(debug) << "Not correct Data-type for Temperature DP: {}" << aliasStr;

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
	
  int HMPIDDCSProcessor::subStringToInt(std::string inputString, std::size_t startIndex, std::size_t endIndex)
  { 
  	// legg til sjekk om begge verdiene er det samme?
  	char stringPos = inputString.at(startIndex);
  	return ((int)stringPos) - ((int)'0');
 }	
	
}	// end namespace


      


