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

#ifndef HMPIDDCSPROCESSOR_H
#define HMPIDDCSPROCESSOR_H

// calibration/HMPIDCalibration header-files:
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




namespace o2::hmpid
{

  using namespace std::literals; // Should not be in h.file !! move constexpr to cxx?
	
	
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using DPID = o2::dcs::DataPointIdentifier;
  //using DPVAL = o2::dcs::DataPointValue; 
  using DPCOM = o2::dcs::DataPointCompositeObject;		
  class HMPIDDCSProcessor{


	public:	

		struct TimeRange {
			uint64_t first{};
			uint64_t last{};		
		};
  
                using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
		   
		HMPIDDCSProcessor() = default;
		~HMPIDDCSProcessor() = default;
		
	
		// Process Datapoints:  ===================================================================================================
		//void init(const std::vector<DPID>& pids);

		// process span of DPs:
		// process DPs, fetch IDs and call processIR or processHMPID
		void process(const gsl::span<const DPCOM> dps); 		

		void processIR(DPCOM dp);    // if it mathces IR_ID = "HMP_DET/HMP_INFR"
		void processHMPID(DPCOM dp); // if it matches HMPID_ID = "HMP_DET", but not IR_ID

		// Fill entries of DPs==================================================
		void fillChamberPressures(const DPCOM& dpcom);			// fill element[0-6] in chamber-pressure vector

		void fillEnvironmentPressure(const DPCOM& dpcom);		// fill environment-pressure vector 
	
		// HV in each chamber_section = 7*3 --> will result in Q_thre  
		void fillHV(const DPCOM& dpcom); 				// fill element[0-20] in HV vector
		
		// Temp in (T1) and out (T2), in each chamber_radiator = 7*3 :  
		void fill_InTemperature(const DPCOM& dpcom);			// fill element[0-20] in tempIn vector
		void fill_OutTemperature(const DPCOM& dpcom); 			// fill element[0-20] in tempOut vector


		 // finalize DPs, after run is finished  ===================================================================================================
		 void finalizeEnvPressure(); 
		 void finalizeHV_Entry(Int_t iCh,Int_t iSec); 
		 void finalizeChPressureEntry(Int_t iCh); 
		 void finalizeTempOutEntry(Int_t iCh,Int_t iRad); 
		 void finalizeTempInEntry(Int_t iCh,Int_t iRad); 
		 void finalize(); 


		// procTrans	 ===================================================================================================
		double DefaultEMean();					//just set a refractive index for C6F14 at ephot=6.675 eV @ T=25 C 				
		double ProcTrans();
	
		// help-functions ================================================================================			
	        void setStartValidity(long t) { mStartValidity = t; }
	        void useVerboseMode() { mVerbose = true; }
	  
           	// convert char or substring to int (i.e. fetch int in string/char) 
		int subStringToInt(std::string inputString, std::size_t startIndex);
	  	uint64_t processFlags(const uint64_t flags, const char* pid);
		
	  
	 	// DCS-CCDB methods and members Used in HMPIDDCSDataProcessor===============================================================================
		CcdbObjectInfo& getccdbREF_INDEXsInfo() { return mccdbREF_INDEX_Info; }
         	//CcdbObjectInfo& getccdbREF_INDEXsInfo() { return mccdbREF_INDEX_Info; }
	
		// for calculating refractive index: 
		std::vector<TF1>& getRefIndexObj()  { return arNmean; } // mRefIndex
		//TF1 arNmean[43]; /// 21* Tin and 21*Tout (1 per radiator, 3 radiators per chambers)
				 // + 1 for ePhotMean (mean photon energy) 		


	
   		CcdbObjectInfo& getHmpidChargeCutInfo() { return mccdbCHARGE_CUT_Info; }
    		//CcdbObjectInfo& getHmpidChargeCutInfo() { return mccdbCHARGE_CUT_Info; }
	
		std::vector<TF1>& getChargeCutObj() { return arQthre; }// mChargeCut
		// Charge Threshold: 
		// TF1 arQthre[42];  //42 Qthre=f(time) one per sector
	
	 	// get methods for time-ranges ===============================================================================
	  
	  	const auto& getTimeQThresh() const { return mTimeQThresh;}
		const auto& getTimeArNmean() const { return mTimeArNmean;}
	  
	 private:
		// DCS-CCDB ====================================================================================================

		std::unordered_map<DPID, bool> mPids;

	 	long mFirstTime;         // time when a CCDB object was stored first
	  	long mStartValidity = 0; // TF index for processing, used to store CCDB object
	  	bool mFirstTimeSet = false; 	
	
		bool mVerbose = false;		

		CcdbObjectInfo mccdbREF_INDEX_Info;
		std::vector<TF1> mRefIndex[43];

		CcdbObjectInfo mccdbCHARGE_CUT_Info;
		std::vector<TF1> mChargeCut[42];

// private variables  ===================================================================================	
		Double_t xP,yP;
		TF2 *thr = new TF2("RthrCH4"  ,"3*10^(3.01e-3*x-4.72)+170745848*exp(-y*0.0162012)"             ,2000,3000,900,1200); 
	
	
		// for calculating refractive index: 
		std::vector<TF1> arNmean[43];   // 21* Tin and 21*Tout (1 per radiator, 3 radiators per chambers)
				 		 // + 1 for ePhotMean (mean photon energy) 
	
	
		// Charge Threshold: 
		std::vector<TF1> arQthre[42];	 //42 Qthre=f(time) one per sector

	
		// env pressure 
		Int_t cntEnvPressure=0;  	 // cnt Environment-pressure entries
		std::vector<DPCOM> pEnv; 	 // environment-pressure vector

	
		// ch pressure 
		std::vector<DPCOM> pChamber[7];  //  chamber-pressure vector [0..6]
		Int_t cntChPressure = 0; 	 // cnt chamber-pressure entries in element iCh[0..6] 


		Int_t cntTin = 0, cntTOut = 0; 	  // cnt tempereature entries in element i[0..20]; i = 3*iCh+iSec
		std::vector<DPCOM> tempIn[21];    //  tempIn vector [0..20]
		std::vector<DPCOM> tempOut[21];   //  tempOut vector [0..20]		

	
		// HV 
		std::vector<DPCOM> dpcomHV[42];   //  HV vector [0..41]; 7 chambers * 6 sectors
		Int_t cntHV=0;			  // cnt HV entries in element i[0..41];  i = iCh*6 + iSec
     		



		// Timestamps and TimeRanges ======================================================================================
		// timestamps of last and first HV-datapoint entry in 
		// 1d-array of vectors of HV
		uint64_t hvFirstTime, hvLastTime; 

		uint64_t pChFirstTime, pChLastTime; 	// chamberprssure timestamps
		uint64_t pEnvFirstTime, pEnvLastTime;   // envPressure timestamps
		
		
		TimeRange mTimeEMean; // Timerange for mean photon energy(procTrans) 

		TimeRange mTimeQThresh; // Timerange for QThresh (ChargeCut)
					// min, max of timestamps {envP, chP, HV}


 		// indexes for getting chamber-numbers etc ==================================================================================================
	        
	 	// Chamber Pressures
		//HMP_DET/HMP_MP0/HMP_MP0_GAS/HMP_MP0_GAS_PMWPC.actual.value 
		std::size_t startI_chamberPressure = 14;
		
		// High Voltage
		//HMP_DET/HMP_MP0/HMP_MP0_PW/HMP_MP0_SEC0/HMP_MP0_SEC0_HV.actual.vMon
		std::size_t startI_chamberHV = 14; 
		std::size_t startI_sectorHV = 38;	//HMP_DET/HMP_MP0/HMP_MP0_PW/HMP_MP0_SEC0
	
		// Temperatures
		//HMP_DET/HMP_MP0/HMP_MP0_LIQ_LOOP.actual.sensors.Rad0In_Temp 
		std::size_t startI_chamberTemp = 14; //HMP_DET/HMP_MP0
		std::size_t startI_radiatorTemp = 51; //HMP_DET/HMP_MP0/HMP_MP0_LIQ_LOOP.actual.sensors.Rad0 
 

		uint64_t tempFirstTime, tempLastTime; 
		uint64_t tOutFirstTime, tOutLastTime; 
		uint64_t procTransFirstTime, procTransLastTime; 

		TimeRange mTimeArNmean;	// Timerange for arNmean (RefIndex)
					// min, max of timestamps {Tin, Tout, ProcTrans}
				
	  
	  	// procTrans variables ======================================================================
	  
		double  sEnergProb=0, sProb=0; 				// energy probaility, probability
     		// Double_t tRefCR5 = 19. ;                             // mean temperature of CR5 where the system is in place
     		double  eMean = 0;					// initialize eMean (Photon energy mean) to 0
     		int indexOfIR = 69; 
		
		double  aCorrFactor[30] = {0.937575212,0.93805688,0.938527113,0.938986068,0.939433897,0.939870746,0.940296755,0.94071206,0.941116795,0.941511085,0.941895054,0.942268821,0.942632502,
					0.942986208,0.943330047,0.943664126,0.943988544,0.944303401,0.944608794,0.944904814,0.945191552,0.945469097,0.945737533,0.945996945,0.946247412,
					0.946489015,0.94672183,0.946945933,0.947161396,0.947368291}; 
    
    		double nm2eV;  // conversion factor, nanometer to eV 
		double photEn; // photon energy 
    				
	
		// wavelenght // phototube current for argon reference  //  phototube current for argon cell 
	 	// phototube current for freon reference // phototube current for freon cell
		DPCOM 	    	     dpWaveLen,   dpArgonRef,   dpArgonCell,   dpFreonRef,   dpFreonCell;	// DPCOM names
	 	double       		lambda,    aRefArgon,    aCellArgon,    aRefFreon,    aCellFreon;	// double (DVAL) names
		std::vector<DPCOM> waveLen[30], argonRef[30], argonCell[30], freonRef[30], freonCell[30]; 	// vectors

	
	   	double aTransRad, aConvFactor; // evaluate 15 mm of thickness C6F14 Trans
		double  aTransSiO2;	       // evaluate 0.5 mm of thickness SiO2 Trans
		double  aTransGap;	       // evaluate 80 cm of thickness Gap (low density CH4) transparency 
		double  aCsIQE; 	       // evaluate CsI quantum efficiency
		double  aTotConvolution;      // evaluate total convolution of all material optical properties	
	  
	  
		// constExpression string-literals to assign DPs to the correct method: =================================================================
		
		// check if IR or other HMPID specifciation
		static constexpr auto HMPID_ID{"HMP_DET"sv};
		static constexpr auto IR_ID{"HMP_DET/HMP_INFR"sv};

		// HMPID-temp, HV, pressure IDs (HMPID_ID{"HMP_DET"sv};)
		static constexpr auto TEMP_OUT_ID{"Out_Temp"sv};
		static constexpr auto TEMP_IN_ID{"In_Temp"sv};
		static constexpr auto HV_ID{"vMon"sv};
		static constexpr auto ENV_PRESS_ID{"PENV.actual.value"sv};
		static constexpr auto CH_PRESS_ID{"PMWPC.actual.value"sv};

		// HMPID-IR IDs (IR_ID{"HMP_DET/HMP_INFR"sv})
		static constexpr auto WAVE_LEN_ID{"waveLenght"sv}; // 0-9 
		static constexpr auto REF_ID{"Reference"sv}; // argonReference and freonRef
		static constexpr auto ARGON_CELL_ID{"argonCell"sv}; // argon Cell reference 
		static constexpr auto FREON_CELL_ID{"c6f14Cell"sv}; // fron Cell Reference

		static constexpr auto ARGON_REF_ID{"argonReference"sv}; // argonReference 
		static constexpr auto FREON_REF_ID{"c6f14Reference"sv}; // freonReference
	
	
	ClassDefNV(HMPIDDCSProcessor,0);
};// end class 
} // end o2::hmpid
#endif 
