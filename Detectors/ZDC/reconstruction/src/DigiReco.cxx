#include <TMath.h>
#include "FairLogger.h"
#include "ZDCReconstruction/DigiReco.h"
namespace o2
{
namespace zdc
{
  
void DigiReco::init(){
  // Load configuration parameters
  auto& sopt = ZDCSimParam::Instance();
  mIsContinuous = sopt.continuous;
  mNBCAHead = mIsContinuous ? sopt.nBCAheadCont : sopt.nBCAheadTrig;

  if (!mModuleConfig) {
    LOG(FATAL) << "Missing ModuleConfig configuration object";
    return;
  }

  // Prepare tapered sinc function
  // tsc/TSN =3.75 (~ 4) and TSL*TSN*sqrt(2)/tsc >> 1 (n. of sigma)
  const Double_t tsc = 750;
  Int_t n = TSL * TSN;
  for (Int_t tsi = 0; tsi <= n; tsi++) {
    Double_t arg1 = TMath::Pi() * Double_t(tsi) / Double_t(TSN);
    Double_t fs = 1;
    if (arg1 != 0)
      fs = TMath::Sin(arg1) / arg1;
    Double_t arg2 = Double_t(tsi) / tsc;
    Double_t fg = TMath::Exp(-arg2 * arg2);
    mTS[n + tsi] = fs * fg;
    mTS[n - tsi] = mTS[n + tsi]; // Function is even
  }

  // Open debug file
  mDbg = new TFile("ZDCReco.root", "recreate");
  mTDbg = new TTree("zdcr", "ZDCReco");
  mTDbg->Branch("zdcr", "RecEvent", &mRec);
  // Update reconstruction parameters
  //auto& ropt=RecoParamZDC::Instance();
  o2::zdc::RecoParamZDC& ropt = const_cast<o2::zdc::RecoParamZDC&>(RecoParamZDC::Instance());

  // Fill maps
  for (Int_t itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    // If the reconstruction parameters were not manually set
    if (ropt.tmod[itdc] < 0 || ropt.tch[itdc] < 0) {
      Int_t isig = TDCSignal[itdc];
      for (Int_t im = 0; im < NModules; im++) {
        for (UInt_t ic = 0; ic < NChPerModule; ic++) {
          if (mModuleConfig->modules[im].channelID[ic] == isig && mModuleConfig->modules[im].readChannel[ic]) {
            //ropt.updateFromString(TString::Format("RecoParamZDC.tmod[%d]=%d;",itdc,im));
            //ropt.updateFromString(TString::Format("RecoParamZDC.tch[%d]=%d;",itdc,ic));
            ropt.tmod[itdc] = im;
            ropt.tch[itdc] = ic;
            goto next_itdc;
          }
        }
      }
    }
  next_itdc:;
    LOG(INFO) << "TDC " << itdc << "(" << ChannelNames[TDCSignal[itdc]] << ")" << " mod " << ropt.tmod[itdc] << " ch " << ropt.tch[itdc];
  }

  // Fill maps channel maps for integration
  for (Int_t ich = 0; ich < NChannels; ich++) {
    // If the reconstruction parameters were not manually set
    if (ropt.amod[ich] < 0 || ropt.ach[ich] < 0) {
      for (Int_t im = 0; im < NModules; im++) {
        for (UInt_t ic = 0; ic < NChPerModule; ic++) {
          if (mModuleConfig->modules[im].channelID[ic] == ich && mModuleConfig->modules[im].readChannel[ic]) {
            ropt.amod[ich] = im;
            ropt.ach[ich] = ic;
            goto next_ich;
          }
        }
      }
    }
  next_ich:;
    LOG(INFO) << "ADC " << ich << "(" << ChannelNames[ich] << ") mod " << ropt.amod[ich] << " ch " << ropt.ach[ich];
  }

  // Integration ranges
  for (Int_t ich = 0; ich < NChannels; ich++) {
    // If the reconstruction parameters were not manually set
    if (ropt.beg_int[ich] < 0 || ropt.end_int[ich] < 0) {
      if (!mIntParam) {
        LOG(ERROR) << "Integration for signal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.beg_int[ich] = mIntParam->beg_int[ich];
        ropt.end_int[ich] = mIntParam->end_int[ich];
      }
    }
    if (ropt.beg_ped_int[ich] < 0 || ropt.end_ped_int[ich] < 0) {
      if (!mIntParam) {
        LOG(ERROR) << "Integration for pedestal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.beg_ped_int[ich] = mIntParam->beg_ped_int[ich];
        ropt.end_ped_int[ich] = mIntParam->end_ped_int[ich];
      }
    }
    LOG(INFO) << ChannelNames[ich] << " integration: signal=[" << ropt.beg_int[ich] << "-" << ropt.end_int[ich] << "] pedestal=[" << ropt.beg_ped_int[ich] << "-" << ropt.end_ped_int[ich] <<"]";
  }
}

int DigiReco::process(const std::vector<o2::zdc::OrbitData> *orbitdata, const std::vector<o2::zdc::BCData> *bcdata, std::vector<o2::zdc::ChannelData> *chdata){
  // We assume that vectors contain data from a full time frame
  mOrbitData=orbitdata;
  mBCData=bcdata;
  mChData=chdata;

  const std::vector<o2::zdc::OrbitData> &OrbitData=*orbitdata;
  const std::vector<o2::zdc::BCData> &BCData=*bcdata;
  const std::vector<o2::zdc::ChannelData> &ChData=*chdata;
		  
  // Initialization of lookup structure for pedestals
  mOrbit.clear();
  int norb=OrbitData.size();
  for(Int_t iorb=0; iorb<norb; iorb++){
    mOrbit[OrbitData[iorb].ir.orbit]=iorb;
    LOG(INFO) << "Entry " << iorb << " for orbit " << OrbitData[iorb].ir.orbit;
  }
  mNBC=BCData.size();
  mReco.clear();
  mReco.reserve(mNBC);
  // Initialization of reco structure
  for(Int_t ibc=0; ibc<mNBC; ibc++){
    mReco[ibc].ir=BCData[ibc].ir;
  }
  
  // Probably this is not necessary
//   for(Int_t itdc=0; itdc<NTDCChannels; itdc++){
//     mReco.pattern[itdc]=0;
//     for(Int_t itb=0; itb<NTimeBinsPerBC; itb++){
//       mReco.fired[itdc][itb]=0;
//     }
//     for(Int_t isb=0; isb<mNSB; isb++){
//       mReco.inter[itdc][isb]=0;
//     }
//   }

  for(Int_t ibc=0; ibc<mNBC; ibc++){
    mReco[ibc].ir=BCData[ibc].ir;
  }

  // Find consecutive bunch crossings and perform interpolation
  Int_t seq_beg = 0;
  Int_t seq_end = 0;
  LOG(INFO) << "Reconstruction for " << mNBC << " bunch crossings";
  for(Int_t ibc=0; ibc<mNBC; ibc++){
    auto &ir=BCData[seq_end].ir;
    auto bcd=BCData[ibc].ir.differenceInBC(ir);
    if(bcd<0){
      LOG(FATAL) << "Orbit number is not increasing " <<  BCData[seq_end].ir.orbit << "." << BCData[seq_end].ir.bc << " followed by " << BCData[ibc].ir.orbit << "." << BCData[ibc].ir.bc;
      return __LINE__;
    } else if(bcd>1){
      // Detected a gap
      reconstruct(seq_beg, seq_end);
      seq_beg=ibc;
      seq_end=ibc;
    }else if(ibc==(mNBC-1)){
      // Last bunch
      seq_end=ibc;
      reconstruct(seq_beg, seq_end);
      seq_beg=mNBC;
      seq_end=mNBC;
    }else{
      // Look for another bunch
      seq_end=ibc;
    }
  }

/*

std::map<char,int>::iterator it;

  mymap['a']=50;
  mymap['b']=100;
  mymap['c']=150;
  mymap['d']=200;

  it = mymap.find('b');
  if (it != mymap.end())
    mymap.erase (it);
  */
  return 0;
}

int DigiReco::reconstruct(int seq_beg, int seq_end){
  // Process consecutive BCs
  if (seq_beg == seq_end) {
    LOG(INFO) << "Lonely bunch " << mReco[seq_beg].ir.orbit << "." << mReco[seq_beg].ir.bc;
    return 0;
  }
  LOG(INFO) << "Processing " << mReco[seq_beg].ir.orbit << "." << mReco[seq_beg].ir.bc << " - " << mReco[seq_end].ir.orbit << "." << mReco[seq_end].ir.bc;
  auto& ropt = ZDCRecoParam::Instance();
  /*
  // Apply differential discrimination with triple condition
  for (Int_t itdc = 0; itdc < NTDCChannels; itdc++) {
    Int_t im = ropt.tmod[itdc];
    Int_t ic = ropt.tch[itdc];
    // Check if the TDC channel is connected
    if (im >= 0 && ic >= 0) {
      // Check if channel has valid data for consecutive bunches in current bunch range
      // N.B. there are events recorded from ibeg-iend but we are not sure if it is the
      // case for every TDC channel
      int istart = -1, istop = -1;
      // Loop allows for gaps in the data sequence for each TDC channel
      for (int ibun = ibeg; ibun <= iend; ibun++) {
        auto& ch = mData[ibun].data[im][ic];
        if (ch.f.fixed_0 == Id_w0 && ch.f.fixed_1 == Id_w1 && ch.f.fixed_2 == Id_w2) {
          if (istart < 0) {
            istart = ibun;
          }
          istop = ibun;
        } else {
          // A gap is detected gap
          if (istart >= 0 && (istop - istart) > 0) {
            // Need data for at least two consecutive bunch crossings
            processTrigger(itdc, istart, istop);
          }
          istart = -1;
          istop = -1;
        }
      }
      // Check if there are consecutive bunch crossings at the end of group
      if (istart >= 0 && (istop - istart) > 0) {
        processTrigger(itdc, istart, istop);
      }
    }
  }
  // Reconstruct integrated charges and fill output tree
  // TODO: compare average pedestal with estimation from current event
  // TODO: failover in case of discrepancy
  for (Int_t ibun = ibeg; ibun <= iend; ibun++) {
    mRec = mReco[ibun];
    for (Int_t itdc = 0; itdc < NTDCChannels; itdc++) {
      printf("%d %u.%u %d ", ibun, mReco[ibun].ir.orbit, mReco[ibun].ir.bc, itdc);
      for (Int_t isam = 0; isam < NTimeBinsPerBC; isam++) {
        printf("%d", mRec.fired[itdc][isam]);
      }
      printf("\n");
      for (Int_t i = 0; i < MaxTDCValues; i++) {
        mRec.TdcChannels[itdc][i] = kMinShort;
        mRec.TdcAmplitudes[itdc][i] = -999;
      }
      Int_t i = 0;
      mRec.pattern[itdc] = 0;
      for (int16_t val : mReco[ibun].tdcChannels[itdc]) {
        LOG(INFO) << "TdcChannels[" << itdc << "][" << i << "]=" << val;
        mRec.TdcChannels[itdc][i] = val;
        // There is a TDC value in the search zone around main-main position
        if (std::abs(mRec.TdcChannels[itdc][i]) < ropt.tdc_search[itdc]) {
          mRec.pattern[itdc] = 1;
        }
        i++;
      }
      i = 0;
      for (float val : mReco[ibun].tdcAmplitudes[itdc]) {
        //mRec.tdcAmplitudes[itdc].push_back(val);
        //LOG(INFO) << itdc << " valt=" << val;
        LOG(INFO) << "TdcAmplitudes[" << itdc << "][" << i << "]=" << val;
        mRec.TdcAmplitudes[itdc][i] = val;
        i++;
      }
    }
    printf("%d PATTERN: ", ibun);
    for (Int_t itdc = 0; itdc < NTDCChannels; itdc++) {
      printf("%d", mRec.pattern[itdc]);
    }
    printf("\n");

    // Check if coincidence of common PM and sum of towers is satisfied
    bool fired[NChannels] = {0};
    // Side A
    if ((mRec.pattern[TDCZNAC] || ropt.bitset[TDCZNAC]) && (mRec.pattern[TDCZNAS] || ropt.bitset[TDCZNAS])) {
      for (Int_t ich = IdZNAC; ich <= IdZNASum; ich++) {
        fired[ich] = true;
      }
    }
    if ((mRec.pattern[TDCZPAC] || ropt.bitset[TDCZPAC]) && (mRec.pattern[TDCZPAS] || ropt.bitset[TDCZPAS])) {
      for (Int_t ich = IdZPAC; ich <= IdZPASum; ich++) {
        fired[ich] = true;
      }
    }
    // ZEM1 and ZEM2 are not in coincidence
    fired[IdZEM1] = mRec.pattern[TDCZEM1];
    fired[IdZEM2] = mRec.pattern[TDCZEM2];
    // Side C
    if ((mRec.pattern[TDCZNCC] || ropt.bitset[TDCZNCC]) && (mRec.pattern[TDCZNCS] || ropt.bitset[TDCZNCS])) {
      for (Int_t ich = IdZNCC; ich <= IdZNCSum; ich++) {
        fired[ich] = true;
      }
    }
    if ((mRec.pattern[TDCZPCC] || ropt.bitset[TDCZPCC]) && (mRec.pattern[TDCZPCS] || ropt.bitset[TDCZPCS])) {
      for (Int_t ich = IdZPCC; ich <= IdZPCSum; ich++) {
        fired[ich] = true;
      }
    }

    // Access samples from raw data
    for (Int_t i = 0; i < NChannelsZEM; i++) {
      mRec.energyZEM[i] = -999;
    }
    for (Int_t i = 0; i < NChannelsZN; i++) {
      mRec.energyZNA[i] = -999;
    }
    for (Int_t i = 0; i < NChannelsZN; i++) {
      mRec.energyZNC[i] = -999;
    }
    for (Int_t i = 0; i < NChannelsZP; i++) {
      mRec.energyZPA[i] = -999;
    }
    for (Int_t i = 0; i < NChannelsZP; i++) {
      mRec.energyZPC[i] = -999;
    }
    auto& data = mData[ibun];
    printf("%d FIRED: ", ibun);
    for (Int_t ich = 0; ich < NChannels; ich++) {
      printf("%d", fired[ich]);
    }
    printf("\n");
    for (Int_t ich = 0; ich < NChannels; ich++) {
      // Check if the corresponding TDC is fired
      if (fired[ich]) {
        // Check if channels are present in payload
        Int_t im = ropt.amod[ich];
        Int_t ic = ropt.ach[ich];
        // Check if the ADC channel is connected
        if (im >= 0 && ic >= 0) {
          // Check if the ADC has payload
          auto& ch = data.data[im][ic];
          if (ch.f.fixed_0 == Id_w0 && ch.f.fixed_1 == Id_w1 && ch.f.fixed_2 == Id_w2) {
            float sum = 0;
            for (Int_t is = ropt.beg_int[ich]; is <= ropt.end_int[ich]; is++) {
              // TODO: fallback if offset is missing
              // TODO: fallback if channel has pile-up
              sum += (mOffset[im][ic] - float(data.s[im][ic][is]));
            }
            printf("CH %d %s: %f\n", ich, ChannelNames[ich].data(), sum);
            if (ich == IdZNAC) {
              mRec.energyZNA[0] = sum;
            }
            if (ich == IdZNA1) {
              mRec.energyZNA[1] = sum;
            }
            if (ich == IdZNA2) {
              mRec.energyZNA[2] = sum;
            }
            if (ich == IdZNA3) {
              mRec.energyZNA[3] = sum;
            }
            if (ich == IdZNA4) {
              mRec.energyZNA[4] = sum;
            }
            if (ich == IdZNASum) {
              mRec.energyZNA[5] = sum;
            }
            if (ich == IdZPAC) {
              mRec.energyZPA[0] = sum;
            }
            if (ich == IdZPA1) {
              mRec.energyZPA[1] = sum;
            }
            if (ich == IdZPA2) {
              mRec.energyZPA[2] = sum;
            }
            if (ich == IdZPA3) {
              mRec.energyZPA[3] = sum;
            }
            if (ich == IdZPA4) {
              mRec.energyZPA[4] = sum;
            }
            if (ich == IdZPASum) {
              mRec.energyZPA[5] = sum;
            }
            if (ich == IdZEM1) {
              mRec.energyZEM[0] = sum;
            }
            if (ich == IdZEM2) {
              mRec.energyZEM[1] = sum;
            }
            if (ich == IdZNCC) {
              mRec.energyZNC[0] = sum;
            }
            if (ich == IdZNC1) {
              mRec.energyZNC[1] = sum;
            }
            if (ich == IdZNC2) {
              mRec.energyZNC[2] = sum;
            }
            if (ich == IdZNC3) {
              mRec.energyZNC[3] = sum;
            }
            if (ich == IdZNC4) {
              mRec.energyZNC[4] = sum;
            }
            if (ich == IdZNCSum) {
              mRec.energyZNC[5] = sum;
            }
            if (ich == IdZPCC) {
              mRec.energyZPC[0] = sum;
            }
            if (ich == IdZPC1) {
              mRec.energyZPC[1] = sum;
            }
            if (ich == IdZPC2) {
              mRec.energyZPC[2] = sum;
            }
            if (ich == IdZPC3) {
              mRec.energyZPC[3] = sum;
            }
            if (ich == IdZPC4) {
              mRec.energyZPC[4] = sum;
            }
            if (ich == IdZPCSum) {
              mRec.energyZPC[5] = sum;
            }
          }
        }
      }
    }
    // TODO: energy calibration
    mTDbg->Fill();
  }
  */
}

} // namespace zdc
} // namespace o2
