#ifdef __CINT__
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: ESDLinkDef.h 54829 2012-02-25 20:47:28Z morsch $ */

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
 
#pragma link C++ enum   AliESDEvent::ESDListIndex;

#pragma link C++ class std::map<std::pair<int, int>, int>;

#pragma link C++ class  AliESD+;
#pragma link C++ class  AliESDEvent+;
#pragma link C++ class  AliESDInputHandler+;
#pragma link C++ class  AliESDInputHandlerRP+;
#pragma link C++ class  AliESDRun+;
#pragma link C++ class  AliESDHeader+;

#pragma read \
  sourceClass="AliESDHeader" \
  targetClass="AliESDHeader" \
  source="TObjArray fIRBufferArray" \
  version="[10-13]"	\
  target="fIRBufferArray" \
  targetType="TObjArray" \
  code="{fIRBufferArray=onfile.fIRBufferArray; fIRBufferArray.SetOwner(kTRUE); onfile.fIRBufferArray.SetOwner(kFALSE);onfile.fIRBufferArray.Clear();}"

#pragma link C++ class  AliESDHLTDecision+;
#pragma link C++ class  AliESDZDC+;
#pragma link C++ class  AliESDCaloTrigger+;

#pragma read \
  sourceClass="AliESDCaloTrigger" \
  targetClass="AliESDCaloTrigger" \
  source="Int_t fNEntries; Int_t * fColumn; Int_t * fRow; Char_t fTriggerBits[48][64]" \
  version="[2]"	\
  target="fTriggerBits" \
  targetType="Int_t*" \
  code="{fTriggerBits = new Int_t[onfile.fNEntries]; for (Int_t i=0; i<onfile.fNEntries; ++i) fTriggerBits[i]=(onfile.fColumn && onfile.fRow)?onfile.fTriggerBits[onfile.fColumn[i]][onfile.fRow[i]]:0;}"

#pragma link C++ class  AliESDfriend+;

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="UChar_t fTRDpidQuality"  version="[-47]" target="fTRDntracklets" targetType="UChar_t" code="{fTRDntracklets=onfile.fTRDpidQuality;}"
// see http://root.cern.ch/svn/root/trunk/io/doc/DataModelEvolution.txt

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Int_t fTOFLabel[3]"  version="[-68]" target="fTOFLabel" targetType="Int_t*" code="{fTOFLabel = new Int_t[3];for(Int_t i=0;i < 3;i++) fTOFLabel[i]=onfile.fTOFLabel[i];}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTrackTime[5]"  version="[-68]" \
target="fTrackTime" targetType="Double32_t*" include="AliPID.h" \
code="{fTrackTime = new Double32_t[AliPID::kSPECIESC];for(Int_t isp=AliPID::kSPECIESC;isp--;) fTrackTime[isp]=isp<AliPID::kSPECIES ? onfile.fTrackTime[isp]:0;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTrackLength"  version="[-68]" target="fTrackLength" targetType="Double32_t" code="{fTrackLength=onfile.fTrackLength;}"


#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTOFsignal"  version="[-68]" target="fTOFsignal" targetType="Double32_t" code="{fTOFsignal=onfile.fTOFsignal;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTOFsignalToT"  version="[-68]" target="fTOFsignalToT" targetType="Double32_t" code="{fTOFsignalToT=onfile.fTOFsignalToT;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTOFsignalRaw"  version="[-68]" target="fTOFsignalRaw" targetType="Double32_t" code="{fTOFsignalRaw=onfile.fTOFsignalRaw;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTOFsignalDx"  version="[-68]" target="fTOFsignalDx" targetType="Double32_t" code="{fTOFsignalDx=onfile.fTOFsignalDx;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTOFsignalDz"  version="[-68]" target="fTOFsignalDz" targetType="Double32_t" code="{fTOFsignalDz=onfile.fTOFsignalDz;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Short_t fTOFdeltaBC"  version="[-68]" target="fTOFdeltaBC" targetType="Short_t" code="{fTOFdeltaBC=onfile.fTOFdeltaBC;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Short_t fTOFl0l1"  version="[-68]" target="fTOFl0l1" targetType="Short_t" code="{fTOFl0l1=onfile.fTOFl0l1;}"
#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Int_t fTOFCalChannel"  version="[-68]" target="fTOFCalChannel" targetType="Int_t" code="{fTOFCalChannel=onfile.fTOFCalChannel;}"

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fR[5]"  version="[-70]" \
 target="fR" targetType="Double32_t*" include="AliPID.h" \
   code="{fR = new Double32_t[AliPID::kSPECIES];for(Int_t isp=5;isp--;) fR[isp]=onfile.fR[isp];}"

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTPCr[5]"  version="[-70]" \
   target="fTPCr" targetType="Double32_t*" include="AliPID.h" \
   code="{fTPCr = new Double32_t[AliPID::kSPECIES];for(Int_t isp=5;isp--;) fTPCr[isp]=onfile.fTPCr[isp];}"

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fITSr[5]"  version="[-70]" \
   target="fITSr" targetType="Double32_t*" include="AliPID.h" \
   code="{fITSr = new Double32_t[AliPID::kSPECIES];for(Int_t isp=5;isp--;) fITSr[isp]=onfile.fITSr[isp];}"

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTRDr[5]"  version="[-70]" \
   target="fTRDr" targetType="Double32_t*" include="AliPID.h" \
   code="{fTRDr = new Double32_t[AliPID::kSPECIES];for(Int_t isp=5;isp--;) fTRDr[isp]=onfile.fTRDr[isp];}"

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fTOFr[5]"  version="[-70]" \
   target="fTOFr" targetType="Double32_t*" include="AliPID.h" \
   code="{fTOFr = new Double32_t[AliPID::kSPECIES];for(Int_t isp=5;isp--;) fTOFr[isp]=onfile.fTOFr[isp];}"

#pragma read sourceClass="AliESDtrack" targetClass="AliESDtrack" source="Double32_t fHMPIDr[5]"  version="[-70]" \
   target="fHMPIDr" targetType="Double32_t*" include="AliPID.h" \
   code="{fHMPIDr = new Double32_t[AliPID::kSPECIES];for(Int_t isp=5;isp--;) fHMPIDr[isp]=onfile.fHMPIDr[isp];}"

#pragma link C++ class  AliESDtrack+;
#pragma read sourceClass="AliESDfriendTrack" targetClass="AliESDfriendTrack" source="Int_t fITSindex" version="[-3]" \
        target="fnMaxITScluster, fITSindex" targetType="Int_t, Int_t*" code="{fnMaxITScluster = 12; fITSindex= new Int_t[fnMaxITScluster]; memcpy(fITSindex, &(onfile.fITSindex), fnMaxITScluster*sizeof(Int_t));}"
#pragma read sourceClass="AliESDfriendTrack" targetClass="AliESDfriendTrack" source="Int_t fTPCindex" version="[-3]" \
        target="fnMaxTPCcluster, fTPCindex" targetType="Int_t, Int_t*" code="{fnMaxTPCcluster = 160; fTPCindex= new Int_t[fnMaxTPCcluster]; memcpy(fTPCindex, &(onfile.fTPCindex), fnMaxTPCcluster*sizeof(Int_t));}"
#pragma read sourceClass="AliESDfriendTrack" targetClass="AliESDfriendTrack" source="Int_t fTRDindex" version="[-3]" \
        target="fnMaxTRDcluster, fTRDindex" targetType="Int_t, Int_t*" code="{fnMaxTRDcluster = 180; fTRDindex= new Int_t[fnMaxTRDcluster]; memcpy(fTRDindex, &(onfile.fTRDindex), fnMaxTRDcluster*sizeof(Int_t));}"

#pragma link C++ class  AliESDfriendTrack+;
#pragma link C++ class  AliESDMuonTrack+;
#pragma link C++ class  AliESDPmdTrack+;
#pragma link C++ class  AliESDTrdTrigger+;
#pragma link C++ class  AliESDTrdTrack+;
#pragma link C++ class  AliESDTrdTracklet+;
#pragma read sourceClass="AliESDTrdTracklet" targetClass="AliESDTrdTracklet" \
  source="Int_t fLabel" version="[-2]"	target="fLabel" targetType="Int_t[3]"\
  code="{fLabel[0]=onfile.fLabel; fLabel[1]=fLabel[2]=-1;}"
 
#pragma link C++ class  AliESDHLTtrack+;
#pragma link C++ class  AliESDv0+;
#pragma link C++ class  AliESDcascade+;
#pragma link C++ class  AliVertex+;
#pragma link C++ class  AliESDVertex+;
#pragma link C++ class  AliESDpid+;
#pragma link C++ class  AliESDkink+;
#pragma link C++ class  AliESDV0Params+;
#pragma link C++ class  AliV0HypSel+;
#pragma link C++ class  AliESDCaloCluster+;
#pragma link C++ class  AliESDCalofriend+;
#pragma link C++ class  AliESDMuonCluster+;
#pragma link C++ class  AliESDMuonPad+;

#pragma link C++ class  AliKFParticleBase+;
#pragma link C++ class  AliKFParticle+;
#pragma link C++ class  AliKFVertex+;

#pragma link C++ class  AliKalmanTrack+;
#pragma link C++ class  AliVertexerTracks+;
#pragma link C++ class  AliStrLine+;
#pragma link C++ class  AliTrackPointArray+;
#pragma link C++ class  AliTrackPoint+;

#pragma link C++ class AliTrackPointArray+;
#pragma link C++ class AliTrackPoint+;

#pragma link C++ class  AliESDFMD+;
#pragma read sourceClass="AliESDFMD" targetClass="AliESDFMD" source="float fNoiseFactor"  version="[-3]" target="fNoiseFactor" targetType="float" code="{newObj->SetNoiseFactor(onfile.fNoiseFactor < 1 ? 4 : onfile.fNoiseFactor);newObj->SetBit(AliESDFMD::kNeedNoiseFix);}"
#pragma read sourceClass="AliESDFMD" targetClass="AliESDFMD" source="bool fAngleCorrected"  version="[-3]" target="fAngleCorrected" targetType="bool" code="{newObj->SetAngleCorrected(onfile.fAngleCorrected ? onfile.fAngleCorrected : true);}"

#pragma link C++ class  AliFMDMap+;
#pragma link C++ class  AliFMDFloatMap+;

#pragma link C++ class  AliESDVZERO+;
#pragma link C++ class  AliESDTZERO+;
#pragma link C++ class  AliESDACORDE+;
#pragma link C++ class  AliESDAD+;

#pragma link C++ class  AliESDMultITS+;
#pragma link C++ class  AliMultiplicity+;

#pragma link C++ class  AliSelector+;

#pragma link C++ class  AliRawDataErrorLog+;

#pragma link C++ class  AliMeanVertex+;
#pragma link C++ class  AliESDCaloCells+;

#pragma link C++ class  AliESDVZEROfriend+;
#pragma link C++ class  AliESDTZEROfriend+;
#pragma link C++ class  AliESDADfriend+;

#pragma link C++ class  AliESDHandler+;
#pragma link C++ class  AliTrackerBase+;

#pragma link C++ namespace AliESDUtils;

#pragma link C++ class  AliTriggerIR+;
#pragma link C++ class  AliTriggerScalersESD+;
#pragma link C++ class  AliTriggerScalersRecordESD+;
#pragma link C++ class AliTriggerCluster+;
#pragma link C++ class AliTriggerDescriptor+;
#pragma link C++ class AliTriggerInput+;
#pragma link C++ class AliTriggerInteraction+;
#pragma link C++ class AliTriggerPFProtection+;
#pragma link C++ class AliTriggerBCMask+;
#pragma link C++ class AliTriggerClass+;
#pragma link C++ class AliTriggerConfiguration+;
#pragma link C++ class AliExpression+;
#pragma link C++ class AliVariableExpression+;
#pragma link C++ class AliESDCosmicTrack+;

#pragma link C++ class  AliV0vertexer+;
#pragma link C++ class  AliCascadeVertexer+;

#pragma link C++ class  AliESDTOFHit+;
#pragma link C++ class  AliESDTOFMatch+;
#pragma link C++ class  AliESDTOFCluster+;

#pragma link C++ function AliESDUtils::GetCorrV0(const AliESDEvent*,Float_t &);
#pragma link C++ function AliESDUtils::GetCorrSPD2(Float_t,Float_t);
#pragma link C++ function operator*(const AliFMDMap&,const AliFMDMap&);
#pragma link C++ function operator/(const AliFMDMap&,const AliFMDMap&);
#pragma link C++ function operator+(const AliFMDMap&,const AliFMDMap&);
#pragma link C++ function operator-(const AliFMDMap&,const AliFMDMap&);
  
#pragma link C++ class  AliESDMuonGlobalTrack+;  

#pragma link C++ class  AliESDFIT+;  
#endif
