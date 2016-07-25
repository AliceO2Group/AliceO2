// $Id$

//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Mikolaj Krzewicki <mikolaj.krzewicki@cern.ch>         *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

//  @file   AliHLTTPCClusterStatComponent.cxx
//  @author David Rohr <drohr@cern.ch>
// 

#include "AliHLTTPCClusterStatComponent.h"
#include "AliHLTTPCDefinitions.h"
#include "AliHLTTPCRawCluster.h"
#include "AliHLTTPCClusterXYZ.h"
#include "AliHLTTPCReverseTransformInfoV1.h"
#include "AliTPCcalibDB.h"
#include "AliTPCParam.h"
#include "AliHLTTPCGeometry.h"
#include "AliHLTExternalTrackParam.h"
#include "AliHLTGlobalBarrelTrack.h"
#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCDataCompressionComponent.h"
#include "AliHLTTPCClusterTransformation.h"
#include "AliTPCTransform.h"
#include "AliCDBManager.h"
#include "AliCDBEntry.h"
#include "AliGRPObject.h"
#include "AliGeomManager.h"
#include "AliRunInfo.h"
#include "AliEventInfo.h"
#include <TGeoGlobalMagField.h>
#include "AliRawEventHeaderBase.h"
#include "AliRecoParam.h"
#include "AliTPCRecoParam.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCGMTrackLinearisation.h"
#include "AliHLTTPCGMTrackParam.h"

ClassImp(AliHLTTPCClusterStatComponent)

AliHLTTPCClusterStatComponent::AliHLTTPCClusterStatComponent() : AliHLTProcessor()
, fSliceParam(NULL)
, fTotal(0)
, fSplitPad(0)
, fSplitTime(0)
, fSplitPadTime(0)
, fSplitPadOrTime(0)
, fAssigned(0)
, fPrintClusters(0)
, fPrintClustersScaled(0)
, fDumpClusters(0)
, fAggregate(0)
, fEvent(0)
{
}

AliHLTTPCClusterStatComponent::~AliHLTTPCClusterStatComponent()
{
}

void AliHLTTPCClusterStatComponent::GetInputDataTypes(AliHLTComponentDataTypeList& list)
{
  list.push_back(AliHLTTPCDefinitions::fgkRawClustersDataType | kAliHLTDataOriginTPC);
  list.push_back(AliHLTTPCDefinitions::fgkTPCReverseTransformInfoDataType);
  list.push_back(AliHLTTPCDefinitions::ClustersXYZDataType());
  list.push_back((kAliHLTDataTypeTrack | kAliHLTDataOriginTPC));
}

AliHLTComponentDataType AliHLTTPCClusterStatComponent::GetOutputDataType()
{
  return kAliHLTDataTypeHistogram|kAliHLTDataOriginOut;
}

void AliHLTTPCClusterStatComponent::GetOutputDataSize(unsigned long& constBase, double& inputMultiplier)
{
    constBase = 2000000;
    inputMultiplier = 0.0;
}

int AliHLTTPCClusterStatComponent::ProcessOption(TString option, TString value)
{
    int iResult = 0;

    if (option.EqualTo("print-clusters"))
    {
	fPrintClusters = 1;
    }
    else if (option.EqualTo("aggregate"))
    {
	fAggregate = 1;
    }
    else if (option.EqualTo("print-clusters-scaled"))
    {
	fPrintClustersScaled = 1;
    }
    else if (option.EqualTo("dump-clusters"))
    {
	fDumpClusters = 1;
    }
    else
    {
        HLTError("invalid option: %s", value.Data());
        return -EINVAL;
    }
    return iResult;
}

int AliHLTTPCClusterStatComponent::DoInit(int argc, const char** argv)
{
    int iResult=0;

    if (ProcessOptionString(GetComponentArgs())<0)
    {
        HLTFatal("wrong config string! %s", GetComponentArgs().c_str());
        return -EINVAL;
    }

    if (fDumpClusters) if ((fp = fopen("clusters.dump", "w+b")) == NULL) return -1;
    
    AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();
    const AliMagF * field = (AliMagF*) TGeoGlobalMagField::Instance()->GetField();
    pCalib->SetExBField(field);
    AliCDBEntry *entry = AliCDBManager::Instance()->Get("GRP/GRP/Data"); 
    AliGRPObject tmpGRP, *pGRP=0; 
    pGRP = dynamic_cast<AliGRPObject*>(entry->GetObject());
    AliRunInfo runInfo(pGRP->GetLHCState(),pGRP->GetBeamType(),pGRP->GetBeamEnergy(),pGRP->GetRunType(),pGRP->GetDetectorMask());
    AliEventInfo evInfo;
    evInfo.SetEventType(AliRawEventHeaderBase::kPhysicsEvent);
    entry=AliCDBManager::Instance()->Get("TPC/Calib/RecoParam"); 
    TObject *recoParamObj = entry->GetObject(); 

    static AliRecoParam fOfflineRecoParam;
    if (dynamic_cast<TObjArray*>(recoParamObj)) {
      TObjArray *copy = (TObjArray*)( static_cast<TObjArray*>(recoParamObj)->Clone() );
      fOfflineRecoParam.AddDetRecoParamArray(1,copy);
    } else if (dynamic_cast<AliDetectorRecoParam*>(recoParamObj)) {
      AliDetectorRecoParam *copy = (AliDetectorRecoParam*)static_cast<AliDetectorRecoParam*>(recoParamObj)->Clone();
      fOfflineRecoParam.AddDetRecoParam(1,copy);
    }
    fOfflineRecoParam.SetEventSpecie(&runInfo, evInfo, 0);    
    AliTPCRecoParam* recParam = (AliTPCRecoParam*)fOfflineRecoParam.GetDetRecoParam(1); 
    pCalib->GetTransform()->SetCurrentRecoParam(recParam); 
    
    fSliceParam = new AliHLTTPCCAParam();

    {
	//From GMMerger
        int iSec = 0;
        float inRmin = 83.65;
        float outRmax = 247.7;
        float plusZmin = 0.0529937;
        float plusZmax = 249.778;
        float dalpha = 0.349066;
        float alpha = 0.174533 + dalpha * iSec;
        float zMin =  plusZmin; //zPlus ? plusZmin : minusZmin;
        float zMax =  plusZmax; //zPlus ? plusZmax : minusZmax;
        int nRows = AliHLTTPCGeometry::GetNRows();
        float padPitch = 0.4;
        float sigmaZ = 0.228808;
        float *rowX = new float [nRows];
        for ( int irow = 0; irow < nRows; irow++ ) {
            rowX[irow] = AliHLTTPCGeometry::Row2X( irow );
        }
        
        float solenoidBz = GetBz();
        
        fSliceParam->Initialize( iSec, nRows, rowX, alpha, dalpha, inRmin, outRmax, zMin, zMax, padPitch, sigmaZ, solenoidBz );
        fSliceParam->Update();
        delete[] rowX;
    }
    
    {
	const double kCLight = 0.000299792458;
        double constBz = fSliceParam->BzkG() * kCLight;
        
        fPolinomialFieldBz[0] = constBz * (  0.999286   );
        fPolinomialFieldBz[1] = constBz * ( -4.54386e-7 );
        fPolinomialFieldBz[2] = constBz * (  2.32950e-5 );
        fPolinomialFieldBz[3] = constBz * ( -2.99912e-7 );
        fPolinomialFieldBz[4] = constBz * ( -2.03442e-8 );
        fPolinomialFieldBz[5] = constBz * (  9.71402e-8 );    
    }
    
    return iResult;
}

int AliHLTTPCClusterStatComponent::DoDeinit()
{
    if (fDumpClusters) fclose(fp);
    delete fSliceParam;
    fSliceParam = NULL;
    return 0;
}

void AliHLTTPCClusterStatComponent::TransformReverse(int slice, int row, float y, float z, float padtime[], const AliHLTTPCReverseTransformInfoV1* revInfo, bool applyCorrection)
{
    AliTPCcalibDB* calib = AliTPCcalibDB::Instance();
    AliTPCParam* param = calib->GetParameters();
    
    float padWidth = 0;
    float padLength = 0;
    float maxPad = 0;
    float sign = slice < 18 ? 1 : -1;
    float zwidth;
    
    int sector;
    int sectorrow;
    if (row < AliHLTTPCGeometry::GetNRowLow())
    {
        sector = slice;
        sectorrow = row;
	maxPad = param->GetNPadsLow(sectorrow);
	padLength = param->GetPadPitchLength(sector,sectorrow);
	padWidth = param->GetPadPitchWidth(sector);
    }
    else
    {
        sector = slice + 36;
        sectorrow = row - AliHLTTPCGeometry::GetNRowLow();
	maxPad = param->GetNPadsUp(sectorrow);
        padLength = param->GetPadPitchLength(sector,sectorrow);
        padWidth  = param->GetPadPitchWidth(sector);
    }

    if (applyCorrection)
    {
	float correctionY = revInfo->fCorrectY1 + revInfo->fCorrectY2 * param->GetPadRowRadii(sector,sectorrow) + revInfo->fCorrectY3 * (AliHLTTPCGeometry::GetZLength() - fabs(z));
	y -= correctionY;
    }

    padtime[0] = y * sign / padWidth + 0.5 * maxPad;

    float vdcorrectionTime, vdcorrectionTimeGY, time0corrTime, deltaZcorrTime, zLength;
    if (slice < 18)
    {
	vdcorrectionTime = revInfo->fVdcorrectionTimeA;
	vdcorrectionTimeGY = revInfo->fVdcorrectionTimeGYA;
	time0corrTime = revInfo->fTime0corrTimeA;
	deltaZcorrTime = revInfo->fDeltaZcorrTimeA;
	zLength = revInfo->fZLengthA;
    }
    else
    {
	vdcorrectionTime = revInfo->fVdcorrectionTimeC;
	vdcorrectionTimeGY = revInfo->fVdcorrectionTimeGYC;
	time0corrTime = revInfo->fTime0corrTimeC;
	deltaZcorrTime = revInfo->fDeltaZcorrTimeC;
	zLength = revInfo->fZLengthC;
    }
    
    float xyzGlobal[2] = {param->GetPadRowRadii(sector,sectorrow), y};
    AliHLTTPCGeometry::Local2Global(xyzGlobal, slice);

    zwidth = revInfo->fZWidth * revInfo->fDriftCorr;
    zwidth *= vdcorrectionTime * (1 + xyzGlobal[1] * vdcorrectionTimeGY);

    float time = z + deltaZcorrTime;
    time = zLength - time * sign;
    time += 3. * revInfo->fZSigma + time0corrTime;
    time /= zwidth;
    time += revInfo->fNTBinsL1;
    padtime[1] = time;
    
    if (applyCorrection)
    {
	float m = slice >= 18 ? revInfo->fDriftTimeFactorC : revInfo->fDriftTimeFactorA;
	float n = slice >= 18 ? revInfo->fDriftTimeOffsetC : revInfo->fDriftTimeOffsetA;
	float correctionTime = (z - n) / m;
	padtime[1] += correctionTime;
    }
}


void AliHLTTPCClusterStatComponent::TransformForward(int slice, int row, float pad, float time, float xyz[], const AliHLTTPCReverseTransformInfoV1* revInfo, bool applyCorrection)
{
    AliTPCcalibDB* calib = AliTPCcalibDB::Instance();
    AliTPCParam* param = calib->GetParameters();

    float padWidth = 0;
    float padLength = 0;
    float maxPad = 0;
    float sign = slice < 18 ? 1 : -1;
    float zwidth;
    
    int sector;
    int sectorrow;
    if (row < AliHLTTPCGeometry::GetNRowLow())
    {
        sector = slice;
        sectorrow = row;
	maxPad = param->GetNPadsLow(sectorrow);
	padLength = param->GetPadPitchLength(sector,sectorrow);
	padWidth = param->GetPadPitchWidth(sector);
    }
    else
    {
        sector = slice + 36;
        sectorrow = row - AliHLTTPCGeometry::GetNRowLow();
	maxPad = param->GetNPadsUp(sectorrow);
        padLength = param->GetPadPitchLength(sector,sectorrow);
        padWidth  = param->GetPadPitchWidth(sector);
    }

    xyz[0] = param->GetPadRowRadii(sector,sectorrow);
    xyz[1] = (pad - 0.5 * maxPad) * padWidth * sign;
    
    float vdcorrectionTime, vdcorrectionTimeGY, time0corrTime, deltaZcorrTime, zLength;
    if (slice < 18)
    {
	vdcorrectionTime = revInfo->fVdcorrectionTimeA;
	vdcorrectionTimeGY = revInfo->fVdcorrectionTimeGYA;
	time0corrTime = revInfo->fTime0corrTimeA;
	deltaZcorrTime = revInfo->fDeltaZcorrTimeA;
	zLength = revInfo->fZLengthA;
    }
    else
    {
	vdcorrectionTime = revInfo->fVdcorrectionTimeC;
	vdcorrectionTimeGY = revInfo->fVdcorrectionTimeGYC;
	time0corrTime = revInfo->fTime0corrTimeC;
	deltaZcorrTime = revInfo->fDeltaZcorrTimeC;
	zLength = revInfo->fZLengthC;
    }
    
    float xyzGlobal[2] = {xyz[0], xyz[1]};
    AliHLTTPCGeometry::Local2Global(xyzGlobal, slice);

    zwidth = revInfo->fZWidth * revInfo->fDriftCorr;
    zwidth *= vdcorrectionTime * (1 + xyzGlobal[1] * vdcorrectionTimeGY);

    xyz[2] = time - revInfo->fNTBinsL1;
    xyz[2] *= zwidth;
    xyz[2] -= 3. * revInfo->fZSigma + time0corrTime;
    xyz[2] = sign * ( zLength - xyz[2]);
    xyz[2] -= deltaZcorrTime;
    
    if (applyCorrection)
    {
	float m = slice >= 18 ? revInfo->fDriftTimeFactorC : revInfo->fDriftTimeFactorA;
	float n = slice >= 18 ? revInfo->fDriftTimeOffsetC : revInfo->fDriftTimeOffsetA;
	float correctionTime = (xyz[2] - n) / m;
	xyz[2] += correctionTime * zwidth * sign;
	float correctionY = revInfo->fCorrectY1 + revInfo->fCorrectY2 * xyz[0] + revInfo->fCorrectY3 * (AliHLTTPCGeometry::GetZLength() - fabs(xyz[2]));
	xyz[1] += correctionY;
    }

}

int AliHLTTPCClusterStatComponent::DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& /*trigData*/, AliHLTUInt8_t* /*outputPtr*/, AliHLTUInt32_t& /*size*/, AliHLTComponentBlockDataList& /*outputBlocks*/)
{
    int iResult=0;

    if (!IsDataEvent()) {return iResult;}

    if (!fAggregate)
    {
	fTotal = fSplitPad = fSplitTime = fSplitPadTime = fSplitPadOrTime = 0;
    }
    int nBlocks = evtData.fBlockCnt;
    
    const AliHLTTPCReverseTransformInfoV1* revInfo = NULL;

    AliHLTTPCRawClusterData* clustersArray[36][6];
    AliHLTTPCClusterXYZData* clustersTransformedArray[36][6];
    AliHLTTPCTrackHelperStruct* clustersTrackIDArray[36][6];
    memset(clustersArray, 0, 36 * 6 * sizeof(void*));
    memset(clustersTransformedArray, 0, 36 * 6 * sizeof(void*));
    memset(clustersTrackIDArray, 0, 36 * 6 * sizeof(void*));
    
    AliHLTTracksData* tracks = NULL;
    
    float bz = GetBz();
    
    AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();
    AliTPCParam *tpcParam = pCalib->GetParameters(); 
    tpcParam->Update();
    tpcParam->ReadGeoMatrices();  
    AliTPCTransform* transform = pCalib->GetTransform();
    const AliTPCRecoParam *rec = transform->GetCurrentRecoParam();
    transform->SetCurrentTimeStamp(GetTimeStamp());

    for (int ndx=0; ndx<nBlocks; ndx++)
    {
        const AliHLTComponentBlockData* iter = blocks+ndx;
        if (iter->fDataType == (AliHLTTPCDefinitions::fgkTPCReverseTransformInfoDataType))
        {
            revInfo = (const AliHLTTPCReverseTransformInfoV1*) iter->fPtr;
            /*HLTImportant("Reverse Transform Info: NTBin %f ZWid %f ZSig %f ZLenA %f ZLenC %f driftc %f t0A %f deltaA %f vdA %f vdGYA %f t0C %f deltaC %f vdC %f vdGYC %f",
              revInfo->fNTBinsL1, revInfo->fZWidth, revInfo->fZSigma, revInfo->fZLengthA, revInfo->fZLengthC, revInfo->fDriftCorr,
              revInfo->fTime0corrTimeA, revInfo->fDeltaZcorrTimeA, revInfo->fVdcorrectionTimeA, revInfo->fVdcorrectionTimeGYA,
              revInfo->fTime0corrTimeC, revInfo->fDeltaZcorrTimeC, revInfo->fVdcorrectionTimeC, revInfo->fVdcorrectionTimeGYC);*/
        }

        if (iter->fDataType == (AliHLTTPCDefinitions::fgkRawClustersDataType | kAliHLTDataOriginTPC))
        {
            int slice = AliHLTTPCDefinitions::GetMinSliceNr(iter->fSpecification);
            int patch = AliHLTTPCDefinitions::GetMinPatchNr(iter->fSpecification);

            clustersArray[slice][patch] = (AliHLTTPCRawClusterData*)(iter->fPtr);
        }

        if (iter->fDataType == AliHLTTPCDefinitions::ClustersXYZDataType())
        {
            int slice = AliHLTTPCDefinitions::GetMinSliceNr(iter->fSpecification);
            int patch = AliHLTTPCDefinitions::GetMinPatchNr(iter->fSpecification);

            clustersTransformedArray[slice][patch] = (AliHLTTPCClusterXYZData*)(iter->fPtr);
            if (clustersTransformedArray[slice][patch]->fCount)
            {
        	clustersTrackIDArray[slice][patch] = new AliHLTTPCTrackHelperStruct[clustersTransformedArray[slice][patch]->fCount];
        	memset(clustersTrackIDArray[slice][patch], 0, clustersTransformedArray[slice][patch]->fCount * sizeof(AliHLTTPCTrackHelperStruct));
        	for (int i = 0;i < clustersTransformedArray[slice][patch]->fCount;i++) clustersTrackIDArray[slice][patch][i].fID = -1;
            }
        }

        if (iter->fDataType == (kAliHLTDataTypeTrack | kAliHLTDataOriginTPC))
        {
            tracks = (AliHLTTracksData*) iter->fPtr;
        }
    }

    if (revInfo == NULL)
    {
	HLTError("RevInfo missing");
	return(0);
    }
    if (tracks == NULL)
    {
	HLTError("Tracks missing");
	return(0);
    }
    
    double residualBarrelTrackY = 0, residualBarrelTrackZ = 0, residualExternalTrackY = 0, residualExternalTrackZ = 0, residualBacktransformPad = 0, residualBacktransformTime = 0;
    double residualBarrelTrackYabs = 0, residualBarrelTrackZabs = 0, residualExternalTrackYabs = 0, residualExternalTrackZabs = 0, residualBacktransformPadabs = 0, residualBacktransformTimeabs = 0;
    double residualFitTrackY = 0, residualFitTrackZ = 0, residualFitTrackYabs = 0, residualFitTrackZabs = 0, residualTrackRawPad = 0, residualTrackRawTime = 0, residualTrackRawPadabs = 0, residualTrackRawTimeabs = 0;
    int nClusterTracks = 0, nClusters = 0, nClusterTracksRaw = 0;

    const AliHLTUInt8_t* pCurrent = reinterpret_cast<const AliHLTUInt8_t*>(tracks->fTracklets);
    for (unsigned i = 0;i < tracks->fCount;i++)
    {
        const AliHLTExternalTrackParam* track = reinterpret_cast<const AliHLTExternalTrackParam*>(pCurrent);
        if (track->fNPoints == 0) continue;

	AliHLTGlobalBarrelTrack btrack(*track);
	btrack.CalculateHelixParams(bz);
	
	AliExternalTrackParam etrack(btrack);
	
	AliHLTTPCGMTrackParam ftrack;
	float falpha;
	AliHLTTPCGMTrackLinearisation ft0;
	float ftrDzDs2;
	AliHLTTPCGMTrackParam::AliHLTTPCGMTrackFitParam fpar;
	
	int hitsUsed = 0;
	float averageCharge = 0;
	float averageQMax = 0;
	AliHLTTPCTrackHelperStruct* hitIndexCache[1024];
        for (int ip = 0;ip < track->fNPoints;ip++)
        {
            int clusterID = track->fPointIDs[ip];
	    int slice = AliHLTTPCGeometry::CluID2Slice(clusterID);
	    int patch = AliHLTTPCGeometry::CluID2Partition(clusterID);
	    int index = AliHLTTPCGeometry::CluID2Index(clusterID);
	    
	    if (clustersTrackIDArray[slice][patch][index].fID != -1)
	    {
		HLTDebug("Already assigned hit %d of track %d, skipping", ip, i);
		continue;
	    }
	    
	    if (index > clustersArray[slice][patch]->fCount)
	    {
		HLTError("Cluster index out of range");
		continue;
	    }
	    
	    AliHLTTPCRawCluster &cluster = clustersArray[slice][patch]->fClusters[index];
	    AliHLTTPCClusterXYZ &clusterTransformed = clustersTransformedArray[slice][patch]->fClusters[index];
	    
	    int padrow = AliHLTTPCGeometry::GetFirstRow(patch) + cluster.GetPadRow();
	    float x = AliHLTTPCGeometry::Row2X(padrow);
	    float y = 0.0;
	    float z = 0.0;

	    float xyz[3];
	    if (1) //Use forward (exact reverse-reverse) transformation of raw cluster (track fit in distorted coordinates)
	    {
		TransformForward(slice, padrow, cluster.GetPad(), cluster.GetTime(), xyz, revInfo, true);
	    }
	    else
	    {	//Correct cluster coordinates using correct transformation
		xyz[0] = x;
		xyz[1] = clusterTransformed.fY;
		xyz[2] = clusterTransformed.fZ;
	    }
	    
	    float alpha = slice;
	    if (alpha > 18) alpha -= 18;
	    if (alpha > 9) alpha -= 18;
	    alpha = (alpha + 0.5f) * TMath::Pi() / 9.f;
	    btrack.CalculateCrossingPoint(x, alpha /* Better use btrack.GetAlpha() ?? */, y, z);
	    
	    etrack.Propagate(alpha, x, bz);
	    int rowType = padrow < 64 ? 0 : (padrow < 128 ? 2 : 1);
	    float fdL = 0;
	    float fex1i = 0;
	    if (ip == 0)
	    {
		ftrack.Par()[0] = xyz[1];
		ftrack.Par()[1] = xyz[2];
		for (int k = 2;k < 5;k++) ftrack.Par()[k] = etrack.GetParameter()[k];
		ftrack.SetX(xyz[0]);
		falpha = alpha;

		ft0.Set(ftrack.GetSinPhi(), 0., 0., ftrack.GetDzDs(), 0., ftrack.GetQPt());
		ftrDzDs2 = ft0.DzDs() * ft0.DzDs();

		const float kRho = 1.025e-3; //From GMMerger
		const float kRadLen = 29.532;
		const float kRhoOverRadLen = kRho / kRadLen;
		ftrack.CalculateFitParameters( fpar, kRhoOverRadLen, kRho, 0 );

		int fakeN;
		ftrack.PropagateTrack(fPolinomialFieldBz, xyz[0], ftrack.GetY(), ftrack.GetZ(), falpha, rowType, *fSliceParam, fakeN, falpha, .999, 0, 1, fpar, ft0, fdL, fex1i, ftrDzDs2);
	    }
	    else
	    {
		int fakeN;
		ftrack.PropagateTrack(fPolinomialFieldBz, xyz[0], ftrack.GetY(), ftrack.GetZ(), alpha, rowType, *fSliceParam, fakeN, falpha, .999, 0, 0, fpar, ft0, fdL, fex1i, ftrDzDs2);
	    }
	    
	    nClusterTracks++;
	    residualBarrelTrackYabs += fabs(clusterTransformed.fY - y);
	    residualBarrelTrackZabs += fabs(clusterTransformed.fZ - z);
	    residualExternalTrackYabs += fabs(clusterTransformed.fY - etrack.GetY());
	    residualExternalTrackZabs += fabs(clusterTransformed.fZ - etrack.GetZ());
	    residualBarrelTrackY += clusterTransformed.fY - y;
	    residualBarrelTrackZ += clusterTransformed.fZ - z;
	    residualExternalTrackY += clusterTransformed.fY - etrack.GetY();
	    residualExternalTrackZ += clusterTransformed.fZ - etrack.GetZ();
	    residualFitTrackY += clusterTransformed.fY - ftrack.GetY();
	    residualFitTrackZ += clusterTransformed.fZ - ftrack.GetZ();
	    residualFitTrackYabs += fabs(clusterTransformed.fY - ftrack.GetY());
	    residualFitTrackZabs += fabs(clusterTransformed.fZ - ftrack.GetZ());
	    
	    //Show residuals wrt track position
	    //HLTImportant("Residual %d btrack %f %f etrack %f %f ftrack %f %f", padrow, clusterTransformed.fY - y, clusterTransformed.fZ - z,
		//clusterTransformed.fY - etrack.GetY(), clusterTransformed.fZ - etrack.GetZ(),
		//clusterTransformed.fY - ftrack.GetY(), clusterTransformed.fZ - ftrack.GetZ());

	    float padtime[2];
	    TransformReverse(slice, padrow, ftrack.GetY(), ftrack.GetZ(), padtime, revInfo, true);
	    
	    //Check forward / backward transformation
	    /*float xyzChk[3];
	    TransformForward(slice, padrow, padtime[0], padtime[1], xyzChk, revInfo, true);
	    HLTImportant("BackwardForward Residual %f %f %f: %f %f", ftrack.GetX(), ftrack.GetY(), ftrack.GetZ(), ftrack.GetY() - xyzChk[1], ftrack.GetZ() - xyzChk[2]);*/

	    //Show residual wrt to raw cluster position
	    //HLTImportant("Raw Cluster Residual %d (%d/%d) %d: %f %f (%f %f)", i, ip, track->fNPoints, padrow, cluster.GetPad() - padtime[0], cluster.GetTime() - padtime[1], clusterTransformed.fY - ftrack.GetY(), clusterTransformed.fZ - ftrack.GetZ());
	    if (fabs(clusterTransformed.fY - ftrack.GetY()) > 5 || fabs(clusterTransformed.fZ - ftrack.GetZ()) > 5)
	    {
		break;
	    }

	    if (ip != 0)
	    {
		clustersTrackIDArray[slice][patch][index].fResidualPad = cluster.GetPad() - padtime[0];
		clustersTrackIDArray[slice][patch][index].fResidualTime = cluster.GetTime() - padtime[1];
		clustersTrackIDArray[slice][patch][index].fFirstHit = 0;

		residualTrackRawPad += cluster.GetPad() - padtime[0];
		residualTrackRawTime += cluster.GetTime() - padtime[1];
		residualTrackRawPadabs += fabs(cluster.GetPad() - padtime[0]);
		residualTrackRawTimeabs += fabs(cluster.GetTime() - padtime[1]);
		nClusterTracksRaw++;
	    }
	    else
	    {
		clustersTrackIDArray[slice][patch][index].fResidualPad = cluster.GetPad();
		clustersTrackIDArray[slice][patch][index].fResidualTime = cluster.GetTime();
		clustersTrackIDArray[slice][patch][index].fFirstHit = 1;
	    }
	    clustersTrackIDArray[slice][patch][index].fID = i;
	    clustersTrackIDArray[slice][patch][index].fTrack = track;
	    if (hitsUsed >= 1024)
	    {
		HLTFatal("hitIndex cache exceeded");
	    }
	    hitIndexCache[hitsUsed] = &clustersTrackIDArray[slice][patch][index];
	    hitsUsed++;
	    averageCharge += cluster.GetCharge();
	    averageQMax += cluster.GetQMax();

	    if (ip != 0)
	    {
		int fakeN;
	    	ftrack.UpdateTrack(fPolinomialFieldBz, xyz[0], xyz[1], xyz[2], alpha, rowType, *fSliceParam, fakeN, falpha, .999, fpar, ft0, fdL, fex1i, ftrDzDs2);
	    }
        }
        if (hitsUsed)
        {
	    averageCharge /= hitsUsed;
	    averageQMax /= hitsUsed;
	}
        for (int ip = 0;ip < hitsUsed;ip++)
        {
	    hitIndexCache[ip]->fAverageQMax = averageQMax;
	    hitIndexCache[ip]->fAverageQTot = averageCharge;
	}
        pCurrent += sizeof(AliHLTExternalTrackParam) + track->fNPoints * sizeof(UInt_t);
    }

    for (int is = 0;is < 36;is++) for (int ip = 0;ip < 6;ip++)
    {
	AliHLTTPCRawClusterData* clusters = clustersArray[is][ip];
	AliHLTTPCClusterXYZData* clustersTransformed = clustersTransformedArray[is][ip];
	int firstRow = AliHLTTPCGeometry::GetFirstRow(ip);
	
	if (clusters == NULL || clustersTransformed == NULL)
	{
	    HLTDebug("Clusters missing for slice %d patch %d\n", is, ip);
	    continue;
	}
	if (clusters->fCount != clustersTransformed->fCount)
	{
	    HLTError("Cluster cound not equal");
	    continue;
	}
	
    	for(UInt_t iCluster=0;iCluster<clusters->fCount;iCluster++)
    	{
    	    AliHLTTPCRawCluster &cluster = clusters->fClusters[iCluster];
    	    AliHLTTPCClusterXYZ &clusterTransformed = clustersTransformed->fClusters[iCluster];
    	    AliHLTTPCTrackHelperStruct &clusterTrack = clustersTrackIDArray[is][ip][iCluster];
    	    
	    int row = cluster.GetPadRow() + firstRow;

    	    float xyz[3];
    	    TransformForward(is, row, cluster.GetPad(), cluster.GetTime(), xyz, revInfo);
    	    
    	    float xyzOrig[3], xyzLocGlob[3];
	    {
		int sector = AliHLTTPCGeometry::GetNRowLow() ? is : is + 36;
		int sectorrow = AliHLTTPCGeometry::GetNRowLow() ? row : row - AliHLTTPCGeometry::GetNRowLow();
		
		Double_t xx[]={(double) sectorrow,cluster.GetPad(),cluster.GetTime()};
		transform->Transform(xx,&sector,0,1);
		
		Double_t yy[]={(double) sectorrow,cluster.GetPad(),cluster.GetTime()};
		transform->Local2RotatedGlobal(sector, yy);
		for (int k = 0;k < 3;k++) {xyzOrig[k] = xx[k];xyzLocGlob[k] = yy[k];}
	    }
	
	    float padtime[2];
	    TransformReverse(is, row, clusterTransformed.fY, clusterTransformed.fZ, padtime, revInfo, true);
	
	    nClusters++;
	    residualBacktransformPadabs += fabs(cluster.GetPad() - padtime[0]);
	    residualBacktransformTimeabs += fabs(cluster.GetTime() - padtime[1]);
	    residualBacktransformPad += cluster.GetPad() - padtime[0];
	    residualBacktransformTime += cluster.GetTime() - padtime[1];
	    
	    fTotal++;
	    if (cluster.GetFlagSplitPad()) fSplitPad++;
	    if (cluster.GetFlagSplitTime()) fSplitTime++;
	    if (cluster.GetFlagSplitAny()) fSplitPadOrTime++;
	    if (cluster.GetFlagSplitPad() && cluster.GetFlagSplitTime()) fSplitPadTime++;
	
	    if (fPrintClusters) HLTImportant("Slice %d, Patch %d, Row %d, Pad %.2f, Time %.2f, SPad %.2f, STime %.2f, QMax %d, QTot %d, SplitPad %d, SplitTime %d, TrackId %d, ResPad %.2f ResTime %.2f AvgQTot %d AvgQMax %d",
	        is, ip, (int) cluster.GetPadRow(), cluster.GetPad(), cluster.GetTime(), cluster.GetSigmaPad2(), cluster.GetSigmaTime2(), (int) cluster.GetQMax(), (int) cluster.GetCharge(),
	        (int) cluster.GetFlagSplitPad(), (int) cluster.GetFlagSplitTime(), (int) clusterTrack.fID, clusterTrack.fResidualPad, clusterTrack.fResidualTime, (int) clusterTrack.fAverageQTot, (int) clusterTrack.fAverageQMax);
	
	    if (clusterTrack.fID == -1) PrintDumpClustersScaled(is, ip, cluster, clusterTransformed, clusterTrack);
	}
    }

    if (fDumpClusters || fPrintClustersScaled)
    {
        const AliHLTUInt8_t* pCurrent = reinterpret_cast<const AliHLTUInt8_t*>(tracks->fTracklets);
	for (unsigned i = 0;i < tracks->fCount;i++)
	{
	    const AliHLTExternalTrackParam* track = reinterpret_cast<const AliHLTExternalTrackParam*>(pCurrent);
    	    for (int ip = 0;ip < track->fNPoints;ip++)
    	    {
        	int clusterID = track->fPointIDs[ip];
		int slice = AliHLTTPCGeometry::CluID2Slice(clusterID);
		int patch = AliHLTTPCGeometry::CluID2Partition(clusterID);
		int index = AliHLTTPCGeometry::CluID2Index(clusterID);

		AliHLTTPCRawCluster &cluster = clustersArray[slice][patch]->fClusters[index];
		AliHLTTPCClusterXYZ &clusterTransformed = clustersTransformedArray[slice][patch]->fClusters[index];
		AliHLTTPCTrackHelperStruct &clusterTrack = clustersTrackIDArray[slice][patch][index];
		
		if (clusterTrack.fID == i) PrintDumpClustersScaled(slice, patch, cluster, clusterTransformed, clusterTrack);
	    }
	    pCurrent += sizeof(AliHLTExternalTrackParam) + track->fNPoints * sizeof(UInt_t);
	}
    }


    for (int is = 0;is < 36;is++) for (int ip = 0;ip < 6;ip++) if (clustersTrackIDArray[is][ip]) delete[] clustersTrackIDArray[is][ip];

    fAssigned += nClusterTracks;
    HLTImportant("Total %d Assigned %d SplitPad %d SplitTime %d SplitPadTime %d SplitPadOrTime %d", fTotal, fAssigned, fSplitPad, fSplitTime, fSplitPadTime, fSplitPadOrTime);
    
    if (nClusterTracks)
    {
	residualBarrelTrackY /= nClusterTracks;
	residualBarrelTrackZ /= nClusterTracks;
	residualExternalTrackY /= nClusterTracks;
	residualExternalTrackZ /= nClusterTracks;
	residualBarrelTrackYabs /= nClusterTracks;
	residualBarrelTrackZabs /= nClusterTracks;
	residualExternalTrackYabs /= nClusterTracks;
	residualExternalTrackZabs /= nClusterTracks;
	residualFitTrackYabs /= nClusterTracks;
	residualFitTrackZabs /= nClusterTracks;
	residualFitTrackY /= nClusterTracks;
	residualFitTrackZ /= nClusterTracks;
    }
    if (nClusterTracksRaw)
    {
	residualTrackRawPadabs /= nClusterTracksRaw;
	residualTrackRawTimeabs /= nClusterTracksRaw;
	residualTrackRawPad /= nClusterTracksRaw;
	residualTrackRawTime /= nClusterTracksRaw;
    }
    if (nClusters)
    {
	residualBacktransformPad /= nClusters;
	residualBacktransformTime /= nClusters;
	residualBacktransformPadabs /= nClusters;
	residualBacktransformTimeabs /= nClusters;
    }

    HLTImportant("Average Res: BarrelTr %f %f, ExtlTr %f %f, FitTr %f %f BackTr %f %f TrkRaw %f %f", residualBarrelTrackY, residualBarrelTrackZ, residualExternalTrackY, residualExternalTrackZ, residualFitTrackY, residualFitTrackZ, residualBacktransformPad, residualBacktransformTime, residualTrackRawPad, residualTrackRawTime);
    HLTImportant("Average Abs Res: BarrelTr %f %f, ExtTr %f %f, FitTr %f %f BackTr %f %f TrkRaw %f %f", residualBarrelTrackYabs, residualBarrelTrackZabs, residualExternalTrackYabs, residualExternalTrackZabs, residualFitTrackYabs, residualFitTrackZabs, residualBacktransformPadabs, residualBacktransformTimeabs, residualTrackRawPadabs, residualTrackRawTimeabs);

    fEvent++;

    return iResult;
}

void AliHLTTPCClusterStatComponent::PrintDumpClustersScaled(int is, int ip, AliHLTTPCRawCluster &cluster, AliHLTTPCClusterXYZ &clusterTransformed, AliHLTTPCClusterStatComponent::AliHLTTPCTrackHelperStruct &clusterTrack)
{
    AliHLTUInt64_t pad64=0;
    if (!isnan(cluster.GetPad())) pad64=(AliHLTUInt64_t)round(cluster.GetPad()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kPad].fScale);

    AliHLTUInt64_t time64=0;
    if (!isnan(cluster.GetTime())) time64=(AliHLTUInt64_t)round(cluster.GetTime()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kTime].fScale);

    AliHLTUInt64_t sigmaPad64=0;
    if (!isnan(cluster.GetSigmaPad2())) sigmaPad64=(AliHLTUInt64_t)round(cluster.GetSigmaPad2()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fScale);

    AliHLTUInt64_t sigmaTime64=0;
    if (!isnan(cluster.GetSigmaTime2())) sigmaTime64=(AliHLTUInt64_t)round(cluster.GetSigmaTime2()*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fScale);

    if (sigmaPad64 >= (unsigned)1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fBitLength)
        sigmaPad64 = (1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaY2].fBitLength)-1;
    if (sigmaTime64 >= (unsigned)1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fBitLength)
                sigmaTime64 = (1<<AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kSigmaZ2].fBitLength)-1;

    AliHLTUInt64_t pad64res=0;
    pad64res=(AliHLTUInt64_t)round(clusterTrack.fResidualPad*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kPad].fScale);

    AliHLTUInt64_t time64res=0;
    time64res=(AliHLTUInt64_t)round(clusterTrack.fResidualTime*AliHLTTPCDefinitions::fgkClusterParameterDefinitions[AliHLTTPCDefinitions::kTime].fScale);

    if (fDumpClusters)
    {
        int dumpVals[16] = {fEvent, (int) is, (int) ip, (int) cluster.GetPadRow(), (int) pad64, (int) time64, (int) sigmaPad64, (int) sigmaTime64, (int) cluster.GetQMax(), (int) cluster.GetCharge(),
            (int) (cluster.GetFlagSplitPad() * 2 + cluster.GetFlagSplitTime()), (int) clusterTrack.fID, (int) pad64res, (int) time64res, (int) clusterTrack.fAverageQTot, (int) clusterTrack.fAverageQMax};
        fwrite(dumpVals, sizeof(int), 16, fp);
    }

    if (fPrintClustersScaled)
    {
	HLTImportant("Slice %d, Patch %d, Row %d, Pad %d, Time %d, SPad %d, STime %d, QMax %d, QTot %d, SplitPad %d, SplitTime %d, TrackID %d, PadRes %d, TimeRes %d AvgTot %d AvgMax %d",
	is, ip, (int) cluster.GetPadRow(), (int) pad64, (int) time64, (int) sigmaPad64, (int) sigmaTime64, (int) cluster.GetQMax(), (int) cluster.GetCharge(),
	    (int) cluster.GetFlagSplitPad(), (int) cluster.GetFlagSplitTime(), (int) clusterTrack.fID, (int) pad64res, (int) time64res, (int) clusterTrack.fAverageQTot, (int) clusterTrack.fAverageQMax);
    }
}
