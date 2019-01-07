#include "AliHLTGPUDumpComponent.h"

#include "AliGeomManager.h"
#include "AliGPUReconstruction.h"
#include "AliHLTTPCDefinitions.h"
#include "AliHLTTPCCAMCInfo.h"
#include "AliHLTTPCGMMergedTrackHit.h"
#include "AliHLTTPCClusterXYZ.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCRawCluster.h"
#include "AliHLTTPCCAGeometry.h"
#include "AliRunLoader.h"
#include "AliHeader.h"
#include "AliStack.h"
#include "AliExternalTrackParam.h"
#include "AliTrackReference.h"
#include "AliHLTTRDDefinitions.h"
#include "AliHLTTRDTrackletWord.h"
#include "AliHLTTRDTrackletLabels.h"
#include "TPCFastTransform.h"
#include "TPCFastTransformManager.h"
#include "AliRecoParam.h"
#include "AliTPCTransform.h"
#include "AliTPCcalibDB.h"
#include "AliCDBManager.h"
#include "AliGRPObject.h"
#include "AliCDBEntry.h"
#include "AliRunInfo.h"
#include "AliEventInfo.h"
#include "AliRawEventHeaderBase.h"
#include "AliTPCRecoParam.h"
#include <TGeoGlobalMagField.h>
#include <TVirtualMC.h>

#include "TTree.h"
#include "TParticle.h"
#include "TParticlePDG.h"
#include "TPDGCode.h"

AliHLTGPUDumpComponent::AliHLTGPUDumpComponent() : fSolenoidBz(0.f), fRec(NULL), fClusterData(NULL),
	fFastTransformManager( new ali_tpc_common::tpc_fast_transformation::TPCFastTransformManager ),
	fCalib(NULL), fRecParam(NULL), fOfflineRecoParam(), fOrigTransform(nullptr), fIsMC(false)
{
	fRec = AliGPUReconstruction::CreateInstance();
	fClusterData = new AliHLTTPCCAClusterData[36];
}

AliHLTGPUDumpComponent::~AliHLTGPUDumpComponent()
{
	delete fRec;
	delete[] fClusterData;
	delete fFastTransformManager;
}

const char *AliHLTGPUDumpComponent::GetComponentID()
{
	return "GPUDump";
}

void AliHLTGPUDumpComponent::GetInputDataTypes(vector<AliHLTComponentDataType> &list)
{
	list.clear();
	list.push_back(AliHLTTPCDefinitions::RawClustersDataType());
	list.push_back(AliHLTTPCDefinitions::ClustersXYZDataType());
	list.push_back(AliHLTTPCDefinitions::AliHLTDataTypeClusterMCInfo());
	list.push_back(AliHLTTRDDefinitions::fgkTRDTrackletDataType);
	list.push_back( AliHLTTRDDefinitions::fgkTRDMCTrackletDataType );
}

AliHLTComponentDataType AliHLTGPUDumpComponent::GetOutputDataType()
{
	return AliHLTTPCDefinitions::RawClustersDataType();
}

void AliHLTGPUDumpComponent::GetOutputDataSize(unsigned long &constBase, double &inputMultiplier)
{
	constBase = 10000;     // minimum size
	inputMultiplier = 0.6; // size relative to input
}

AliHLTComponent *AliHLTGPUDumpComponent::Spawn()
{
	return new AliHLTGPUDumpComponent;
}

int AliHLTGPUDumpComponent::DoInit(int argc, const char **argv)
{
	fSolenoidBz = GetBz();
	fIsMC = TVirtualMC::GetMC();
	
	if(!AliGeomManager::GetGeometry()) AliGeomManager::LoadGeometry();
	if(!AliGeomManager::GetGeometry()) HLTFatal("Can not initialise geometry");
	
	fCalib=AliTPCcalibDB::Instance();
	if(!fCalib) HLTFatal("Calibration not found");
	fCalib->SetRun(GetRunNo());
	fCalib->UpdateRunInformations(GetRunNo());
	
	const AliMagF * field = (AliMagF*) TGeoGlobalMagField::Instance()->GetField();
	fCalib->SetExBField(field);

	if(!fCalib->GetTransform()) HLTFatal("No TPC transformation found");
	
	AliGRPObject *pGRP=0;
	AliCDBEntry *entry = AliCDBManager::Instance()->Get("GRP/GRP/Data");
	if(!entry) HLTFatal("No GRP object found in data base");
	pGRP = dynamic_cast<AliGRPObject*>(entry->GetObject());
	  
	if(!pGRP) HLTFatal("Unknown format of the GRP object in data base");

	AliRunInfo runInfo(pGRP->GetLHCState(), pGRP->GetBeamType(), pGRP->GetBeamEnergy(), pGRP->GetRunType(), pGRP->GetDetectorMask());
	AliEventInfo evInfo;
	evInfo.SetEventType(AliRawEventHeaderBase::kPhysicsEvent);

	entry = AliCDBManager::Instance()->Get("TPC/Calib/RecoParam");
	if(!entry) HLTFatal("No TPC reco param entry found in data base");
	TObject *recoParamObj = entry->GetObject();
	if (dynamic_cast<TObjArray*>(recoParamObj))
	{
		TObjArray *copy = (TObjArray*)( static_cast<TObjArray*>(recoParamObj)->Clone() );
		fOfflineRecoParam.AddDetRecoParamArray(1,copy);
	}
	else if (dynamic_cast<AliDetectorRecoParam*>(recoParamObj))
	{
		AliDetectorRecoParam *copy = (AliDetectorRecoParam*)static_cast<AliDetectorRecoParam*>(recoParamObj)->Clone();
		fOfflineRecoParam.AddDetRecoParam(1,copy);
	}
	else
	{
		HLTFatal("Unknown format of the TPC Reco Param entry in the data base");
	}
	
	
	fOfflineRecoParam.SetEventSpecie(&runInfo, evInfo, 0);
	fRecParam = (AliTPCRecoParam*) fOfflineRecoParam.GetDetRecoParam(1);
	if( !fRecParam ) HLTFatal("No TPC Reco Param entry found for the given event specification");
	fCalib->GetTransform()->SetCurrentRecoParam(fRecParam);
	
	return 0;
}

int AliHLTGPUDumpComponent::DoDeinit()
{
	return 0;
}

int AliHLTGPUDumpComponent::Reconfigure(const char *cdbEntry, const char *chainId)
{
	return 0;
}

int AliHLTGPUDumpComponent::DoEvent(const AliHLTComponentEventData &evtData, const AliHLTComponentBlockData *blocks, AliHLTComponentTriggerData & /*trigData*/,
                                    AliHLTUInt8_t *outputPtr, AliHLTUInt32_t &size, vector<AliHLTComponentBlockData> &outputBlocks)
{
	if (GetFirstInputBlock(kAliHLTDataTypeSOR) || GetFirstInputBlock(kAliHLTDataTypeEOR))
	{
		return 0;
	}

	if (evtData.fBlockCnt <= 0)
	{
		HLTWarning("no blocks in event");
		return 0;
	}

	//Prepare everything for all slices
	const AliHLTTPCClusterMCData *clusterLabels[36][6] = {NULL};
	const AliHLTTPCClusterXYZData *clustersXYZ[36][6] = {NULL};
	const AliHLTTPCRawClusterData *clustersRaw[36][6] = {NULL};
	bool labelsPresent = false;
	AliHLTTRDTrackletWord *TRDtracklets = NULL;
    AliHLTTRDTrackletLabels *TRDtrackletsMC = NULL;
	int nTRDTrackletsTotal = 0;
    int nTRDTrackletsMCTotal = 0;

	for (unsigned long ndx = 0; ndx < evtData.fBlockCnt; ndx++)
	{
		const AliHLTComponentBlockData &pBlock = blocks[ndx];
		int slice = AliHLTTPCDefinitions::GetMinSliceNr(pBlock);
		int patch = AliHLTTPCDefinitions::GetMinPatchNr(pBlock);
		if (pBlock.fDataType == AliHLTTPCDefinitions::RawClustersDataType())
		{
			clustersRaw[slice][patch] = (const AliHLTTPCRawClusterData *) pBlock.fPtr;
		}
		else if (pBlock.fDataType == AliHLTTPCDefinitions::ClustersXYZDataType())
		{
			clustersXYZ[slice][patch] = (const AliHLTTPCClusterXYZData *) pBlock.fPtr;
		}
		else if (pBlock.fDataType == AliHLTTPCDefinitions::AliHLTDataTypeClusterMCInfo())
		{
			clusterLabels[slice][patch] = (const AliHLTTPCClusterMCData *) pBlock.fPtr;
			labelsPresent = true;
		}
		else if (pBlock.fDataType == AliHLTTRDDefinitions::fgkTRDTrackletDataType)
		{
			TRDtracklets = reinterpret_cast<AliHLTTRDTrackletWord*>(pBlock.fPtr);
			nTRDTrackletsTotal = pBlock.fSize / sizeof(AliHLTTRDTrackletWord);
		}
		else if (pBlock.fDataType == (AliHLTTRDDefinitions::fgkTRDMCTrackletDataType))
		{
			TRDtrackletsMC = reinterpret_cast<AliHLTTRDTrackletLabels*>(pBlock.fPtr);
			nTRDTrackletsMCTotal = pBlock.fSize / sizeof(AliHLTTRDTrackletLabels);
		}
	}
	
	std::vector<AliHLTTPCRawCluster> rawClusters[36];

	int nClustersTotal = 0;
	for (int slice = 0;slice < 36;slice++)
	{
		int nClustersSliceTotal = 0;
		rawClusters[slice].clear();
		for (int patch = 0; patch < 6; patch++)
		{
			if (clustersXYZ[slice][patch]) nClustersSliceTotal += clustersXYZ[slice][patch]->fCount;
		}
		fClusterData[slice].StartReading(slice, nClustersSliceTotal);
		AliHLTTPCCAClusterData::Data *pCluster = fClusterData[slice].Clusters();
		for (int patch = 0; patch < 6; patch++)
		{
			if (clustersXYZ[slice][patch] != NULL && clustersRaw[slice][patch] != NULL)
			{
				const AliHLTTPCClusterXYZData &clXYZ = *clustersXYZ[slice][patch];
				const AliHLTTPCRawClusterData &clRaw = *clustersRaw[slice][patch];

				if (clXYZ.fCount != clRaw.fCount)
				{
					HLTError("Number of entries in raw and xyz clusters are not mached %d vs %d", clXYZ.fCount, clRaw.fCount);
					continue;
				}

				const int firstRow = AliHLTTPCCAGeometry::GetFirstRow(patch);
				for (int ic = 0; ic < clXYZ.fCount; ic++)
				{
					const AliHLTTPCClusterXYZ &c = clXYZ.fClusters[ic];
					const AliHLTTPCRawCluster &cRaw = clRaw.fClusters[ic];
					if (fabs(c.GetZ()) > 300) continue;
					if (c.GetX() < 1.f) continue; // cluster xyz position was not calculated for whatever reason
					pCluster->fId = AliHLTTPCCAGeometry::CreateClusterID(slice, patch, ic);
					pCluster->fX = c.GetX();
					pCluster->fY = c.GetY();
					pCluster->fZ = c.GetZ();
					pCluster->fRow = firstRow + cRaw.GetPadRow();
					pCluster->fFlags = cRaw.GetFlags();
					if (cRaw.GetSigmaPad2() < kAlmost0 || cRaw.GetSigmaTime2() < kAlmost0) pCluster->fFlags |= AliHLTTPCGMMergedTrackHit::flagSingle;
					pCluster->fAmp = cRaw.GetCharge();
#ifdef GPUCA_FULL_CLUSTERDATA
					pCluster->fPad = cRaw.GetPad();
					pCluster->fTime = cRaw.GetTime();
					pCluster->fAmpMax = cRaw.GetQMax();
					pCluster->fSigmaPad2 = cRaw.GetSigmaPad2();
					pCluster->fSigmaTime2 = cRaw.GetSigmaTime2();
#endif
					AliHLTTPCRawCluster tmp = cRaw;
					tmp.fPadRow += firstRow;
					rawClusters[slice].emplace_back(tmp);
					
					pCluster++;
				}
			}
		}
		fClusterData[slice].SetNumberOfClusters(pCluster - fClusterData[slice].Clusters());
		nClustersTotal += fClusterData[slice].NumberOfClusters();
		HLTDebug("Read %d->%d hits for slice %d", nClustersSliceTotal, fClusterData[slice].NumberOfClusters(), slice);
	}

	if (nClustersTotal < 100) return (0);
	fRec->ClearIOPointers();

	for (int i = 0;i < 36;i++)
	{
		fRec->mIOPtrs.nClusterData[i] = fClusterData[i].NumberOfClusters();
		fRec->mIOPtrs.clusterData[i] = fClusterData[i].Clusters();
		fRec->mIOPtrs.nRawClusters[i] = fClusterData[i].NumberOfClusters();
		fRec->mIOPtrs.rawClusters[i] = rawClusters[i].data();
		HLTDebug("Slice %d - Clusters %d", i, (int) fClusterData[i].NumberOfClusters());
	}

	std::vector<AliHLTTPCClusterMCLabel> labels;
	std::vector<AliHLTTPCCAMCInfo> mcInfo;

	if (labelsPresent)
	{
		//Write cluster labels
		for (int iSlice = 0; iSlice < 36; iSlice++)
		{
			AliHLTTPCCAClusterData::Data *pCluster = fClusterData[iSlice].Clusters();
			for (int iPatch = 0; iPatch < 6; iPatch++)
			{
				if (clusterLabels[iSlice][iPatch] == NULL || clustersXYZ[iSlice][iPatch] == NULL || clusterLabels[iSlice][iPatch]->fCount != clustersXYZ[iSlice][iPatch]->fCount) continue;
				const AliHLTTPCClusterXYZData &clXYZ = *clustersXYZ[iSlice][iPatch];
				for (int ic = 0; ic < clXYZ.fCount; ic++)
				{
					if (pCluster->fId != AliHLTTPCCAGeometry::CreateClusterID(iSlice, iPatch, ic)) continue;
					pCluster->fId = labels.size();
					labels.push_back(clusterLabels[iSlice][iPatch]->fLabels[ic]);
					pCluster++;
				}
			}
		}
		
		if (labels.size() != nClustersTotal)
		{
			HLTFatal("Error getting cluster MC labels (%d labels, %d clusters)", (int) labels.size(), nClustersTotal);
			return(-EINVAL);
		}
		
		fRec->mIOPtrs.nMCLabelsTPC = labels.size();
		fRec->mIOPtrs.mcLabelsTPC = labels.data();
		HLTDebug("Number of mc labels %d", (int) labels.size());
		
		//Write MC tracks
		bool OK = false;
		do
		{
			AliRunLoader *rl = AliRunLoader::Instance();
			if (rl == NULL)
			{
				HLTFatal("error: RL");
				break;
			}

			rl->LoadKinematics();
			rl->LoadTrackRefs();

			int nTracks = rl->GetHeader()->GetNtrack();
			mcInfo.resize(nTracks);

			AliStack *stack = rl->Stack();
			if (stack == NULL)
			{
				HLTFatal("error: stack");
				break;
			}
			TTree *TR = rl->TreeTR();
			if (TR == NULL)
			{
				HLTFatal("error: TR");
				break;
			}
			TBranch *branch = TR->GetBranch("TrackReferences");
			if (branch == NULL)
			{
				HLTFatal("error: branch");
				break;
			}

			int nPrimaries = stack->GetNprimary();

			std::vector<AliTrackReference *> trackRefs(nTracks, NULL);
			TClonesArray *tpcRefs = NULL;
			branch->SetAddress(&tpcRefs);
			int nr = TR->GetEntries();
			for (int r = 0; r < nr; r++)
			{
				TR->GetEvent(r);
				for (int i = 0; i < tpcRefs->GetEntriesFast(); i++)
				{
					AliTrackReference *tpcRef = (AliTrackReference *) tpcRefs->UncheckedAt(i);
					if (tpcRef->DetectorId() != AliTrackReference::kTPC) continue;
					if (tpcRef->Label() < 0 || tpcRef->Label() >= nTracks)
					{
						HLTFatal("Invalid reference %d / %d", tpcRef->Label(), nTracks);
						continue;
					}
					if (trackRefs[tpcRef->Label()] != NULL) continue;
					trackRefs[tpcRef->Label()] = new AliTrackReference(*tpcRef);
				}
			}

			memset(mcInfo.data(), 0, nTracks * sizeof(mcInfo[0]));

			for (int i = 0; i < nTracks; i++)
			{
				mcInfo[i].fPID = -100;
				TParticle *particle = (TParticle *) stack->Particle(i);
				if (particle == NULL) continue;
				if (particle->GetPDG() == NULL) continue;

				int charge = (int) particle->GetPDG()->Charge();
				int prim = stack->IsPhysicalPrimary(i);
				int hasPrimDaughter = particle->GetFirstDaughter() != -1 && particle->GetFirstDaughter() < nPrimaries;

				mcInfo[i].fCharge = charge;
				mcInfo[i].fPrim = prim;
				mcInfo[i].fPrimDaughters = hasPrimDaughter;
				mcInfo[i].fGenRadius = sqrt(particle->Vx() * particle->Vx() + particle->Vy() * particle->Vy() + particle->Vz() * particle->Vz());

				Int_t pid = -1;
				if (TMath::Abs(particle->GetPdgCode()) == kElectron) pid = 0;
				if (TMath::Abs(particle->GetPdgCode()) == kMuonMinus) pid = 1;
				if (TMath::Abs(particle->GetPdgCode()) == kPiPlus) pid = 2;
				if (TMath::Abs(particle->GetPdgCode()) == kKPlus) pid = 3;
				if (TMath::Abs(particle->GetPdgCode()) == kProton) pid = 4;
				mcInfo[i].fPID = pid;

				AliTrackReference *ref = trackRefs[i];
				if (ref)
				{
					mcInfo[i].fX = ref->X();
					mcInfo[i].fY = ref->Y();
					mcInfo[i].fZ = ref->Z();
					mcInfo[i].fPx = ref->Px();
					mcInfo[i].fPy = ref->Py();
					mcInfo[i].fPz = ref->Pz();
				}

				//if (ref) printf("Particle %d: Charge %d, Prim %d, PrimDaughter %d, Pt %f %f ref %p\n", i, charge, prim, hasPrimDaughter, ref->Pt(), particle->Pt(), ref);
			}
			for (int i = 0; i < nTracks; i++)
				delete trackRefs[i];

			OK = true;
		} while (false);

		if (!OK)
		{
			HLTFatal("Error accessing MC data");
			return(-EINVAL);
		}
		
		fRec->mIOPtrs.nMCInfosTPC = mcInfo.size();
		fRec->mIOPtrs.mcInfosTPC = mcInfo.data();
		HLTDebug("Number of MC infos: %d", (int) mcInfo.size());
	}
	
	fRec->mIOPtrs.nTRDTracklets = nTRDTrackletsTotal;
	fRec->mIOPtrs.trdTracklets = TRDtracklets;
    fRec->mIOPtrs.nTRDTrackletsMC = nTRDTrackletsMCTotal;
	fRec->mIOPtrs.trdTrackletsMC = TRDtrackletsMC;
	HLTDebug("Number of TRD tracklets: %d", (int) nTRDTrackletsTotal);
	
	static int nEvent = 0;
	char filename[256];
	std::ofstream out;

	if (nEvent == 0)
	{
		std::unique_ptr<ali_tpc_common::tpc_fast_transformation::TPCFastTransform> fFastTransformIRS(new ali_tpc_common::tpc_fast_transformation::TPCFastTransform);
		long TimeStamp = GetTimeStamp();
		if (fIsMC && !fRecParam->GetUseCorrectionMap()) TimeStamp = 0;
		if (fFastTransformManager->create(*fFastTransformIRS, fCalib->GetTransform(), TimeStamp))
		{
			HLTFatal("Initialisation of Fast Transformation failed with error %s", fFastTransformManager->getLastError());
		}
		fRec->SetTPCFastTransform(std::move(fFastTransformIRS));
		
		fRec->SetSettings(fSolenoidBz);
		fRec->DumpSettings();
	}

	sprintf(filename, GPUCA_EVDUMP_FILE ".%d.dump", nEvent++);
	fRec->DumpData(filename);
	return (0);
}
