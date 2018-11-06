#include <cstring>
#include <stdio.h>

#ifdef WIN32
#include <windows.h>
#include <winbase.h>
#else
#include <dlfcn.h>
#endif

#include "AliGPUReconstruction.h"

#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCASliceOutTrack.h"
#include "AliHLTTPCCASliceOutCluster.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCGMMergedTrackHit.h"
#include "AliHLTTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCCAMCInfo.h"
#include "AliHLTTRDTrack.h"
#include "AliHLTTRDTracker.h"

#define HLTCA_LOGGING_PRINTF
#include "AliCAGPULogging.h"

static constexpr char DUMP_HEADER[] = "CAv1";

AliGPUReconstruction::AliGPUReconstruction() : mIOPtrs(), mIOMem(), mTRDTracker(new AliHLTTRDTracker), mTPCTracker(nullptr)
{
	mTRDTracker->Init();
}

AliGPUReconstruction::~AliGPUReconstruction()
{
	mTRDTracker.reset();
	mTPCTracker.reset();
}

AliGPUReconstruction::InOutMemory::InOutMemory() : mcLabelsTPC(), mcInfosTPC(), mergedTracks(), mergedTrackHits(), trdTracks(), trdTracklets()
{}
	
AliGPUReconstruction::InOutMemory::~InOutMemory()
{}

void AliGPUReconstruction::ClearIOPointers()
{
	std::memset((void*) &mIOPtrs, 0, sizeof(mIOPtrs));
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		mIOMem.clusterData[i].reset();
		mIOMem.sliceOutTracks[i].reset();
		mIOMem.sliceOutClusters[i].reset();
	}
	mIOMem.mcLabelsTPC.reset();
	mIOMem.mcInfosTPC.reset();
	mIOMem.mergedTracks.reset();
	mIOMem.mergedTrackHits.reset();
	mIOMem.trdTracks.reset();
	mIOMem.trdTracklets.reset();
}

void AliGPUReconstruction::DumpData(const char* filename)
{
	FILE* fp = fopen(filename, "w+b");
	if (fp == nullptr) return;
	fwrite(DUMP_HEADER, 1, strlen(DUMP_HEADER), fp);
	DumpData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, InOutPointerType::CLUSTER_DATA);
	DumpData(fp, mIOPtrs.sliceOutTracks, mIOPtrs.nSliceOutTracks, InOutPointerType::SLICE_OUT_TRACK);
	DumpData(fp, mIOPtrs.sliceOutClusters, mIOPtrs.nSliceOutClusters, InOutPointerType::SLICE_OUT_CLUSTER);
	DumpData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, InOutPointerType::MC_LABEL_TPC);
	DumpData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, InOutPointerType::MC_INFO_TPC);
	DumpData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, InOutPointerType::MERGED_TRACK);
	DumpData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
	DumpData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, InOutPointerType::TRD_TRACK);
	DumpData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, InOutPointerType::TRD_TRACKLET);
	fclose(fp);
}

template <class T> void AliGPUReconstruction::DumpData(FILE* fp, T** entries, unsigned int* num, InOutPointerType type)
{
	int count;
	if (type == CLUSTER_DATA || type == SLICE_OUT_TRACK || type == SLICE_OUT_CLUSTER) count = NSLICES;
	else count = 1;
	unsigned int numTotal = 0;
	for (int i = 0;i < count;i++) numTotal += num[i];
	if (numTotal == 0) return;
	fwrite(&type, sizeof(type), 1, fp);
	for (int i = 0;i < count;i++)
	{
		fwrite(&num[i], sizeof(num[i]), 1, fp);
		if (num[i])
		{
			fwrite(entries[i], sizeof(*entries[i]), num[i], fp);
		}
	}
}

int AliGPUReconstruction::ReadData(const char* filename)
{
	ClearIOPointers();
	FILE* fp = fopen(filename, "rb");
	if (fp == nullptr) return(1);
	char buf[strlen(DUMP_HEADER)] = "";
	size_t r = fread(buf, 1, strlen(DUMP_HEADER), fp);
	(void) r;
	if (strncmp(DUMP_HEADER, buf, strlen(DUMP_HEADER)))
	{
		return(1);
	}
		
	ReadData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, mIOMem.clusterData, InOutPointerType::CLUSTER_DATA);
	ReadData(fp, mIOPtrs.sliceOutTracks, mIOPtrs.nSliceOutTracks, mIOMem.sliceOutTracks, InOutPointerType::SLICE_OUT_TRACK);
	ReadData(fp, mIOPtrs.sliceOutClusters, mIOPtrs.nSliceOutClusters, mIOMem.sliceOutClusters, InOutPointerType::SLICE_OUT_CLUSTER);
	ReadData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, &mIOMem.mcLabelsTPC, InOutPointerType::MC_LABEL_TPC);
	ReadData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, &mIOMem.mcInfosTPC, InOutPointerType::MC_INFO_TPC);
	ReadData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, &mIOMem.mergedTracks, InOutPointerType::MERGED_TRACK);
	ReadData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, &mIOMem.mergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
	ReadData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, &mIOMem.trdTracks, InOutPointerType::TRD_TRACK);
	ReadData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, &mIOMem.trdTracklets, InOutPointerType::TRD_TRACKLET);
	fclose(fp);
	
	return(0);
}

template <class T> void AliGPUReconstruction::ReadData(FILE* fp, T** entries, unsigned int* num, std::unique_ptr<T[]>* mem, InOutPointerType type)
{
	if (feof(fp)) return;
	InOutPointerType inType;
	size_t r, pos = ftell(fp);
	r = fread(&inType, sizeof(inType), 1, fp);
	if (r != 1 || inType != type)
	{
		fseek(fp, pos, SEEK_SET);
		return;
	}
	
	int count;
	if (type == CLUSTER_DATA || type == SLICE_OUT_TRACK || type == SLICE_OUT_CLUSTER) count = NSLICES;
	else count = 1;
	unsigned int numTotal = 0;
	for (int i = 0;i < count;i++)
	{
		r = fread(&num[i], sizeof(num[i]), 1, fp);
		AllocateIOMemoryHelper(num[i], entries[i], mem[i]);
		if (num[i]) r = fread(entries[i], sizeof(*entries[i]), num[i], fp);
		numTotal += num[i];
	}
	(void) r;
	//printf("Read %d %s\n", numTotal, IOTYPENAMES[type]);
}

template <class T> void AliGPUReconstruction::AllocateIOMemoryHelper(unsigned int n, T* &ptr, std::unique_ptr<T[]> &u)
{
	u.reset(n ? (ptr = new T[n]) : nullptr);
}

void AliGPUReconstruction::AllocateIOMemory()
{
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		AllocateIOMemoryHelper(mIOPtrs.nClusterData[i], mIOPtrs.clusterData[i], mIOMem.clusterData[i]);
		AllocateIOMemoryHelper(mIOPtrs.nSliceOutTracks[i], mIOPtrs.sliceOutTracks[i], mIOMem.sliceOutTracks[i]);
		AllocateIOMemoryHelper(mIOPtrs.nSliceOutClusters[i], mIOPtrs.sliceOutClusters[i], mIOMem.sliceOutClusters[i]);
	}
	AllocateIOMemoryHelper(mIOPtrs.nMCLabelsTPC, mIOPtrs.mcLabelsTPC, mIOMem.mcLabelsTPC);
	AllocateIOMemoryHelper(mIOPtrs.nMCInfosTPC, mIOPtrs.mcInfosTPC, mIOMem.mcInfosTPC);
	AllocateIOMemoryHelper(mIOPtrs.nMergedTracks, mIOPtrs.mergedTracks, mIOMem.mergedTracks);
	AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHits, mIOMem.mergedTrackHits);
	AllocateIOMemoryHelper(mIOPtrs.nTRDTracks, mIOPtrs.trdTracks, mIOMem.trdTracks);
	AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdTracklets, mIOMem.trdTracklets);
}

int AliGPUReconstruction::RunTRDTracking()
{
	std::vector< HLTTRDTrack > tracksTPC;
	std::vector< int > tracksTPCLab;
	std::vector< int > tracksTPCId;

	for (unsigned int i = 0;i < mIOPtrs.nMergedTracks;i++)
	{
		const AliHLTTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
		if (!trk.OK()) continue;
		tracksTPC.emplace_back(trk.OuterParam());
		tracksTPCId.push_back(i);
		tracksTPCLab.push_back(-1);
	}
	
	mTRDTracker->Reset();
	mTRDTracker->StartLoadTracklets(mIOPtrs.nTRDTracklets);

	for (unsigned int iTracklet = 0;iTracklet < mIOPtrs.nTRDTracklets;++iTracklet)
	{
		mTRDTracker->LoadTracklet(mIOPtrs.trdTracklets[iTracklet]);
	}

	mTRDTracker->DoTracking(&(tracksTPC[0]), &(tracksTPCLab[0]), tracksTPC.size());
	
	printf("TRD Tracker reconstructed %d tracks\n", mTRDTracker->NTracks());
	
	return 0;
}

AliGPUReconstruction* AliGPUReconstruction::CreateInstance(DeviceType type, bool forceType)
{
	if (type == DeviceType::CPU)
	{
		return new AliGPUReconstruction;
	}
	else if (type == DeviceType::CUDA)
	{
		return sLibCUDA.GetPtr();
	}
	else if (type == DeviceType::HIP)
	{
		return sLibHIP.GetPtr();
	}
	else if (type == DeviceType::OCL)
	{
		return sLibOCL.GetPtr();
	}
	
	return nullptr;
}

AliGPUReconstruction* AliGPUReconstruction::CreateInstance(const char* type, bool forceType)
{
	for (unsigned int i = 1;i < sizeof(DEVICE_TYPE_NAMES) / sizeof(DEVICE_TYPE_NAMES[0]);i++)
	{
		if (strcmp(DEVICE_TYPE_NAMES[i], type) == 0)
		{
			return CreateInstance(i, forceType);
		}
	}
	printf("Invalid device type provided\n");
	return nullptr;
}

#ifdef WIN32
#define LIBRARY_EXTENSION ".dll"
#else
#define LIBRARY_EXTENSION ".so"
#endif

#if defined(HLTCA_BUILD_ALIROOT_LIB)
#define LIBRARY_PREFIX "Ali"
#elif defined(HLTCA_BUILD_O2_LIB)
#define LIBRARY_PREFIX "O2"
#else
#define LIBRARY_PREFIX ""
#endif

AliGPUReconstruction::LibraryLoader AliGPUReconstruction::sLibCUDA("lib" LIBRARY_PREFIX "TPCCAGPUTracking" "CUDA" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "CUDA");
AliGPUReconstruction::LibraryLoader AliGPUReconstruction::sLibHIP("lib" LIBRARY_PREFIX "TPCCAGPUTracking" "HIP" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "HIP");
AliGPUReconstruction::LibraryLoader AliGPUReconstruction::sLibOCL("lib" LIBRARY_PREFIX "TPCCAGPUTracking" "OCL" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "OCL");

AliGPUReconstruction::LibraryLoader::LibraryLoader(const char* lib, const char* func) : mLibName(lib), mFuncName(func), mGPULib(nullptr), mGPUEntry(nullptr)
{
	LoadLibrary();
}

AliGPUReconstruction::LibraryLoader::~LibraryLoader()
{
	CloseLibrary();
}

int AliGPUReconstruction::LibraryLoader::LoadLibrary()
{
	#ifdef WIN32
		HMODULE hGPULib;
		hGPULib = LoadLibraryEx(mLibName, NULL, NULL);
	#else
		void* hGPULib;
		hGPULib = dlopen(mLibName, RTLD_NOW);
	#endif

	if (hGPULib == NULL)
	{
		#ifndef WIN32
			CAGPUImportant("The following error occured during dlopen: %s", dlerror());
		#endif
		CAGPUError("Error Opening cagpu library for GPU Tracker (%s)", mLibName);
		return 1;
	}
	else
	{
		#ifdef WIN32
			FARPROC createFunc = GetProcAddress(hGPULib, mFuncName);
		#else
			void* createFunc = (void*) dlsym(hGPULib, mFuncName);
		#endif
		if (createFunc == NULL)
		{
			CAGPUError("Error fetching entry function in GPU library\n");
	#ifdef WIN32
			FreeLibrary(hGPULib);
	#else
			dlclose(hGPULib);
	#endif
			return 1;
		}
		else
		{
			mGPULib = (void*) (size_t) hGPULib;
			mGPUEntry = (void*) createFunc;
			CAGPUInfo("GPU Tracker library loaded and GPU tracker object created sucessfully");
		}
	}
	return 0;
}

AliGPUReconstruction* AliGPUReconstruction::LibraryLoader::GetPtr()
{
	if (mGPUEntry == nullptr) return nullptr;
	AliGPUReconstruction* (*tmp)() = (AliGPUReconstruction* (*)()) mGPUEntry;
	return tmp();
}

int AliGPUReconstruction::LibraryLoader::CloseLibrary()
{
	if (mGPUEntry == nullptr) return 1;
	#ifdef WIN32
		HMODULE hGPULib = (HMODULE) (size_t) mGPULib;
		FreeLibrary(hGPULib);
	#else
		void* hGPULib = mGPULib;
		dlclose(hGPULib);
	#endif
	mGPULib = nullptr;
	mGPUEntry = nullptr;
	return 0;
}
