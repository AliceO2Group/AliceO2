#include <cstring>
#include <stdio.h>
#include <mutex>
#include <string>

#ifdef WIN32
#include <windows.h>
#include <winbase.h>
#else
#include <dlfcn.h>
#endif

#ifdef HLTCA_HAVE_OPENMP
#include <omp.h>
#endif

#include "AliGPUReconstruction.h"
#include "AliGPUReconstructionConvert.h"

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
#include "TPCFastTransform.h"
#include "AliHLTTPCRawCluster.h"
#include "ClusterNativeAccessExt.h"
#include "AliHLTTRDTrackletLabels.h"
#include "AliGPUCADisplay.h"
#include "AliGPUCAQA.h"

#ifdef HLTCA_STANDALONE
#include <omp.h>
#ifdef WIN32
#include <conio.h>
#else
#include <pthread.h>
#include <unistd.h>
#include "../cmodules/linux_helpers.h"
#endif
#endif

#define HLTCA_LOGGING_PRINTF
#include "AliCAGPULogging.h"

#ifdef HAVE_O2HEADERS
#include "ITStracking/TrackerTraitsCPU.h"
#include "TRDBase/TRDGeometryFlat.h"
#else
namespace o2 { namespace ITS { class TrackerTraits {}; class TrackerTraitsCPU : public TrackerTraits {}; }}
namespace o2 { namespace trd { struct TRDGeometryFlat {}; }}
#endif
using namespace o2::ITS;
using namespace o2::TPC;
using namespace o2::trd;

constexpr const char* const AliGPUReconstruction::DEVICE_TYPE_NAMES[];
constexpr const char* const AliGPUReconstruction::GEOMETRY_TYPE_NAMES[];
constexpr const char* const AliGPUReconstruction::IOTYPENAMES[];
constexpr AliGPUReconstruction::GeometryType AliGPUReconstruction::geometryType;

static constexpr unsigned int DUMP_HEADER_SIZE = 4;
static constexpr char DUMP_HEADER[DUMP_HEADER_SIZE + 1] = "CAv1";

AliGPUReconstruction::AliGPUReconstruction(const AliGPUCASettingsProcessing& cfg) : mTRDTracker(new AliHLTTRDTracker), mITSTrackerTraits(nullptr), mTPCFastTransform(nullptr), mClusterNativeAccess(new ClusterNativeAccessExt)
{
	mProcessingSettings = cfg;
	mDeviceProcessingSettings.SetDefaults();
	mEventSettings.SetDefaults();
	mParam.SetDefaults(&mEventSettings);
	if (mProcessingSettings.deviceType == CPU)
	{
		mITSTrackerTraits.reset(new o2::ITS::TrackerTraitsCPU);
	}
	memset(mSliceOutput, 0, sizeof(mSliceOutput));
}

AliGPUReconstruction::~AliGPUReconstruction()
{
	//Reset these explicitly before the destruction of other members unloads the library
	mTRDTracker.reset();
	mITSTrackerTraits.reset();
}

int AliGPUReconstruction::Init()
{
	mTRDTracker->Init((AliHLTTRDGeometry*) mTRDGeometry.get()); //Cast is safe, we just add some member functions
	if (mDeviceProcessingSettings.debugLevel >= 4)
	{
		mDebugFile.open(IsGPU() ? "GPU.out" : "CPU.out");
	}
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		mTPCSliceTrackersCPU[i].Initialize(&mParam, i);
		mTPCSliceTrackersCPU[i].SetGPUDebugOutput(&mDebugFile);
		mTPCSliceTrackersCPU[i].SetAliGPUReconstruction(this);
	}
	mTPCMergerCPU.Initialize(this);
	if (InitDevice()) return 1;
	mInitialized = true;
	return 0;
}

int AliGPUReconstruction::Finalize()
{
	if (mDeviceProcessingSettings.debugLevel >= 4) mDebugFile.close();
	ExitDevice();
	mInitialized = false;
	return 0;
}

int AliGPUReconstruction::InitDevice() {return 0;}
int AliGPUReconstruction::ExitDevice() {return 0;}

AliGPUReconstruction::InOutMemory::InOutMemory()
{}
	
AliGPUReconstruction::InOutMemory::~InOutMemory()
{}

void AliGPUReconstruction::ClearIOPointers()
{
	std::memset((void*) &mIOPtrs, 0, sizeof(mIOPtrs));
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		mIOMem.clusterData[i].reset();
		mIOMem.rawClusters[i].reset();
		mIOMem.sliceOutTracks[i].reset();
		mIOMem.sliceOutClusters[i].reset();
	}
	for (unsigned int i = 0;i < NSLICES * HLTCA_ROW_COUNT;i++)
	{
		mIOMem.clustersNative[i].reset();
	}
	mIOMem.mcLabelsTPC.reset();
	mIOMem.mcInfosTPC.reset();
	mIOMem.mergedTracks.reset();
	mIOMem.mergedTrackHits.reset();
	mIOMem.trdTracks.reset();
	mIOMem.trdTracklets.reset();
	mIOMem.trdTrackletsMC.reset();
}

void AliGPUReconstruction::DumpData(const char* filename)
{
	FILE* fp = fopen(filename, "w+b");
	if (fp == nullptr) return;
	fwrite(DUMP_HEADER, 1, DUMP_HEADER_SIZE, fp);
	fwrite(&geometryType, sizeof(geometryType), 1, fp);
	DumpData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, InOutPointerType::CLUSTER_DATA);
	DumpData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, InOutPointerType::RAW_CLUSTERS);
	if (mIOPtrs.clustersNative) DumpData(fp, &mIOPtrs.clustersNative->clusters[0][0], &mIOPtrs.clustersNative->nClusters[0][0], InOutPointerType::CLUSTERS_NATIVE);
	DumpData(fp, mIOPtrs.sliceOutTracks, mIOPtrs.nSliceOutTracks, InOutPointerType::SLICE_OUT_TRACK);
	DumpData(fp, mIOPtrs.sliceOutClusters, mIOPtrs.nSliceOutClusters, InOutPointerType::SLICE_OUT_CLUSTER);
	DumpData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, InOutPointerType::MC_LABEL_TPC);
	DumpData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, InOutPointerType::MC_INFO_TPC);
	DumpData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, InOutPointerType::MERGED_TRACK);
	DumpData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
	DumpData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, InOutPointerType::TRD_TRACK);
	DumpData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, InOutPointerType::TRD_TRACKLET);
	DumpData(fp, &mIOPtrs.trdTrackletsMC, &mIOPtrs.nTRDTrackletsMC, InOutPointerType::TRD_TRACKLET_MC);
	fclose(fp);
}

template <class T> void AliGPUReconstruction::DumpData(FILE* fp, const T* const* entries, const unsigned int* num, InOutPointerType type)
{
	int count;
	if (type == CLUSTER_DATA || type == SLICE_OUT_TRACK || type == SLICE_OUT_CLUSTER || type == RAW_CLUSTERS) count = NSLICES;
	else if (type == CLUSTERS_NATIVE) count = NSLICES * HLTCA_ROW_COUNT;
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
	char buf[DUMP_HEADER_SIZE + 1] = "";
	size_t r = fread(buf, 1, DUMP_HEADER_SIZE, fp);
	if (strncmp(DUMP_HEADER, buf, DUMP_HEADER_SIZE))
	{
		printf("Invalid file header\n");
		return -1;
	}
	GeometryType geo;
	r = fread(&geo, sizeof(geo), 1, fp);
	if (geo != geometryType)
	{
		printf("File has invalid geometry (%s v.s. %s)\n", GEOMETRY_TYPE_NAMES[geo], GEOMETRY_TYPE_NAMES[geometryType]);
		return 1;
	}
	(void) r;
	ReadData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, mIOMem.clusterData, InOutPointerType::CLUSTER_DATA);
	int nClustersTotal = 0;
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		for (unsigned int j = 0;j < mIOPtrs.nClusterData[i];j++)
		{
			mIOMem.clusterData[i][j].fId = nClustersTotal++;
		}
	}
	ReadData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, mIOMem.rawClusters, InOutPointerType::RAW_CLUSTERS);
	mIOPtrs.clustersNative = ReadData<ClusterNative>(fp, (const ClusterNative**) &mClusterNativeAccess->clusters[0][0], &mClusterNativeAccess->nClusters[0][0], mIOMem.clustersNative, InOutPointerType::CLUSTERS_NATIVE) ? mClusterNativeAccess.get() : nullptr;
	ReadData(fp, mIOPtrs.sliceOutTracks, mIOPtrs.nSliceOutTracks, mIOMem.sliceOutTracks, InOutPointerType::SLICE_OUT_TRACK);
	ReadData(fp, mIOPtrs.sliceOutClusters, mIOPtrs.nSliceOutClusters, mIOMem.sliceOutClusters, InOutPointerType::SLICE_OUT_CLUSTER);
	ReadData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, &mIOMem.mcLabelsTPC, InOutPointerType::MC_LABEL_TPC);
	ReadData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, &mIOMem.mcInfosTPC, InOutPointerType::MC_INFO_TPC);
	ReadData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, &mIOMem.mergedTracks, InOutPointerType::MERGED_TRACK);
	ReadData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, &mIOMem.mergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
	ReadData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, &mIOMem.trdTracks, InOutPointerType::TRD_TRACK);
	ReadData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, &mIOMem.trdTracklets, InOutPointerType::TRD_TRACKLET);
	ReadData(fp, &mIOPtrs.trdTrackletsMC, &mIOPtrs.nTRDTrackletsMC, &mIOMem.trdTrackletsMC, InOutPointerType::TRD_TRACKLET_MC);
	fclose(fp);
	
	return(0);
}

template <class T> size_t AliGPUReconstruction::ReadData(FILE* fp, const T** entries, unsigned int* num, std::unique_ptr<T[]>* mem, InOutPointerType type)
{
	if (feof(fp)) return 0;
	InOutPointerType inType;
	size_t r, pos = ftell(fp);
	r = fread(&inType, sizeof(inType), 1, fp);
	if (r != 1 || inType != type)
	{
		fseek(fp, pos, SEEK_SET);
		return 0;
	}
	
	int count;
	if (type == CLUSTER_DATA || type == SLICE_OUT_TRACK || type == SLICE_OUT_CLUSTER || type == RAW_CLUSTERS) count = NSLICES;
	else if (type == CLUSTERS_NATIVE) count = NSLICES * HLTCA_ROW_COUNT;
	else count = 1;
	size_t numTotal = 0;
	for (int i = 0;i < count;i++)
	{
		r = fread(&num[i], sizeof(num[i]), 1, fp);
		AllocateIOMemoryHelper(num[i], entries[i], mem[i]);
		if (num[i]) r = fread(mem[i].get(), sizeof(*entries[i]), num[i], fp);
		numTotal += num[i];
	}
	(void) r;
	printf("Read %d %s\n", (int) numTotal, IOTYPENAMES[type]);
	return numTotal;
}

void AliGPUReconstruction::DumpSettings(const char* dir)
{
	std::string f;
	f = dir;
	f += "settings.dump";
	DumpStructToFile(&mEventSettings, f.c_str());
	f = dir;
	f += "tpctransform.dump";
	if (mTPCFastTransform != nullptr) DumpFlatObjectToFile(mTPCFastTransform.get(), f.c_str());
	f = dir;
	f += "trdgeometry.dump";
	if (mTRDGeometry != nullptr) DumpStructToFile(mTRDGeometry.get(), f.c_str());
	
}

void AliGPUReconstruction::ReadSettings(const char* dir)
{
	std::string f;
	f = dir;
	f += "settings.dump";
	mEventSettings.SetDefaults();
	mParam.UpdateEventSettings(&mEventSettings);
	ReadStructFromFile(f.c_str(), &mEventSettings);
	f = dir;
	f += "tpctransform.dump";
	mTPCFastTransform = ReadFlatObjectFromFile<TPCFastTransform>(f.c_str());
	f = dir;
	f += "trdgeometry.dump";
	mTRDGeometry = ReadStructFromFile<o2::trd::TRDGeometryFlat>(f.c_str());
}

template <class T> void AliGPUReconstruction::AllocateIOMemoryHelper(unsigned int n, const T* &ptr, std::unique_ptr<T[]> &u)
{
	if (n == 0)
	{
		u.reset(nullptr);
		return;
	}
	u.reset(new T[n]);
	ptr = u.get();
}

void AliGPUReconstruction::AllocateIOMemory()
{
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		AllocateIOMemoryHelper(mIOPtrs.nClusterData[i], mIOPtrs.clusterData[i], mIOMem.clusterData[i]);
		AllocateIOMemoryHelper(mIOPtrs.nRawClusters[i], mIOPtrs.rawClusters[i], mIOMem.rawClusters[i]);
		AllocateIOMemoryHelper(mIOPtrs.nSliceOutTracks[i], mIOPtrs.sliceOutTracks[i], mIOMem.sliceOutTracks[i]);
		AllocateIOMemoryHelper(mIOPtrs.nSliceOutClusters[i], mIOPtrs.sliceOutClusters[i], mIOMem.sliceOutClusters[i]);
	}
	AllocateIOMemoryHelper(mIOPtrs.nMCLabelsTPC, mIOPtrs.mcLabelsTPC, mIOMem.mcLabelsTPC);
	AllocateIOMemoryHelper(mIOPtrs.nMCInfosTPC, mIOPtrs.mcInfosTPC, mIOMem.mcInfosTPC);
	AllocateIOMemoryHelper(mIOPtrs.nMergedTracks, mIOPtrs.mergedTracks, mIOMem.mergedTracks);
	AllocateIOMemoryHelper(mIOPtrs.nMergedTrackHits, mIOPtrs.mergedTrackHits, mIOMem.mergedTrackHits);
	AllocateIOMemoryHelper(mIOPtrs.nTRDTracks, mIOPtrs.trdTracks, mIOMem.trdTracks);
	AllocateIOMemoryHelper(mIOPtrs.nTRDTracklets, mIOPtrs.trdTracklets, mIOMem.trdTracklets);
	AllocateIOMemoryHelper(mIOPtrs.nTRDTrackletsMC, mIOPtrs.trdTrackletsMC, mIOMem.trdTrackletsMC);
}

template <class T> void AliGPUReconstruction::DumpFlatObjectToFile(const T* obj, const char* file)
{
	FILE* fp = fopen(file, "w+b");
	if (fp == nullptr) return;
	size_t size[2] = {sizeof(*obj), obj->getFlatBufferSize()};
	fwrite(size, sizeof(size[0]), 2, fp);
	fwrite(obj, 1, size[0], fp);
	fwrite(obj->getFlatBufferPtr(), 1, size[1], fp);
	fclose(fp);
}

template <class T> std::unique_ptr<T> AliGPUReconstruction::ReadFlatObjectFromFile(const char* file)
{
	FILE* fp = fopen(file, "rb");
	if (fp == nullptr) return nullptr;
	size_t size[2], r;
	r = fread(size, sizeof(size[0]), 2, fp);
	if (r == 0 || size[0] != sizeof(T)) {fclose(fp); return nullptr;}
	std::unique_ptr<T> retVal(new T);
	std::unique_ptr<char[]> buf(new char[size[1]]);
	r = fread((void*) retVal.get(), 1, size[0], fp);
	r = fread(buf.get(), 1, size[1], fp);
	fclose(fp);
	printf("Read %d bytes from %s\n", (int) r, file);
	retVal->clearInternalBufferUniquePtr();
	retVal->setActualBufferAddress(buf.get());
	retVal->adoptInternalBuffer(std::move(buf));
	return std::move(retVal);
}

template <class T> void AliGPUReconstruction::DumpStructToFile(const T* obj, const char* file)
{
	FILE* fp = fopen(file, "w+b");
	if (fp == nullptr) return;
	size_t size = sizeof(*obj);
	fwrite(&size, sizeof(size), 1, fp);
	fwrite(obj, 1, size, fp);
	fclose(fp);
}

template <class T> std::unique_ptr<T> AliGPUReconstruction::ReadStructFromFile(const char* file)
{
	FILE* fp = fopen(file, "rb");
	if (fp == nullptr) return nullptr;
	size_t size, r;
	r = fread(&size, sizeof(size), 1, fp);
	if (r == 0 || size != sizeof(T)) {fclose(fp); return nullptr;}
	std::unique_ptr<T> newObj(new T);
	r = fread(newObj.get(), 1, size, fp);
	fclose(fp);
	printf("Read %d bytes from %s\n", (int) r, file);
	return std::move(newObj);
}

template <class T> void AliGPUReconstruction::ReadStructFromFile(const char* file, T* obj)
{
	FILE* fp = fopen(file, "rb");
	if (fp == nullptr) return;
	size_t size, r;
	r = fread(&size, sizeof(size), 1, fp);
	if (r == 0) {fclose(fp); return;}
	r = fread(obj, 1, size, fp);
	fclose(fp);
	printf("Read %d bytes from %s\n", (int) r, file);
}

void AliGPUReconstruction::ConvertNativeToClusterData()
{
	o2::TPC::ClusterNativeAccessFullTPC* tmp = mClusterNativeAccess.get();
	if (tmp != mIOPtrs.clustersNative)
	{
		*tmp = *mIOPtrs.clustersNative;
	}
	AliGPUReconstructionConvert::ConvertNativeToClusterData(mClusterNativeAccess.get(), mIOMem.clusterData, mIOPtrs.nClusterData, mTPCFastTransform.get(), mParam.continuousMaxTimeBin);
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		mIOPtrs.clusterData[i] = mIOMem.clusterData[i].get();
	}
	mIOPtrs.clustersNative = nullptr;
}

void AliGPUReconstruction::SetSettings(float solenoidBz)
{
	AliGPUCASettingsEvent ev;
	ev.SetDefaults();
	ev.solenoidBz = solenoidBz;
	SetSettings(&ev, nullptr, nullptr);
}
void AliGPUReconstruction::SetSettings(const AliGPUCASettingsEvent* settings, const AliGPUCASettingsRec* rec, const AliGPUCASettingsDeviceProcessing* proc)
{
	mEventSettings = *settings;
	if (proc) mDeviceProcessingSettings = *proc;
	mParam.SetDefaults(&mEventSettings, rec, proc);
}
void AliGPUReconstruction::LoadClusterErrors()
{
	mParam.LoadClusterErrors();
}
void AliGPUReconstruction::SetTPCFastTransform(std::unique_ptr<TPCFastTransform> tpcFastTransform)
{
	mTPCFastTransform = std::move(tpcFastTransform);
}
void AliGPUReconstruction::SetTRDGeometry(const o2::trd::TRDGeometryFlat& geo)
{
	mTRDGeometry.reset(new o2::trd::TRDGeometryFlat(geo));
}

int AliGPUReconstruction::RunStandalone()
{
	mStatNEvents++;
	
	const bool needQA = AliGPUCAQA::QAAvailable() && (mDeviceProcessingSettings.runQA || (mDeviceProcessingSettings.eventDisplay && mIOPtrs.nMCInfosTPC));
	if (needQA && mQA == nullptr)
	{
		mQA.reset(new AliGPUCAQA(this));
		if (mQA->InitQA()) return 1;
	}
	
#ifdef HLTCA_STANDALONE
	static HighResTimer timerTracking, timerMerger, timerQA;
	static int nCount = 0;
	if (mDeviceProcessingSettings.resetTimers)
	{
		timerTracking.Reset();
		timerMerger.Reset();
		timerQA.Reset();
		nCount = 0;
	}
	timerTracking.Start();
#endif

	RunTPCTrackingSlices();

#ifdef HLTCA_STANDALONE
	timerTracking.Stop();
#endif

#ifdef HLTCA_STANDALONE
	timerMerger.Start();
#endif
	mTPCMergerCPU.Clear();

	for (unsigned int i = 0; i < NSLICES; i++)
	{
		//printf("slice %d clusters %d tracks %d\n", i, fClusterData[i].NumberOfClusters(), mSliceOutput[i]->NTracks());
		mTPCMergerCPU.SetSliceData(i, mSliceOutput[i]);
	}

#ifdef HLTCA_GPU_MERGER
#endif
	RunTPCTrackingMerger();
#ifdef HLTCA_STANDALONE
	timerMerger.Stop();
#endif

#ifdef HLTCA_STANDALONE
	if (needQA)
	{
		timerQA.Start();
		mQA->RunQA(!mDeviceProcessingSettings.runQA);
		timerQA.Stop();
	}

	nCount++;
#ifndef HLTCA_BUILD_O2_LIB
	char nAverageInfo[16] = "";
	if (nCount > 1) sprintf(nAverageInfo, " (%d)", nCount);
	printf("Tracking Time: %'d us%s\n", (int) (1000000 * timerTracking.GetElapsedTime() / nCount), nAverageInfo);
	printf("Merging and Refit Time: %'d us\n", (int) (1000000 * timerMerger.GetElapsedTime() / nCount));
	if (mDeviceProcessingSettings.runQA) printf("QA Time: %'d us\n", (int) (1000000 * timerQA.GetElapsedTime() / nCount));
#endif

	if (mDeviceProcessingSettings.debugLevel >= 1)
	{
		const char *tmpNames[10] = {"Initialisation", "Neighbours Finder", "Neighbours Cleaner", "Starts Hits Finder", "Start Hits Sorter", "Weight Cleaner", "Tracklet Constructor", "Tracklet Selector", "Global Tracking", "Write Output"};

		for (int i = 0; i < 10; i++)
		{
			double time = 0;
			for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++)
			{
				time += mTPCSliceTrackersCPU[iSlice].GetTimer(i);
				mTPCSliceTrackersCPU[iSlice].ResetTimer(i);
			}
			time /= NSLICES;
#ifdef HLTCA_HAVE_OPENMP
			if (!IsGPU()) time /= omp_get_max_threads();
#endif

			printf("Execution Time: Task: %20s ", tmpNames[i]);
			printf("Time: %1.0f us", time * 1000000 / nCount);
			printf("\n");
		}
		printf("Execution Time: Task: %20s Time: %1.0f us\n", "Merger", timerMerger.GetElapsedTime() * 1000000. / nCount);
		if (!HLTCA_TIMING_SUM)
		{
			timerTracking.Reset();
			timerMerger.Reset();
			timerQA.Reset();
			nCount = 0;
		}
	}

	if (mDeviceProcessingSettings.eventDisplay)
	{
		if (mEventDisplay == nullptr)
		{
			mEventDisplay.reset(new AliGPUCADisplay(mDeviceProcessingSettings.eventDisplay, this, mQA.get()));
			mDeviceProcessingSettings.eventDisplay->StartDisplay();
		}
		else
		{
			mEventDisplay->ShowNextEvent();
		}

		while (kbhit()) getch();
		printf("Press key for next event!\n");

		int iKey;
		do
		{
#ifdef WIN32
			Sleep(10);
#else
			usleep(10000);
#endif
			iKey = kbhit() ? getch() : 0;
			if (iKey == 'q') mDeviceProcessingSettings.eventDisplay->displayControl = 2;
			else if (iKey == 'n') break;
			else if (iKey)
			{
				while (mDeviceProcessingSettings.eventDisplay->sendKey != 0)
				{
#ifdef WIN32
					Sleep(1);
#else
					usleep(1000);
#endif
				}
				mDeviceProcessingSettings.eventDisplay->sendKey = iKey;
			}
		} while (mDeviceProcessingSettings.eventDisplay->displayControl == 0);
		if (mDeviceProcessingSettings.eventDisplay->displayControl == 2)
		{
			mDeviceProcessingSettings.eventDisplay->DisplayExit();
			return (2);
		}
		mDeviceProcessingSettings.eventDisplay->displayControl = 0;
		printf("Loading next event\n");

		mEventDisplay->WaitForNextEvent();
	}
#endif
	return 0;
}

int AliGPUReconstruction::RunTPCTrackingSlices()
{
	bool error = false;
#ifdef HLTCA_STANDALONE
	int nLocalTracks = 0, nGlobalTracks = 0, nOutputTracks = 0, nLocalHits = 0, nGlobalHits = 0;
#endif
#ifdef HLTCA_HAVE_OPENMP
	if (mOutputControl.OutputType != AliGPUCAOutputControl::AllocateInternal && omp_get_max_threads() > 1)
	{
		CAGPUError("fOutputPtr must not be used with OpenMP\n");
		return(1);
	}
#pragma omp parallel for
#endif
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (error) continue;
		mClusterData[iSlice].SetClusterData(iSlice, mIOPtrs.nClusterData[iSlice], mIOPtrs.clusterData[iSlice]);
		if (mTPCSliceTrackersCPU[iSlice].ReadEvent(&mClusterData[iSlice]))
		{
			error = true;
			continue;
		}
		mTPCSliceTrackersCPU[iSlice].SetOutput(&mSliceOutput[iSlice]);
		mTPCSliceTrackersCPU[iSlice].Reconstruct();
		mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTracks = mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTracks;
		mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTrackHits = mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTrackHits;
		if (!mParam.rec.GlobalTracking)
		{
			mTPCSliceTrackersCPU[iSlice].ReconstructOutput();
#ifdef HLTCA_STANDALONE
			nOutputTracks += (*mTPCSliceTrackersCPU[iSlice].Output())->NTracks();
			nLocalTracks += mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTracks;
#endif
			if (!mDeviceProcessingSettings.eventDisplay)
			{
				mTPCSliceTrackersCPU[iSlice].SetupCommonMemory();
			}
		}
	}
	if (error) return(1);

	if (mParam.rec.GlobalTracking)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			int sliceLeft = (iSlice + (NSLICES / 2 - 1)) % (NSLICES / 2);
			int sliceRight = (iSlice + 1) % (NSLICES / 2);
			if (iSlice >= NSLICES / 2)
			{
				sliceLeft += NSLICES / 2;
				sliceRight += NSLICES / 2;
			}
			mTPCSliceTrackersCPU[iSlice].PerformGlobalTracking(mTPCSliceTrackersCPU[sliceLeft], mTPCSliceTrackersCPU[sliceRight], mTPCSliceTrackersCPU[sliceLeft].NMaxTracks(), mTPCSliceTrackersCPU[sliceRight].NMaxTracks());
		}
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			mTPCSliceTrackersCPU[iSlice].ReconstructOutput();
#ifdef HLTCA_STANDALONE
			//printf("Slice %d - Tracks: Local %d Global %d - Hits: Local %d Global %d\n", iSlice, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTracks, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTracks, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTrackHits, mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTrackHits);
			nLocalTracks += mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTracks;
			nGlobalTracks += mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTracks;
			nLocalHits += mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNLocalTrackHits;
			nGlobalHits += mTPCSliceTrackersCPU[iSlice].CommonMemory()->fNTrackHits;
			nOutputTracks += (*mTPCSliceTrackersCPU[iSlice].Output())->NTracks();
#endif
			if (!mDeviceProcessingSettings.eventDisplay)
			{
				mTPCSliceTrackersCPU[iSlice].SetupCommonMemory();
			}
		}
	}
	for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
	{
		if (mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError != 0)
		{
			const char* errorMsgs[] = HLTCA_GPU_ERROR_STRINGS;
			const char* errorMsg = (unsigned) mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError >= sizeof(errorMsgs) / sizeof(errorMsgs[0]) ? "UNKNOWN" : errorMsgs[mTPCSliceTrackersCPU[iSlice].GPUParameters()->fGPUError];
			printf("Error during tracking: %s\n", errorMsg);
			return(1);
		}
	}
#ifdef HLTCA_STANDALONE
	//printf("Slice Tracks Output %d: - Tracks: %d local, %d global -  Hits: %d local, %d global\n", nOutputTracks, nLocalTracks, nGlobalTracks, nLocalHits, nGlobalHits);
	/*for (unsigned int i = 0;i < NSLICES;i++)
	{
		mTPCSliceTrackersCPU[i].DumpOutput(stdout);
	}*/
#endif
	return 0;
}

int AliGPUReconstruction::RunTPCTrackingMerger()
{
	mTPCMergerCPU.Reconstruct();
	mIOPtrs.mergedTracks = mTPCMergerCPU.OutputTracks();
	mIOPtrs.nMergedTracks = mTPCMergerCPU.NOutputTracks();
	mIOPtrs.mergedTrackHits = mTPCMergerCPU.Clusters();
	mIOPtrs.nMergedTrackHits = mTPCMergerCPU.NOutputTrackClusters();
	return 0;
}

int AliGPUReconstruction::RunTRDTracking()
{
	if (!mTRDTracker->IsInitialized()) return 1;
	std::vector<HLTTRDTrack> tracksTPC;
	std::vector<int> tracksTPCLab;
	std::vector<int> tracksTPCId;

	for (unsigned int i = 0;i < mIOPtrs.nMergedTracks;i++)
	{
		const AliHLTTPCGMMergedTrack& trk = mIOPtrs.mergedTracks[i];
		if (!trk.OK()) continue;
		if (trk.Looper()) continue;
		if (mParam.rec.NWaysOuter) tracksTPC.emplace_back(trk.OuterParam());
		else tracksTPC.emplace_back(trk);
		tracksTPCId.push_back(i);
		tracksTPCLab.push_back(-1);
	}
	
	mTRDTracker->Reset();
	mTRDTracker->StartLoadTracklets(mIOPtrs.nTRDTracklets);

	for (unsigned int iTracklet = 0;iTracklet < mIOPtrs.nTRDTracklets;++iTracklet)
	{
		if (mIOPtrs.trdTrackletsMC) mTRDTracker->LoadTracklet(mIOPtrs.trdTracklets[iTracklet], mIOPtrs.trdTrackletsMC[iTracklet].fLabel);
		else mTRDTracker->LoadTracklet(mIOPtrs.trdTracklets[iTracklet]);
	}

	mTRDTracker->DoTracking(&(tracksTPC[0]), &(tracksTPCLab[0]), tracksTPC.size());
	
	printf("TRD Tracker reconstructed %d tracks\n", mTRDTracker->NTracks());
	
	return 0;
}

AliGPUReconstruction* AliGPUReconstruction::CreateInstance(DeviceType type, bool forceType)
{
	AliGPUCASettingsProcessing cfg;
	cfg.SetDefaults();
	cfg.deviceType = type;
	cfg.forceDeviceType = forceType;
	return CreateInstance(cfg);
}
	
AliGPUReconstruction* AliGPUReconstruction::CreateInstance(const AliGPUCASettingsProcessing& cfg)
{
	AliGPUReconstruction* retVal = nullptr;
	unsigned int type = cfg.deviceType;
	if (type == DeviceType::CPU)
	{
		retVal = new AliGPUReconstruction(cfg);
	}
	else if (type == DeviceType::CUDA)
	{
		if ((retVal = sLibCUDA->GetPtr(cfg))) retVal->mMyLib = sLibCUDA;
	}
	else if (type == DeviceType::HIP)
	{
		if((retVal = sLibHIP->GetPtr(cfg))) retVal->mMyLib = sLibHIP;
	}
	else if (type == DeviceType::OCL)
	{
		if((retVal = sLibOCL->GetPtr(cfg))) retVal->mMyLib = sLibOCL;
	}
	else
	{
		printf("Error: Invalid device type %d\n", type);
		return nullptr;
	}
	
	if (retVal == 0)
	{
		if (cfg.forceDeviceType)
		{
			printf("Error: Could not load AliGPUReconstruction for specified device: %s (%d)\n", DEVICE_TYPE_NAMES[type], type);
		}
		else
		{
			printf("Could not load AliGPUReconstruction for device type %s (%d), falling back to CPU version\n", DEVICE_TYPE_NAMES[type], type);
			AliGPUCASettingsProcessing cfg2 = cfg;
			cfg2.deviceType = CPU;
			retVal = CreateInstance(cfg2);
		}
	}
	else
	{
		printf("Created AliGPUReconstruction instance for device type %s (%d)\n", DEVICE_TYPE_NAMES[type], type);
	}
	
	return retVal;
}

AliGPUReconstruction* AliGPUReconstruction::CreateInstance(const char* type, bool forceType)
{
	DeviceType t = GetDeviceType(type);
	if (t == INVALID_DEVICE)
	{
		printf("Invalid device type: %s\n", type);
		return nullptr;
	}
	return CreateInstance(t, forceType);
}

AliGPUReconstruction::DeviceType AliGPUReconstruction::GetDeviceType(const char* type)
{
	for (unsigned int i = 1;i < sizeof(DEVICE_TYPE_NAMES) / sizeof(DEVICE_TYPE_NAMES[0]);i++)
	{
		if (strcmp(DEVICE_TYPE_NAMES[i], type) == 0)
		{
			return (DeviceType) i;
		}
	}
	return INVALID_DEVICE;
}

#ifdef WIN32
#define LIBRARY_EXTENSION ".dll"
#define LIBRARY_TYPE HMODULE
#define LIBRARY_LOAD(name) LoadLibraryEx(name, NULL, NULL)
#define LIBRARY_CLOSE FreeLibrary
#define LIBRARY_FUNCTION GetProcAddress
#else
#define LIBRARY_EXTENSION ".so"
#define LIBRARY_TYPE void*
#define LIBRARY_LOAD(name) dlopen(name, RTLD_NOW)
#define LIBRARY_CLOSE dlclose
#define LIBRARY_FUNCTION dlsym
#endif

#if defined(HLTCA_BUILD_ALIROOT_LIB)
#define LIBRARY_PREFIX "Ali"
#elif defined(HLTCA_BUILD_O2_LIB)
#define LIBRARY_PREFIX "O2"
#else
#define LIBRARY_PREFIX ""
#endif

std::shared_ptr<AliGPUReconstruction::LibraryLoader> AliGPUReconstruction::sLibCUDA(new AliGPUReconstruction::LibraryLoader("/home/qon/standalone/lib" LIBRARY_PREFIX "TPCCAGPUTracking" "CUDA" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "CUDA"));
std::shared_ptr<AliGPUReconstruction::LibraryLoader> AliGPUReconstruction::sLibHIP(new AliGPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "TPCCAGPUTracking" "HIP" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "HIP"));
std::shared_ptr<AliGPUReconstruction::LibraryLoader> AliGPUReconstruction::sLibOCL(new AliGPUReconstruction::LibraryLoader("lib" LIBRARY_PREFIX "TPCCAGPUTracking" "OCL" LIBRARY_EXTENSION, "AliGPUReconstruction_Create_" "OCL"));

AliGPUReconstruction::LibraryLoader::LibraryLoader(const char* lib, const char* func) : mLibName(lib), mFuncName(func), mGPULib(nullptr), mGPUEntry(nullptr)
{
}

AliGPUReconstruction::LibraryLoader::~LibraryLoader()
{
	CloseLibrary();
}

int AliGPUReconstruction::LibraryLoader::LoadLibrary()
{
	static std::mutex mut;
	std::lock_guard<std::mutex> lock(mut);
	
	if (mGPUEntry) return 0;
	
	LIBRARY_TYPE hGPULib;
	hGPULib = LIBRARY_LOAD(mLibName);
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
		void* createFunc = LIBRARY_FUNCTION(hGPULib, mFuncName);
		if (createFunc == NULL)
		{
			CAGPUError("Error fetching entry function in GPU library\n");
			LIBRARY_CLOSE(hGPULib);
			return 1;
		}
		else
		{
			mGPULib = (void*) (size_t) hGPULib;
			mGPUEntry = createFunc;
			CAGPUInfo("GPU Tracker library loaded and GPU tracker object created sucessfully");
		}
	}
	return 0;
}

AliGPUReconstruction* AliGPUReconstruction::LibraryLoader::GetPtr(const AliGPUCASettingsProcessing& cfg)
{
	if (LoadLibrary()) return nullptr;
	if (mGPUEntry == nullptr) return nullptr;
	AliGPUReconstruction* (*tmp)(const AliGPUCASettingsProcessing& cfg) = (AliGPUReconstruction* (*)(const AliGPUCASettingsProcessing& cfg)) mGPUEntry;
	return tmp(cfg);
}

int AliGPUReconstruction::LibraryLoader::CloseLibrary()
{
	if (mGPUEntry == nullptr) return 1;
	LIBRARY_CLOSE((LIBRARY_TYPE) (size_t) mGPULib);
	mGPULib = nullptr;
	mGPUEntry = nullptr;
	return 0;
}
