// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTrackerComponent.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCTRACKERCOMPONENT_H
#define GPUTPCTRACKERCOMPONENT_H

#ifndef GPUCA_ALIROOT_LIB
#define GPUCA_ALIROOT_LIB
#endif

#include "GPUCommonDef.h"
#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"
#include "AliHLTAsyncMemberProcessor.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCClusterData;
class GPUReconstruction;
class GPUChainTracking;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

/**
 * @class GPUTPCTrackerComponent
 * The Cellular Automaton tracker component.
 */
class GPUTPCTrackerComponent : public AliHLTProcessor
{
 public:
  /** standard constructor */
  GPUTPCTrackerComponent();

  /** dummy copy constructor, defined according to effective C++ style */
  GPUTPCTrackerComponent(const GPUTPCTrackerComponent&);

  /** dummy assignment op, but defined according to effective C++ style */
  GPUTPCTrackerComponent& operator=(const GPUTPCTrackerComponent&);

  /** standard destructor */
  virtual ~GPUTPCTrackerComponent();

  // Public functions to implement AliHLTComponent's interface.
  // These functions are required for the registration process

  /** @see component interface @ref AliHLTComponent::GetComponentID */
  const char* GetComponentID();

  /** @see component interface @ref AliHLTComponent::GetInputDataTypes */
  void GetInputDataTypes(vector<AliHLTComponentDataType>& list);

  /** @see component interface @ref AliHLTComponent::GetOutputDataType */
  AliHLTComponentDataType GetOutputDataType();

  /** @see component interface @ref AliHLTComponent::GetOutputDataSize */
  virtual void GetOutputDataSize(unsigned long& constBase, double& inputMultiplier);

  /** @see component interface @ref AliHLTComponent::Spawn */
  AliHLTComponent* Spawn();

 protected:
  // Protected functions to implement AliHLTComponent's interface.
  // These functions provide initialization as well as the actual processing
  // capabilities of the component.

  /** @see component interface @ref AliHLTComponent::DoInit */
  int DoInit(int argc, const char** argv);

  /** @see component interface @ref AliHLTComponent::DoDeinit */
  int DoDeinit();

  /** reconfigure **/
  int Reconfigure(const char* cdbEntry, const char* chainId);

  /** @see component interface @ref AliHLTProcessor::DoEvent */
  int DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks);

 private:
  struct AliHLTTPCTrackerWrapperData {
    const AliHLTComponentEventData* fEvtData;
    const AliHLTComponentBlockData* fBlocks;
    AliHLTUInt8_t* mOutputPtr;
    AliHLTUInt32_t* fSize;
    vector<AliHLTComponentBlockData>* mOutputBlocks;
  };

  static const int NSLICES = 36;    //* N slices
  static const int fgkNPatches = 6; //* N slices

  /** magnetic field */
  double fSolenoidBz;              // see above
  int fMinNTrackClusters;          //* required min number of clusters on the track
  double fMinTrackPt;              //* required min Pt of tracks
  double fClusterZCut;             //* cut on cluster Z position (for noise rejection at the age of TPC)
  double mNeighboursSearchArea;    //* area in cm for the neighbour search algorithm
  double fClusterErrorCorrectionY; // correction for the cluster errors
  double fClusterErrorCorrectionZ; // correction for the cluster errors

  AliHLTComponentBenchmark fBenchmark;           // benchmarks
  char fAllowGPU;                                //* Allow this tracker to run on GPU
  int fGPUHelperThreads;                         // Number of helper threads for GPU tracker, set to -1 to use default number
  int fCPUTrackers;                              // Number of CPU trackers to run in addition to GPU tracker
  char fGlobalTracking;                          // Activate global tracking feature
  int fGPUDeviceNum;                             // GPU Device to use, default -1 for auto detection
  TString fGPUType;                              // GPU type to use "CUDA", "HIP", "OCL"
  int fGPUStuckProtection;                       // Protect from stuck GPUs
  int fAsync;                                    // Run tracking in async thread to catch GPU hangs....
  float fSearchWindowDZDR;                       // See TPCCAParam
  GPUCA_NAMESPACE::gpu::GPUReconstruction* fRec; // GPUReconstruction
  GPUCA_NAMESPACE::gpu::GPUChainTracking* fChain;

  /** set configuration parameters **/
  void SetDefaultConfiguration();
  int ReadConfigurationString(const char* arguments);
  int ReadCDBEntry(const char* cdbEntry, const char* chainId);
  int Configure(const char* cdbEntry, const char* chainId, const char* commandLine);
  void ConfigureSlices();

  AliHLTAsyncMemberProcessor<GPUTPCTrackerComponent> fAsyncProcessor;
  void* TrackerInit(void*);
  void* TrackerExit(void*);
  void* TrackerDoEvent(void*);

  ClassDef(GPUTPCTrackerComponent, 0);
};
#endif // GPUTPCTRACKERCOMPONENT_H
