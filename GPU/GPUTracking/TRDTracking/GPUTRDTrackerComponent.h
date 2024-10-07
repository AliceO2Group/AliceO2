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

/// \file GPUTRDTrackerComponent.h
/// \brief A TRD tracker processing component for the GPU

/// \author Ole Schmidt

#ifndef GPUTRDTRACKERCOMPONENT_H
#define GPUTRDTRACKERCOMPONENT_H

#ifndef GPUCA_ALIROOT_LIB
#define GPUCA_ALIROOT_LIB
#endif

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"
#include "AliHLTDataTypes.h"

class TH1F;
class TList;

#include "GPUTRDDef.h"
namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTRDGeometry;
class GPUReconstruction;
class GPUChainTracking;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

class GPUTRDTrackerComponent : public AliHLTProcessor
{
 public:
  /*
 * ---------------------------------------------------------------------------------
 *                            Constructor / Destructor
 * ---------------------------------------------------------------------------------
 */

  /** constructor */
  GPUTRDTrackerComponent();

  /** dummy copy constructor, defined according to effective C++ style */
  GPUTRDTrackerComponent(const GPUTRDTrackerComponent&);

  /** dummy assignment op, but defined according to effective C++ style */
  GPUTRDTrackerComponent& operator=(const GPUTRDTrackerComponent&);

  /** destructor */
  virtual ~GPUTRDTrackerComponent();

  /*
 * ---------------------------------------------------------------------------------
 * Public functions to implement AliHLTComponent's interface.
 * These functions are required for the registration process
 * ---------------------------------------------------------------------------------
 */

  /** interface function, see @ref AliHLTComponent for description */
  const char* GetComponentID();

  /** interface function, see @ref AliHLTComponent for description */
  void GetInputDataTypes(vector<AliHLTComponentDataType>& list);

  /** interface function, see @ref AliHLTComponent for description */
  AliHLTComponentDataType GetOutputDataType();

  /** @see component interface @ref AliHLTComponent::GetOutputDataType */
  int32_t GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList);

  /** interface function, see @ref AliHLTComponent for description */
  void GetOutputDataSize(uint64_t& constBase, double& inputMultiplier);

  /** interface function, see @ref AliHLTComponent for description */
  AliHLTComponent* Spawn();

  int32_t ReadConfigurationString(const char* arguments);

 protected:
  /*
 * ---------------------------------------------------------------------------------
 * Protected functions to implement AliHLTComponent's interface.
 * These functions provide initialization as well as the actual processing
 * capabilities of the component.
 * ---------------------------------------------------------------------------------
 */

  // AliHLTComponent interface functions

  /** interface function, see @ref AliHLTComponent for description */
  int32_t DoInit(int argc, const char** argv);

  /** interface function, see @ref AliHLTComponent for description */
  int32_t DoDeinit();

  /** interface function, see @ref AliHLTComponent for description */
  int32_t DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks);

  /** interface function, see @ref AliHLTComponent for description */
  int32_t Reconfigure(const char* cdbEntry, const char* chainId);

  ///////////////////////////////////////////////////////////////////////////////////

 private:
  /*
 * ---------------------------------------------------------------------------------
 * Private functions to implement AliHLTComponent's interface.
 * These functions provide initialization as well as the actual processing
 * capabilities of the component.
 * ---------------------------------------------------------------------------------
 */

  /*
 * ---------------------------------------------------------------------------------
 *                              Helper
 * ---------------------------------------------------------------------------------
 */

  /*
 * ---------------------------------------------------------------------------------
 *                             Members - private
 * ---------------------------------------------------------------------------------
 */
  GPUCA_NAMESPACE::gpu::GPUTRDTrackerGPU* fTracker; // the tracker itself
  GPUCA_NAMESPACE::gpu::GPUTRDGeometry* fGeo;       // TRD geometry needed by the tracker
  GPUCA_NAMESPACE::gpu::GPUReconstruction* fRec;    // GPU Reconstruction object
  GPUCA_NAMESPACE::gpu::GPUChainTracking* fChain;   // Tracking Chain Object

  TList* fTrackList;
  bool fDebugTrackOutput;              // output GPUTRDTracks instead AliHLTExternalTrackParam
  bool fVerboseDebugOutput;            // more verbose information is printed
  bool fRequireITStrack;               // only TPC tracks with ITS match are used as seeds for tracking
  AliHLTComponentBenchmark fBenchmark; // benchmark

  ClassDef(GPUTRDTrackerComponent, 0);
};
#endif // GPUTRDTRACKERCOMPONENT_H
