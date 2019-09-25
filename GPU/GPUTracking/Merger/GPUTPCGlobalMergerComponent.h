// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGlobalMergerComponent.h
/// \author David Rohr, Sergey Gorbunov, Matthias Kretz

#ifndef GPUTPCGLOBALMERGERCOMPONENT_H
#define GPUTPCGLOBALMERGERCOMPONENT_H

/// @file   GPUTPCGlobalMergerComponent.h
/// @author Matthias Kretz
/// @date
/// @brief  HLT TPC CA global merger component.
///

#ifndef GPUCA_ALIROOT_LIB
#define GPUCA_ALIROOT_LIB
#endif

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"
#include "GPUParam.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMMerger;
class GPUReconstruction;
class GPUChainTracking;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

/**
 * @class GPUTPCGlobalMergerComponent
 * The TPC global merger component
 *
 * Interface to the global merger of the CA tracker for HLT.
 */
class GPUTPCGlobalMergerComponent : public AliHLTProcessor
{
 public:
  /**
 * Constructs a GPUTPCGlobalMergerComponent.
 */
  GPUTPCGlobalMergerComponent();

  /**
 * Destructs the GPUTPCGlobalMergerComponent
 */
  virtual ~GPUTPCGlobalMergerComponent();

  // Public functions to implement AliHLTComponent's interface.
  // These functions are required for the registration process

  /**
 * @copydoc AliHLTComponent::GetComponentID
 */
  const char* GetComponentID();

  /**
 * @copydoc AliHLTComponent::GetInputDataTypes
 */
  void GetInputDataTypes(AliHLTComponentDataTypeList& list);
  int GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList);

  /**
 * @copydoc AliHLTComponent::GetOutputDataType
 */
  AliHLTComponentDataType GetOutputDataType();

  /**
 * @copydoc AliHLTComponent::GetOutputDataSize
 */
  virtual void GetOutputDataSize(unsigned long& constBase, double& inputMultiplier);

  /**
 * @copydoc AliHLTComponent::Spawn
 */
  AliHLTComponent* Spawn();

  static const GPUCA_NAMESPACE::gpu::GPUTPCGMMerger* GetCurrentMerger();

 protected:
  // Protected functions to implement AliHLTComponent's interface.
  // These functions provide initialization as well as the actual processing
  // capabilities of the component.

  /**
 * @copydoc AliHLTComponent::DoInit
 */
  int DoInit(int argc, const char** argv);

  /**
 * @copydoc AliHLTComponent::DoDeinit
 */
  int DoDeinit();

  /** reconfigure **/
  int Reconfigure(const char* cdbEntry, const char* chainId);

  /**
 * @copydoc @ref AliHLTProcessor::DoEvent
 */
  int DoEvent(const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, AliHLTUInt32_t& size, AliHLTComponentBlockDataList& outputBlocks);

  using AliHLTProcessor::DoEvent;

 private:
  static GPUTPCGlobalMergerComponent fgGPUTPCGlobalMergerComponent;

  // disable copy
  GPUTPCGlobalMergerComponent(const GPUTPCGlobalMergerComponent&);
  GPUTPCGlobalMergerComponent& operator=(const GPUTPCGlobalMergerComponent&);

  /** set configuration parameters **/
  void SetDefaultConfiguration();
  int ReadConfigurationString(const char* arguments);
  int ReadCDBEntry(const char* cdbEntry, const char* chainId);
  int Configure(const char* cdbEntry, const char* chainId, const char* commandLine);

  /** the global merger object */

  double fSolenoidBz;                                                                 // magnetic field
  double fClusterErrorCorrectionY;                                                    // correction for the cluster error during pre-fit
  double fClusterErrorCorrectionZ;                                                    // correction for the cluster error during pre-fit
  int fNWays;                                                                         // Setting for merger
  char fNWaysOuter;                                                                   // Store outer param after n-way fit
  bool fNoClear;                                                                      // Do not clear memory after processing an event
  static const GPUCA_NAMESPACE::gpu::GPUChainTracking* fgCurrentMergerReconstruction; // Pointer to current merger in case memory is not cleared after processing the event
  AliHLTComponentBenchmark fBenchmark;                                                // benchmark
  GPUCA_NAMESPACE::gpu::GPUParam mParam;                                              // ca params
  GPUCA_NAMESPACE::gpu::GPUReconstruction* fRec;                                      // GPUReconstruction
  GPUCA_NAMESPACE::gpu::GPUChainTracking* fChain;

  ClassDef(GPUTPCGlobalMergerComponent, 0);
};

#endif // GPUTPCGLOBALMERGERCOMPONENT_H
