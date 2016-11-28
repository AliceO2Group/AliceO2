//-*- Mode: C++ -*-
// $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

///  @file   AliHLTTRDTrackerComponent.h
///  @author Marten Ole Schmidt <ole.schmidt@cern.ch>
///  @date   May 2016
///  @brief  A TRD tracker processing component for the HLT

#ifndef ALIHLTTRDTRACKERCOMPONENT_H
#define ALIHLTTRDTRACKERCOMPONENT_H

#include "AliHLTProcessor.h"
#include "AliHLTDataTypes.h"

class TH1F;
class TList;
class AliHLTTRDTracker;


class AliHLTTRDTrackerComponent : public AliHLTProcessor {
public:

  /*
   * ---------------------------------------------------------------------------------
   *                            Constructor / Destructor
   * ---------------------------------------------------------------------------------
   */

  /** constructor */
  AliHLTTRDTrackerComponent();

  /** dummy copy constructor, defined according to effective C++ style */
  AliHLTTRDTrackerComponent( const AliHLTTRDTrackerComponent& );

  /** dummy assignment op, but defined according to effective C++ style */
  AliHLTTRDTrackerComponent& operator=( const AliHLTTRDTrackerComponent& );

  /** destructor */
  virtual ~AliHLTTRDTrackerComponent();

  /*
   * ---------------------------------------------------------------------------------
   * Public functions to implement AliHLTComponent's interface.
   * These functions are required for the registration process
   * ---------------------------------------------------------------------------------
   */

  /** interface function, see @ref AliHLTComponent for description */
  const char* GetComponentID();

  /** interface function, see @ref AliHLTComponent for description */
  void GetInputDataTypes( vector<AliHLTComponentDataType>& list);

  /** interface function, see @ref AliHLTComponent for description */
  AliHLTComponentDataType GetOutputDataType();

  /** @see component interface @ref AliHLTComponent::GetOutputDataType */
  int  GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList);

  /** interface function, see @ref AliHLTComponent for description */
  void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier );

  /** interface function, see @ref AliHLTComponent for description */
  AliHLTComponent* Spawn();

  int ReadConfigurationString( const char* arguments );

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
  int DoInit( int argc, const char** argv );

  /** interface function, see @ref AliHLTComponent for description */
  int DoDeinit();

  /** interface function, see @ref AliHLTComponent for description */
  int DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks,
               AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr,
               AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks );


  /** interface function, see @ref AliHLTComponent for description */
  int Reconfigure(const char* cdbEntry, const char* chainId);


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
  AliHLTTRDTracker *fTracker; // the tracker itself

  TList* fTrackList;
  bool fDebugTrackOutput; // output AliHLTTRDTracks instead AliHLTExternalTrackParam

  ClassDef(AliHLTTRDTrackerComponent, 0)
};
#endif
