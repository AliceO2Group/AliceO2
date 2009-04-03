//-*- Mode: C++ -*-
// @(#) $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKERCOMPONENT_H
#define ALIHLTTPCCATRACKERCOMPONENT_H

#include "AliHLTProcessor.h"

class AliHLTTPCCATracker;
class AliHLTTPCSpacePointData;

/**
 * @class AliHLTTPCCATrackerComponent
 * The Cellular Automaton tracker component.
 */
class AliHLTTPCCATrackerComponent : public AliHLTProcessor
{
public:
  /** standard constructor */
  AliHLTTPCCATrackerComponent();
  
  /** dummy copy constructor, defined according to effective C++ style */
  AliHLTTPCCATrackerComponent(const AliHLTTPCCATrackerComponent&);
  
  /** dummy assignment op, but defined according to effective C++ style */
  AliHLTTPCCATrackerComponent& operator=(const AliHLTTPCCATrackerComponent&);

  /** standard destructor */
  virtual ~AliHLTTPCCATrackerComponent();
      
  // Public functions to implement AliHLTComponent's interface.
  // These functions are required for the registration process
  
  /** @see component interface @ref AliHLTComponent::GetComponentID */
  const char* GetComponentID() ;
  
  /** @see component interface @ref AliHLTComponent::GetInputDataTypes */
  void GetInputDataTypes( vector<AliHLTComponentDataType>& list)  ;
  
  /** @see component interface @ref AliHLTComponent::GetOutputDataType */
  AliHLTComponentDataType GetOutputDataType() ;

  /** @see component interface @ref AliHLTComponent::GetOutputDataSize */
  virtual void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier ) ;

  /** @see component interface @ref AliHLTComponent::Spawn */
  AliHLTComponent* Spawn() ;

protected:

  // Protected functions to implement AliHLTComponent's interface.
  // These functions provide initialization as well as the actual processing
  // capabilities of the component. 
  
  /** @see component interface @ref AliHLTComponent::DoInit */
  Int_t DoInit( Int_t argc, const char** argv );
  
  /** @see component interface @ref AliHLTComponent::DoDeinit */
  Int_t DoDeinit();

  /** reconfigure **/
  Int_t Reconfigure(const char* cdbEntry, const char* chainId);

  /** @see component interface @ref AliHLTProcessor::DoEvent */
  Int_t DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks, 
	       AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr, 
	       AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks );
  
private:
  
  /** the tracker object */
  AliHLTTPCCATracker* fTracker;                                //! transient
  
  /** magnetic field */
  Double_t fSolenoidBz;                                            // see above
  Int_t fMinNTrackClusters; //* required min number of clusters on the track
  Double_t fClusterZCut;  //* cut on cluster Z position (for noise rejection at the age of TPC)
  Double_t fFullTime; //* total time for DoEvent() [s]
  Double_t fRecoTime; //* total reconstruction time [s]
  Long_t    fNEvents;  //* number of reconstructed events
  Bool_t fNewOutputType; //* use new type of output (temporary flag)

  static Bool_t CompareClusters(AliHLTTPCSpacePointData *a, AliHLTTPCSpacePointData *b);

  /** set configuration parameters **/
  Int_t Configure( const char* arguments );

  ClassDef(AliHLTTPCCATrackerComponent, 0);
  
};
#endif
