//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAINPUTDATACOMPRESSORCOMPONENT_H
#define ALIHLTTPCCAINPUTDATACOMPRESSORCOMPONENT_H

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"

class AliHLTTPCClusterData;

/**
 * @class AliHLTTPCCAInputDataCompressorComponent
 *
 * The component produces a compressed TPC data for the Cellular Automaton tracker.
 * Can be optionally used after the clusterfinder.
 * It extracts from HLT clusters information needed by the tracker, and skips the rest.
 * The compress ratio is about 4.7 times
 * 
 */
class AliHLTTPCCAInputDataCompressorComponent : public AliHLTProcessor
{
  public:
    /** standard constructor */
    AliHLTTPCCAInputDataCompressorComponent();

    /** dummy copy constructor, defined according to effective C++ style */
    AliHLTTPCCAInputDataCompressorComponent( const AliHLTTPCCAInputDataCompressorComponent& );

    /** dummy assignment op, but defined according to effective C++ style */
    AliHLTTPCCAInputDataCompressorComponent& operator=( const AliHLTTPCCAInputDataCompressorComponent& );

    /** standard destructor */
    virtual ~AliHLTTPCCAInputDataCompressorComponent();

    // Public functions to implement AliHLTComponent's interface.
    // These functions are required for the registration process

    /** @see component interface @ref AliHLTComponent::GetComponentID */
    const char* GetComponentID() ;

    /** @see component interface @ref AliHLTComponent::GetInputDataTypes */
    void GetInputDataTypes( vector<AliHLTComponentDataType>& list )  ;

    /** @see component interface @ref AliHLTComponent::GetOutputDataType */
    AliHLTComponentDataType GetOutputDataType() ;

    /** @see component interface @ref AliHLTComponent::GetOutputDataSize */
    virtual void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier ) ;

    /** @see component interface @ref AliHLTComponent::Spawn */
    AliHLTComponent* Spawn() ;

  static int Compress( AliHLTTPCClusterData* inputPtr,
		       AliHLTUInt32_t maxBufferSize,
		       AliHLTUInt8_t* outputPtr,
		       AliHLTUInt32_t& outputSize
		       );

  protected:

    // Protected functions to implement AliHLTComponent's interface.
    // These functions provide initialization as well as the actual processing
    // capabilities of the component.

    /** @see component interface @ref AliHLTComponent::DoInit */
    int DoInit( int argc, const char** argv );

    /** @see component interface @ref AliHLTComponent::DoDeinit */
    int DoDeinit();

    /** reconfigure **/
    int Reconfigure( const char* cdbEntry, const char* chainId );

    /** @see component interface @ref AliHLTProcessor::DoEvent */
    int DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks,
                 AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr,
                 AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks );

  private:

  AliHLTComponentBenchmark fBenchmark; // benchmarks

    ClassDef( AliHLTTPCCAInputDataCompressorComponent, 0 );
};
#endif
