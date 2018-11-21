#ifndef ALIHLTGPUDUMPCOMPONENT_H
#define ALIHLTGPUDUMPCOMPONENT_H

#include "AliTPCCommonDef.h"
#include "AliHLTProcessor.h"

class AliGPUReconstruction;
class AliHLTTPCCAClusterData;
#include "AliRecoParam.h"
class AliTPCTransform;
namespace ali_tpc_common { namespace tpc_fast_transformation {
    class TPCFastTransform;
    class TPCFastTransformManager;
}}

class AliHLTGPUDumpComponent : public AliHLTProcessor
{
  public:
    AliHLTGPUDumpComponent();

    AliHLTGPUDumpComponent( const AliHLTGPUDumpComponent& ) CON_DELETE;
    AliHLTGPUDumpComponent& operator=( const AliHLTGPUDumpComponent& ) CON_DELETE;

    virtual ~AliHLTGPUDumpComponent();

    const char* GetComponentID() ;
    void GetInputDataTypes( vector<AliHLTComponentDataType>& list )  ;
    AliHLTComponentDataType GetOutputDataType() ;
    virtual void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier ) ;
    AliHLTComponent* Spawn() ;

  protected:
    int DoInit( int argc, const char** argv );
    int DoDeinit();
    int Reconfigure( const char* cdbEntry, const char* chainId );
    int DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks,
                 AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr,
                 AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks );

  private:

    float fSolenoidBz;
    AliGPUReconstruction* fRec;
    AliHLTTPCCAClusterData* fClusterData;
    std::unique_ptr<ali_tpc_common::tpc_fast_transformation::TPCFastTransform> fFastTransformIRS;
    std::unique_ptr<ali_tpc_common::tpc_fast_transformation::TPCFastTransformManager> fFastTransformManager;
    AliRecoParam fOfflineRecoParam;
    AliTPCTransform* fOrigTransform;
    bool fIsMC;
};

#endif
