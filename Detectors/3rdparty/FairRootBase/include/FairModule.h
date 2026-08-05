/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRMODULE_H
#define FAIRMODULE_H

#include "FairGeoInterface.h"   // for FairGeoInterface
#include "FairGeoLoader.h"      // for FairGeoLoader
#include "FairGeoNode.h"        // for FairGeoNode
#include "FairGeoVolume.h"      // for FairGeoVolume
#include "FairLogger.h"
#include "FairRun.h"         // for FairRun

#include <Rtypes.h>      // for Bool_t, Int_t, etc
#include <TList.h>       // for TList (ptr only), TListIter
#include <TNamed.h>      // for TNamed
#include <TObjArray.h>   // for TObjArray
#include <TString.h>     // for TString, operator!=
#include <string>        // for string

class FairVolumeList;
class FairVolume;
class TArrayI;
class TGeoMatrix;
class TGeoNode;
class TGeoVolume;
class TRefArray;
class TVirtualMC;

/**
 * Base class for constructing all detecors and passive volumes
 * @author M. Al-Turany, Denis Bertini
 * @version 1.0
 * @since 01.04.08 M.Al-Turany
 * Add methods to construct geometry via ROOT files
 * Add some documentation
 *
 * Changelog: 29.02.2012 [O.Merle] Fixed missing material assignment for top volume.
 *                                 ... and please - add some documentation to your code.
 */
class FairModule : public TNamed
{
  public:
    /**default ctor*/
    FairModule();
    /**Standard ctor*/
    FairModule(const char* Name, const char* title, Bool_t Active = kFALSE);
    /**default dtor*/
    virtual ~FairModule();
    /**Print method should be implemented in detector or module*/
    virtual void Print(Option_t*) const { ; }
    /**Set the geometry file name o be used*/
    virtual void SetGeometryFileName(TString fname, TString geoVer = "0");
    /**Get the Geometry file name*/
    virtual TString GetGeometryFileName() { return fgeoName; }
    /**Get the geometry file version if used*/
    virtual TString GetGeometryFileVer() { return fgeoVer; }
    /**method called from the MC application to construct the geometry, has to be implimented by user*/
    virtual void ConstructGeometry();
    /**method called from the MC application to set optical geometry properties*/
    virtual void ConstructOpGeometry();
    /**construct geometry from root files (TGeo)*/
    virtual void ConstructRootGeometry(TGeoMatrix* shiftM = nullptr);
    /**construct geometry from standard ASSCII files (Hades Format)*/
    virtual void ConstructASCIIGeometry();
    /** Modify the geometry for the simulation run using methods of the Root geometry package */
    virtual void ModifyGeometry()
        __attribute__((deprecated("Use FairAlignmentHandler instead, see Tutorial4 for examples")))
    {
        LOG(warn) << "This function is deprecated. Use FairAlignmentHandler instead, see Tutorial4 for examples.";
    }
    virtual void RegisterAlignmentMatrices() { ; }

    /**construct geometry from GDML files*/
    virtual void ConstructGDMLGeometry(__attribute__((unused)) TGeoMatrix* posrot);
    /** custom settings of processes and cuts for media to be forwarded to the
     ** detector simulation */
    virtual void SetSpecialPhysicsCuts() { ; }
    /** Clone this object (used in MT mode only)*/
    virtual FairModule* CloneModule() const;
    /** Init worker run (used in MT mode only) */
    virtual void BeginWorkerRun() const { ; }
    /** Finish worker run (used in MT mode only) */
    virtual void FinishWorkerRun() const { ; }


    /**Set the sensitivity flag for volumes, called from ConstructASCIIRootGeometry(), and has to be implimented for
     * detectors which use ConstructASCIIRootGeometry() to build the geometry */
    virtual Bool_t IsSensitive(const std::string& name);
    /**The function below is depracated, please change to the new method above */
    virtual Bool_t CheckIfSensitive(__attribute__((unused)) std::string name) __attribute__((
        deprecated("The method CheckIfSensitive is deprecated. Implement IsSensitive in the detector classes.")))
    {
        return kFALSE;
    }
    /**called from ConstructRootGeometry()*/
    virtual void ExpandNode(TGeoNode* Node);
    /**called from ConstructGDMLGeometry()*/
    virtual void ExpandNodeForGDML(__attribute__((unused)) TGeoNode* curNode);
    /**return the MC id of a volume named vname*/
    virtual Int_t getVolId(const TString&) const { return 0; }
    /**return the detector/Module id (which was set in the sim macro for the detector)*/
    Int_t GetModId() { return fModId; }
    /**Set the verbose level in this detector*/
    void SetVerboseLevel(Int_t level) { fVerboseLevel = level; }
    /**return the detector status */
    Bool_t IsActive() { return fActive; }
    /**set the detector/module id*/
    void SetModId(Int_t id) { fModId = id; }
    /** Set the name of the mother volume to which a new geometry is added.
     ** This function is needed for geometries which are defined as ROOT geometry manager.
     **/
    void SetMotherVolume(TString volName) { fMotherVolumeName = volName; }
    /**called from ConstuctASCIIGeometry*/
    void ProcessNodes(TList* aList);
    /**Set the parameter containers*/
    virtual void SetParContainers() { ; }
    /** Initialize everything which has to be done before the construction and modification
     ** of the geometry. Mostly this is needed to read data from the parameter containers.*/
    virtual void InitParContainers() { ; }
    /**return the geo parameter of this detector/module*/
    TList* GetListOfGeoPar() { return flGeoPar; }

    /**list of volumes in a simulation session*/
    static thread_local FairVolumeList* vList;   //!
    /**total number of volumes in a simulaion session*/
    static thread_local Int_t fNbOfVolumes;   //!
    /**list of all sensitive volumes in  a simulaion session*/
    static thread_local TRefArray* svList;   //!

    static thread_local TArrayI* volNumber;   //!
    TString fMotherVolumeName;                //!
    FairVolume* getFairVolume(FairGeoNode* fNode);
    void AddSensitiveVolume(TGeoVolume* v);

  private:
    /** Re-implimented from ROOT:  TGeoMatrix::SetDefaultName()  */
    void SetDefaultMatrixName(TGeoMatrix* matrix);
    void AssignMediumAtImport(TGeoVolume* v);   // O.Merle, 29.02.2012 - see impl.

    /**called from ConstructGDMLGeometry. Changes default ID created by TGDMLParse*/
    void ReAssignMediaId();
    void swap(FairModule& other) throw();

  protected:
    FairModule(const FairModule&);
    FairModule& operator=(const FairModule&);
    TString fgeoVer;
    TString fgeoName;
    Int_t fModId;
    Bool_t fActive;
    Int_t fNbOfSensitiveVol;   //!
    Int_t fVerboseLevel;
    TList* flGeoPar;    //!  list of Detector Geometry parameters
    Bool_t fGeoSaved;   //! flag for initialisation
    TVirtualMC* fMC;    //! cahed pointer to MC (available only after initialization)

    ClassDef(FairModule, 4);
};

// The container-based ConstructASCIIGeometry<T,U> template was removed with the
// FairRuntimeDb: it stored imported ASCII geometry into a parameter container.


#endif   // FAIRMODULE_H
