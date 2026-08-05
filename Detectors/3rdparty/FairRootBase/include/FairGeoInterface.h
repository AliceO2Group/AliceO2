/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOINTERFACE_H
#define FAIRGEOINTERFACE_H

#include <Rtypes.h>    // for Bool_t, kFALSE, Int_t, etc
#include <TObject.h>   // for TObject
#include <TString.h>   // for TString

class FairGeoIo;
class FairGeoSet;
class FairGeoMedia;
class FairGeoShapes;
class FairGeoBuilder;
class TObjArray;
class TList;

enum EHGeoDetPart
{
    kFairGeoCave = 0,
    kFairGeoPipe = 1,
    kFairGeoTarget = 2,
    kFairGeoMagnet = 3,
    kFairGeoRich = 4,
    kFairGeoSts = 5,
    kFairGeoTrd = 6,
    kFairGeoTof = 7,
    kFairGeoEcal = 8
};

/**
 * Class to manage geometry for simulations
 * @author Ilse koenig
 * @author M. Al-Turany
 */
class FairGeoInterface : public TObject
{
    FairGeoIo* fileInput;       /** ASCII file I/O */
    FairGeoIo* oraInput;        /** Oracle input */
    FairGeoIo* output;          /** Oracle output */
    Int_t nSets;                /** number of geometry sets (detector parts) */
    Int_t nActualSets;          /** number of set in actual geometry */
    TObjArray* sets;            /** array of geometry  sets */
    FairGeoMedia* media;        /** list of media */
    FairGeoShapes* shapes;      /** list of shapes */
    TList* masterNodes;         /** list of mother nodes used by several sets */
    TString setupFile;          /** file with detector setups (subsets) */
    FairGeoBuilder* geoBuilder; /** actually used geometry builder */

  public:
    FairGeoInterface();
    ~FairGeoInterface();
    void setOracleInput(FairGeoIo* p) { oraInput = p; }
    void setOutput(FairGeoIo* p) { output = p; }
    void setGeomBuilder(FairGeoBuilder* p) { geoBuilder = p; }
    FairGeoIo* getFileInput() { return fileInput; }
    FairGeoIo* getOraInput() { return oraInput; }
    FairGeoIo* getOutput() { return output; }
    FairGeoShapes* getShapes() { return shapes; }
    FairGeoMedia* getMedia() { return media; }
    TList* getMasterNodes() { return masterNodes; }
    void addInputFile(const char*);
    void addGeoModule(FairGeoSet*);
    void setMediaFile(const char* file);
    FairGeoSet* findSet(const char*);
    Bool_t readSet(FairGeoSet*);
    Bool_t writeSet(FairGeoSet*);
    Bool_t writeSet(FairGeoSet*, const char*);
    Bool_t createSet(FairGeoSet*);
    void deleteSet(FairGeoSet* pSet);
    Bool_t readMedia();
    Bool_t writeMedia();
    Bool_t writeMedia(const char*);
    Bool_t readAll();
    Bool_t writeAll();
    Bool_t createAll(Bool_t withCleanup = kFALSE);
    Bool_t createGeometry(Bool_t withCleanup = kFALSE);
    Bool_t readGeomConfig(const char*);
    void addSetupFile(const char* f) { setupFile = f; }
    Bool_t readSetupFile();
    void print();
    void SetNoOfSets(Int_t n) { nSets = n; }

  private:
    FairGeoInterface(const FairGeoInterface&);
    FairGeoInterface& operator=(const FairGeoInterface&);

    FairGeoIo* connectInput(const char*);
    Bool_t connectOutput(const char*);

    ClassDef(FairGeoInterface, 0);
};

#endif /* !FAIRGEOINTERFACE_H */
