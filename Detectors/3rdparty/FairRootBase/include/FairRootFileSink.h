/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -----------------------------------------------------------------------------
// -----                                                                   -----
// -----                             FairRootFileSink                      -----
// -----                    Created 22.12.2017 by R. Karabowicz            -----
// -----                                                                   -----
// -----------------------------------------------------------------------------

#ifndef __FAIRROOT__FairRootFileSink__
#define __FAIRROOT__FairRootFileSink__

#include "FairSink.h"

#include <Rtypes.h>
#include <TFile.h>
#include <TString.h>
#include <typeinfo>

class FairEventHeader;
class FairFileHeader;
class TBranch;
class TObjArray;
class TObject;
class TTree;
class TFolder;

class FairRootFileSink : public FairSink
{
  public:
    FairRootFileSink(TFile* f, const char* Title = "OutputRootFile");
    FairRootFileSink(const TString* RootFileName, const char* Title = "OutputRootFile");
    FairRootFileSink(const TString RootFileName, const char* Title = "OutputRootFile");
    //  FairRootFileSink(const FairRootFileSink& file);
    virtual ~FairRootFileSink();

    virtual Bool_t InitSink();
    virtual void Close();
    virtual void Reset();

    virtual Sink_Type GetSinkType() { return kFILESINK; }

    virtual void FillEventHeader(FairEventHeader* feh);

    virtual TFile* OpenRootFile(TString fileName = "");
    TFile* GetRootFile() { return fRootFile; }
    virtual TString GetFileName() { return (fRootFile ? fRootFile->GetName() : ""); }

    virtual void SetOutTree(TTree* fTree) { fOutTree = fTree; }
    TTree* GetOutTree() { return fOutTree; }

    virtual void Fill();

    virtual Int_t Write(const char* name = 0, Int_t option = 0, Int_t bufsize = 0);

    virtual void RegisterImpl(const char*, const char*, void*);
    virtual void RegisterAny(const char* brname, const std::type_info& oi, const std::type_info& pi, void* obj);

    virtual void WriteFolder();
    virtual bool CreatePersistentBranchesAny();

    virtual void WriteObject(TObject* f, const char*, Int_t option = 0);
    virtual void WriteGeometry();

    virtual FairSink* CloneSink();

  private:
    /** Title of input sink, could be input, background or signal*/
    TString fOutputTitle;
    /** ROOT file */
    TFile* fRootFile;
    /** Output Tree  */
    TTree* fOutTree;
    /**  list of folders from all input (and friends) files */
    TObjArray* fListFolder;   //!
    /** folder structure of output */
    TFolder* fCbmout;
    /** Initialization flag, true if initialized */
    Bool_t fIsInitialized;

    FairRootFileSink(const FairRootFileSink&);
    FairRootFileSink operator=(const FairRootFileSink&);

    void TruncateBranchNames();
    void TruncateBranchNames(TBranch* b, TString ffn);
    // bool CreatePersistentBranchesAny();

    /**File Header*/
    FairFileHeader* fFileHeader;   //!

    ClassDef(FairRootFileSink, 1);
};

#endif /* defined(__FAIRROOT__FairRootFileSink__) */
