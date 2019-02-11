#ifndef ROOT_ALITRDTKDINTERPOLATOR_H
#define ROOT_ALITRDTKDINTERPOLATOR_H

#ifndef ROOT_TKDTree
#include "TKDTree.h"
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#include "TVectorD.h"
#include "TMatrixD.h"
#include "TH2Poly.h"
class TClonesArray;

class AliTRDTKDInterpolator : public TKDTreeIF
{

public:
    enum TRDTKDMode{
	kInterpolation=0,
	kMinError=1,
	kNodeVal=2
    };

      // Bucket Object class
    class AliTRDTKDNodeInfo : public TObject
    {
        friend class AliTRDTKDInterpolator;
    public:

	AliTRDTKDNodeInfo(Int_t ndim = 0);
	AliTRDTKDNodeInfo(const AliTRDTKDInterpolator::AliTRDTKDNodeInfo &ref);
	AliTRDTKDInterpolator::AliTRDTKDNodeInfo& operator=(const AliTRDTKDInterpolator::AliTRDTKDNodeInfo &ref);
	virtual ~AliTRDTKDNodeInfo();
	Bool_t        CookPDF(const Double_t *point, Double_t &result, Double_t &error,TRDTKDMode mod=kInterpolation) const;
	Bool_t Has(const Float_t *p) const;
	void          Print(const Option_t * = "") const;
	void          Store(TVectorD const *par, TMatrixD const *cov,Bool_t storeCov);

    private:
	Int_t fNDim;              // Dimension of Points
        Int_t fNBounds;           // 2* Dimension of Points
	Int_t fNPar;            // Number of Parameters
	Int_t fNCov;            // Size of Cov Matrix
	Float_t   *fData;         //[fNDim] Data Point
	Float_t *fBounds;         //[fNBounds] Boundaries
	Float_t   fVal[2];        //measured value for node
	Double_t  *fPar;          //[fNPar] interpolator parameters
	Double_t  *fCov;          //[fNCov] interpolator covariance matrix

	ClassDef(AliTRDTKDNodeInfo, 1)  // node info for interpolator
    };

public:

    AliTRDTKDInterpolator();
    AliTRDTKDInterpolator(Int_t npoints, Int_t ndim, UInt_t bsize, Float_t **data);
    ~AliTRDTKDInterpolator();

    Bool_t        Eval(const Double_t *point, Double_t &result, Double_t &error);
    void          Print(const Option_t *opt="") const;

    TH2Poly *     Projection(Int_t xdim,Int_t ydim);

    Int_t         GetNDIM() const {return fNDim;}
    Bool_t        GetRange(Int_t idim,Float_t range[2]) const;
  
    void          SetNPointsInterpolation(Int_t np){fNPointsI=np;};
    Int_t         GetNPointsInterpolation(){return fNPointsI;};

    void          SetUseWeights(Bool_t k=kTRUE){fUseWeights=k;}
    void          SetPDFMode(TRDTKDMode mod){fPDFMode=mod;}
    void          SetStoreCov(Bool_t k){fStoreCov=k;}

    Bool_t        Build();

    void          SetUseHelperNodes(Bool_t k){fUseHelperNodes=k;}

private:

    Int_t         GetNodeIndex(const Float_t *p);
    AliTRDTKDInterpolator::AliTRDTKDNodeInfo*  GetNodeInfo(Int_t inode) const;
    Int_t         GetNTNodes() const;
    void          BuildInterpolation();
    void          BuildBoundaryNodes();
    AliTRDTKDInterpolator(const AliTRDTKDInterpolator &ref);
    AliTRDTKDInterpolator &operator=(const AliTRDTKDInterpolator &ref);

    Int_t fNDataNodes;         // Number of filled nodes (total-zero nodes)
    TClonesArray  *fNodes;     //interpolation nodes
    UChar_t       fLambda;      // number of parameters in polynom
    Int_t         fNPointsI;    // number of points for interpolation
    Bool_t        fUseHelperNodes; // Build Helper nodes to ensure boundary conditions
    Bool_t fUseWeights; // Use tricubic weights
    TRDTKDMode    fPDFMode; // Mode for PDF calculation
    Bool_t        fStoreCov;

    ClassDef(AliTRDTKDInterpolator, 2)   // data interpolator based on KD tree
};

#endif
