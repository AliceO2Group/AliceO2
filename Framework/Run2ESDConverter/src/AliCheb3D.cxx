/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include <TString.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TRandom.h>
#include <stdio.h>
#include <TMethodCall.h>
#include <TMath.h>
#include <TH1.h>
#include "AliCheb3D.h"
#include "AliCheb3DCalc.h"
#include "AliLog.h"

ClassImp(AliCheb3D)

const Float_t AliCheb3D::fgkMinPrec = 1.e-12f;

//__________________________________________________________________________________________
AliCheb3D::AliCheb3D() : 
  fDimOut(0), 
  fPrec(0), 
  fChebCalc(1), 
  fMaxCoefs(0), 
  fResTmp(0), 
  fGrid(0), 
  fUsrFunName(""), 
  fUsrMacro(0) 
{
// Default constructor
  for (int i=3;i--;) {
    fBMin[i] = fBMax[i] = fBScale[i] = fBOffset[i] = fArgsTmp[i] = 0;
    fNPoints[i] = 0;
    fGridOffs[i] = 0;
  }
}

//__________________________________________________________________________________________
AliCheb3D::AliCheb3D(const AliCheb3D& src) : 
  TNamed(src),
  fDimOut(src.fDimOut), 
  fPrec(src.fPrec), 
  fChebCalc(1), 
  fMaxCoefs(src.fMaxCoefs), 
  fResTmp(0),
  fGrid(0), 
  fUsrFunName(src.fUsrFunName), 
  fUsrMacro(0)
{
  // read coefs from text file
  for (int i=3;i--;) {
    fBMin[i]    = src.fBMin[i];
    fBMax[i]    = src.fBMax[i];
    fBScale[i]  = src.fBScale[i];
    fBOffset[i] = src.fBOffset[i];
    fNPoints[i] = src.fNPoints[i];
    fGridOffs[i] = src.fGridOffs[i];
    fArgsTmp[i]  = 0;
  }
  for (int i=0;i<fDimOut;i++) {
    AliCheb3DCalc* cbc = src.GetChebCalc(i);
    if (cbc) fChebCalc.AddAtAndExpand(new AliCheb3DCalc(*cbc),i);
  }
}

//__________________________________________________________________________________________
AliCheb3D::AliCheb3D(const char* inpFile) : 
  fDimOut(0), 
  fPrec(0),  
  fChebCalc(1),
  fMaxCoefs(0),  
  fResTmp(0),
  fGrid(0), 
  fUsrFunName(""), 
  fUsrMacro(0)
{
  // read coefs from text file
  for (int i=3;i--;) {
    fBMin[i] = fBMax[i] = fBScale[i] = fBOffset[i] = 0;
    fNPoints[i] = 0;
    fGridOffs[i] = 0;
    fArgsTmp[i]  = 0;
  }
  LoadData(inpFile);
}

//__________________________________________________________________________________________
AliCheb3D::AliCheb3D(FILE* stream) : 
  fDimOut(0), 
  fPrec(0), 
  fChebCalc(1), 
  fMaxCoefs(0),
  fResTmp(0),
  fGrid(0),
  fUsrFunName(""),
  fUsrMacro(0)
{
  // read coefs from stream
  for (int i=3;i--;) {
    fBMin[i] = fBMax[i] = fBScale[i] = fBOffset[i] = 0;
    fNPoints[i] = 0;
    fGridOffs[i] = 0;
    fArgsTmp[i]  = 0;
  }
  LoadData(stream);
}

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
AliCheb3D::AliCheb3D(const char* funName, int DimOut, const Float_t  *bmin, const Float_t  *bmax, Int_t *npoints, Float_t prec, const Float_t* precD) : 
  TNamed(funName,funName), 
  fDimOut(0), 
  fPrec(TMath::Max(fgkMinPrec,prec)), 
  fChebCalc(1), 
  fMaxCoefs(0), 
  fResTmp(0), 
  fGrid(0), 
  fUsrFunName("") ,
  fUsrMacro(0)
{
  // Construct the parameterization for the function
  // funName : name of the file containing the function: void funName(Float_t * inp,Float_t * out)
  // DimOut  : dimension of the vector computed by the user function
  // bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  // bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  // npoints : array of 3 elements with the number of points to compute in each of 3 dimension
  // prec    : max allowed absolute difference between the user function and computed parameterization on the requested grid
  // precD   : optional array with precisions per output dimension (if >fgkMinPrec will override common prec)
  if (DimOut<1) AliFatalF("Requested output dimension is %d",fDimOut);
  for (int i=3;i--;) {
    fBMin[i] = fBMax[i] = fBScale[i] = fBOffset[i] = 0;
    fNPoints[i] = 0;
    fGridOffs[i] = 0.;
    fArgsTmp[i]  = 0;
  }
  SetDimOut(DimOut,precD);
  PrepareBoundaries(bmin,bmax);
  DefineGrid(npoints);
  SetUsrFunction(funName);
  ChebFit();
  //
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
AliCheb3D::AliCheb3D(void (*ptr)(float*,float*), int DimOut, Float_t  *bmin,Float_t  *bmax, Int_t *npoints, Float_t prec, const Float_t* precD) : 
  TNamed("Cheb3D","Cheb3D"),
  fDimOut(0), 
  fPrec(TMath::Max(fgkMinPrec,prec)), 
  fChebCalc(1), 
  fMaxCoefs(0), 
  fResTmp(0), 
  fGrid(0), 
  fUsrFunName(""),
  fUsrMacro(0)
{
  // Construct the parameterization for the function
  // ptr     : pointer on the function: void fun(Float_t * inp,Float_t * out)
  // DimOut  : dimension of the vector computed by the user function
  // bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  // bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  // npoints : array of 3 elements with the number of points to compute in each of 3 dimension
  // prec    : max allowed absolute difference between the user function and computed parameterization on the requested grid
  // precD   : optional array with precisions per output dimension (if >fgkMinPrec will override common prec)
  //
  if (DimOut<1) AliFatalF("Requested output dimension is %d",fDimOut);
  for (int i=3;i--;) {
    fBMin[i] = fBMax[i] = fBScale[i] = fBOffset[i] = 0;
    fNPoints[i] = 0;
    fGridOffs[i] = 0.;
    fArgsTmp[i]  = 0;
  }
  SetDimOut(DimOut,precD);
  PrepareBoundaries(bmin,bmax);
  DefineGrid(npoints);
  SetUsrFunction(ptr);
  ChebFit();
  //
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
AliCheb3D::AliCheb3D(void (*ptr)(float*,float*), int DimOut, Float_t  *bmin,Float_t  *bmax, Int_t *npX,Int_t *npY,Int_t *npZ, Float_t prec, const Float_t* precD) : 
  TNamed("Cheb3D","Cheb3D"),
  fDimOut(0), 
  fPrec(TMath::Max(fgkMinPrec,prec)), 
  fChebCalc(1), 
  fMaxCoefs(0), 
  fResTmp(0), 
  fGrid(0), 
  fUsrFunName(""),
  fUsrMacro(0)
{
  // Construct very economic  parameterization for the function
  // ptr     : pointer on the function: void fun(Float_t * inp,Float_t * out)
  // DimOut  : dimension of the vector computed by the user function
  // bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  // bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  // npX     : array of 3 elements with the number of points to compute in each dimension for 1st component 
  // npY     : array of 3 elements with the number of points to compute in each dimension for 2nd component 
  // npZ     : array of 3 elements with the number of points to compute in each dimension for 3d  component 
  // prec    : max allowed absolute difference between the user function and computed parameterization on the requested grid
  // precD   : optional array with precisions per output dimension (if >fgkMinPrec will override common prec)
  //
  if (DimOut<1) AliFatalF("Requested output dimension is %d",fDimOut);
  for (int i=3;i--;) {
    fBMin[i] = fBMax[i] = fBScale[i] = fBOffset[i] = 0;
    fNPoints[i] = 0;
    fGridOffs[i] = 0.;
    fArgsTmp[i]  = 0;
  }
  SetDimOut(DimOut,precD);
  PrepareBoundaries(bmin,bmax);
  SetUsrFunction(ptr);
  //
  DefineGrid(npX);
  ChebFit(0);
  DefineGrid(npY);
  ChebFit(1);
  DefineGrid(npZ);
  ChebFit(2);
  //
}
#endif


//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
AliCheb3D::AliCheb3D(void (*ptr)(float*,float*), int DimOut, Float_t  *bmin,Float_t  *bmax, Float_t prec, Bool_t run, const Float_t* precD) : 
  TNamed("Cheb3D","Cheb3D"),
  fDimOut(0), 
  fPrec(TMath::Max(fgkMinPrec,prec)), 
  fChebCalc(1), 
  fMaxCoefs(0), 
  fResTmp(0), 
  fGrid(0), 
  fUsrFunName(""),
  fUsrMacro(0)
{
  // Construct very economic  parameterization for the function with automatic calculation of the root's grid
  // ptr     : pointer on the function: void fun(Float_t * inp,Float_t * out)
  // DimOut  : dimension of the vector computed by the user function
  // bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  // bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  // prec    : max allowed absolute difference between the user function and computed parameterization on the requested grid
  // precD   : optional array with precisions per output dimension (if >fgkMinPrec will override common prec)
  //
  if (DimOut!=3) AliFatalF("This constructor works only for 3D fits, %dD fit was requested",fDimOut);
  if (DimOut<1)  AliFatalF("Requested output dimension is %d",fDimOut);
  for (int i=3;i--;) {
    fBMin[i] = fBMax[i] = fBScale[i] = fBOffset[i] = 0;
    fNPoints[i] = 0;
    fGridOffs[i] = 0.;
    fArgsTmp[i]  = 0;
  }
  SetDimOut(DimOut,precD);
  PrepareBoundaries(bmin,bmax);
  SetUsrFunction(ptr);
  //
  if (run) {
    int gridNC[3][3];
    EstimateNPoints(prec,gridNC);
    DefineGrid(gridNC[0]);
    ChebFit(0);
    DefineGrid(gridNC[1]);
    ChebFit(1);
    DefineGrid(gridNC[2]);
    ChebFit(2);
  }
  //
}
#endif


//__________________________________________________________________________________________
AliCheb3D& AliCheb3D::operator=(const AliCheb3D& rhs)
{
  // assignment operator
  //
  if (this != &rhs) {
    Clear();
    fDimOut   = rhs.fDimOut;
    fPrec     = rhs.fPrec;
    fMaxCoefs = rhs.fMaxCoefs;
    fUsrFunName = rhs.fUsrFunName;
    fUsrMacro   = 0;
    for (int i=3;i--;) {
      fBMin[i]    = rhs.fBMin[i];
      fBMax[i]    = rhs.fBMax[i];
      fBScale[i]  = rhs.fBScale[i];
      fBOffset[i] = rhs.fBOffset[i];
      fNPoints[i] = rhs.fNPoints[i];
    } 
    for (int i=0;i<fDimOut;i++) {
      AliCheb3DCalc* cbc = rhs.GetChebCalc(i);
      if (cbc) fChebCalc.AddAtAndExpand(new AliCheb3DCalc(*cbc),i);
    }    
  }
  return *this;
  //
}

//__________________________________________________________________________________________
void AliCheb3D::Clear(const Option_t*)
{
  // clear all dynamic structures
  //
  if (fResTmp)        { delete[] fResTmp; fResTmp = 0; }
  if (fGrid)          { delete[] fGrid;   fGrid   = 0; }
  if (fUsrMacro)      { delete fUsrMacro; fUsrMacro = 0;}
  fChebCalc.SetOwner(kTRUE);
  fChebCalc.Delete();
  //
}

//__________________________________________________________________________________________
void AliCheb3D::Print(const Option_t* opt) const
{
  // print info
  //
  printf("%s: Chebyshev parameterization for 3D->%dD function. Precision: %e\n",GetName(),fDimOut,fPrec);
  printf("Region of validity: [%+.5e:%+.5e] [%+.5e:%+.5e] [%+.5e:%+.5e]\n",fBMin[0],fBMax[0],fBMin[1],fBMax[1],fBMin[2],fBMax[2]);
  TString opts = opt; opts.ToLower();
  if (opts.Contains("l")) for (int i=0;i<fDimOut;i++) {printf("Output dimension %d:\n",i+1); GetChebCalc(i)->Print();}
  //
}

//__________________________________________________________________________________________
void AliCheb3D::PrepareBoundaries(const Float_t  *bmin, const Float_t  *bmax)
{
  // Set and check boundaries defined by user, prepare coefficients for their conversion to [-1:1] interval
  //
  for (int i=3;i--;) {
    fBMin[i]   = bmin[i];
    fBMax[i]   = bmax[i];
    fBScale[i] = bmax[i]-bmin[i];
    if (fBScale[i]<=0) { 
      AliFatalF("Boundaries for %d-th dimension are not increasing: %+.4e %+.4e\nStop\n",i,fBMin[i],fBMax[i]);
    }
    fBOffset[i] = bmin[i] + fBScale[i]/2.0;
    fBScale[i] = 2./fBScale[i];
  }
  //
}


//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_

// Pointer on user function (faster altrnative to TMethodCall)
void (*gUsrFunAliCheb3D) (float* ,float* );

void AliCheb3D::EvalUsrFunction() 
{
  // call user supplied function
  if   (gUsrFunAliCheb3D) gUsrFunAliCheb3D(fArgsTmp,fResTmp);
  else fUsrMacro->Execute(); 
}

void AliCheb3D::SetUsrFunction(const char* name)
{
  // load user macro with function definition and compile it
  gUsrFunAliCheb3D = 0; 
  fUsrFunName = name;
  gSystem->ExpandPathName(fUsrFunName);
  if (fUsrMacro) delete fUsrMacro;
  TString tmpst = fUsrFunName;
  tmpst += "+"; // prepare filename to compile
  if (gROOT->LoadMacro(tmpst.Data())) AliFatalF("Failed to load user function from %s",name);
  fUsrMacro = new TMethodCall();        
  tmpst = tmpst.Data() + tmpst.Last('/')+1; //Strip away any path preceding the macro file name
  int dot = tmpst.Last('.');
  if (dot>0) tmpst.Resize(dot);
  fUsrMacro->InitWithPrototype(tmpst.Data(),"Float_t *,Float_t *");
  long args[2];
  args[0] = (long)fArgsTmp;
  args[1] = (long)fResTmp;
  fUsrMacro->SetParamPtrs(args); 
  //
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
void AliCheb3D::SetUsrFunction(void (*ptr)(float*,float*))
{
  // assign user training function
  //
  if (fUsrMacro) delete fUsrMacro;
  fUsrMacro = 0;
  fUsrFunName = "";
  gUsrFunAliCheb3D = ptr;
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
void AliCheb3D::EvalUsrFunction(const Float_t  *x, Float_t  *res) 
{
  // evaluate user function value
  //
  for (int i=3;i--;) fArgsTmp[i] = x[i];
  if   (gUsrFunAliCheb3D) gUsrFunAliCheb3D(fArgsTmp,fResTmp);
  else fUsrMacro->Execute(); 
  for (int i=fDimOut;i--;) res[i] = fResTmp[i];
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
Int_t AliCheb3D::CalcChebCoefs(const Float_t  *funval,int np, Float_t  *outCoefs, Float_t  prec)
{
  // Calculate Chebyshev coeffs using precomputed function values at np roots.
  // If prec>0, estimate the highest coeff number providing the needed precision
  //
  double sm;                 // do summations in double to minimize the roundoff error
  for (int ic=0;ic<np;ic++) { // compute coeffs
    sm = 0;          
    for (int ir=0;ir<np;ir++) {
      float  rt = TMath::Cos( ic*(ir+0.5)*TMath::Pi()/np);
      sm += funval[ir]*rt;
    }
    outCoefs[ic] = Float_t( sm * ((ic==0) ? 1./np : 2./np) );
  }
  //
  if (prec<=0) return np;
  //
  sm = 0;
  int cfMax = 0;
  for (cfMax=np;cfMax--;) {
    sm += TMath::Abs(outCoefs[cfMax]);
    if (sm>=prec) break;
  }
  if (++cfMax==0) cfMax=1;
  return cfMax;
  //
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
void AliCheb3D::DefineGrid(Int_t* npoints)
{
  // prepare the grid of Chebyshev roots in each dimension
  const int kMinPoints = 1;
  int ntot = 0;
  fMaxCoefs = 1;
  for (int id=3;id--;) { 
    fNPoints[id] = npoints[id];
    if (fNPoints[id]<kMinPoints) AliFatalF("at %d-th dimension %d point is requested, at least %d is needed",id,fNPoints[id],kMinPoints);
    ntot += fNPoints[id];
    fMaxCoefs *= fNPoints[id];
  }
  printf("Computing Chebyshev nodes on [%2d/%2d/%2d] grid\n",npoints[0],npoints[1],npoints[2]);
  if (fGrid) delete[] fGrid;
  fGrid = new Float_t [ntot];
  //
  int curp = 0;
  for (int id=3;id--;) { 
    int np = fNPoints[id];
    fGridOffs[id] = curp;
    for (int ip=0;ip<np;ip++) {
      Float_t x = TMath::Cos( TMath::Pi()*(ip+0.5)/np );
      fGrid[curp++] = MapToExternal(x,id);
    }
  }
  //
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
Int_t AliCheb3D::ChebFit()
{
  // prepare parameterization for all output dimensions
  int ir=0; 
  for (int i=fDimOut;i--;) ir+=ChebFit(i); 
  return ir;
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
Int_t AliCheb3D::ChebFit(int dmOut)
{
  // prepare paramaterization of 3D function for dmOut-th dimension 
  int maxDim = 0;
  for (int i=0;i<3;i++) if (maxDim<fNPoints[i]) maxDim = fNPoints[i];
  Float_t  *fvals      = new Float_t [ fNPoints[0] ];
  Float_t  *tmpCoef3D  = new Float_t [ fNPoints[0]*fNPoints[1]*fNPoints[2] ]; 
  Float_t  *tmpCoef2D  = new Float_t [ fNPoints[0]*fNPoints[1] ]; 
  Float_t  *tmpCoef1D  = new Float_t [ maxDim ];
  //
  // 1D Cheb.fit for 0-th dimension at current steps of remaining dimensions
  int ncmax = 0;
  //
  printf("Dim%d : 00.00%% Done",dmOut);fflush(stdout);
  AliCheb3DCalc* cheb =  GetChebCalc(dmOut);
  //
  Float_t prec = cheb->GetPrecision(); 
  if (prec<fgkMinPrec) prec = fPrec;         // no specific precision for this dim.
  //
  Float_t rTiny = 0.1*prec/Float_t(maxDim); // neglect coefficient below this threshold
  //
  float ncals2count = fNPoints[2]*fNPoints[1]*fNPoints[0];
  float ncals = 0;
  float frac = 0;
  float fracStep = 0.001;
  //
  for (int id2=fNPoints[2];id2--;) {
    fArgsTmp[2] = fGrid[ fGridOffs[2]+id2 ];
    //
    for (int id1=fNPoints[1];id1--;) {
      fArgsTmp[1] = fGrid[ fGridOffs[1]+id1 ];
      //
      for (int id0=fNPoints[0];id0--;) {
	fArgsTmp[0] = fGrid[ fGridOffs[0]+id0 ];
	EvalUsrFunction();     // compute function values at Chebyshev roots of 0-th dimension
	fvals[id0] =  fResTmp[dmOut];
	float fr = (++ncals)/ncals2count;
	if (fr-frac>=fracStep) {
	  frac = fr;
	  printf("\b\b\b\b\b\b\b\b\b\b\b");
	  printf("%05.2f%% Done",fr*100);
	  fflush(stdout);
	}
	//
      }
      int nc = CalcChebCoefs(fvals,fNPoints[0], tmpCoef1D, prec);
      for (int id0=fNPoints[0];id0--;) tmpCoef2D[id1 + id0*fNPoints[1]] = tmpCoef1D[id0];
      if (ncmax<nc) ncmax = nc;              // max coefs to be kept in dim0 to guarantee needed precision
    }
    //
    // once each 1d slice of given 2d slice is parametrized, parametrize the Cheb.coeffs
    for (int id0=fNPoints[0];id0--;) {
      CalcChebCoefs( tmpCoef2D+id0*fNPoints[1], fNPoints[1], tmpCoef1D, -1);
      for (int id1=fNPoints[1];id1--;) tmpCoef3D[id2 + fNPoints[2]*(id1+id0*fNPoints[1])] = tmpCoef1D[id1];
    }
  }
  //
  // now fit the last dimensions Cheb.coefs
  for (int id0=fNPoints[0];id0--;) {
    for (int id1=fNPoints[1];id1--;) {
      CalcChebCoefs( tmpCoef3D+ fNPoints[2]*(id1+id0*fNPoints[1]), fNPoints[2], tmpCoef1D, -1);
      for (int id2=fNPoints[2];id2--;) tmpCoef3D[id2+ fNPoints[2]*(id1+id0*fNPoints[1])] = tmpCoef1D[id2]; // store on place
    }
  }
  //
  // now find 2D surface which separates significant coefficients of 3D matrix from nonsignificant ones (up to prec)
  UShort_t *tmpCoefSurf = new UShort_t[ fNPoints[0]*fNPoints[1] ];
  for (int id0=fNPoints[0];id0--;) for (int id1=fNPoints[1];id1--;) tmpCoefSurf[id1+id0*fNPoints[1]]=0;  
  Double_t resid = 0;
  for (int id0=fNPoints[0];id0--;) {
    for (int id1=fNPoints[1];id1--;) {
      for (int id2=fNPoints[2];id2--;) {
	int id = id2 + fNPoints[2]*(id1+id0*fNPoints[1]);
	Float_t  cfa = TMath::Abs(tmpCoef3D[id]);
	if (cfa < rTiny) {tmpCoef3D[id] = 0; continue;} // neglect coefs below the threshold
	resid += cfa;
	if (resid<prec) continue; // this coeff is negligible
	// otherwise go back 1 step
	resid -= cfa;
	tmpCoefSurf[id1+id0*fNPoints[1]] = id2+1; // how many coefs to keep
	break;
      }
    }
  }
  /*
  printf("\n\nCoeffs\n");  
  int cnt = 0;
  for (int id0=0;id0<fNPoints[0];id0++) {
    for (int id1=0;id1<fNPoints[1];id1++) {
      for (int id2=0;id2<fNPoints[2];id2++) {
	printf("%2d%2d%2d %+.4e |",id0,id1,id2,tmpCoef3D[cnt++]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */
  // see if there are rows to reject, find max.significant column at each row
  int nRows = fNPoints[0];
  UShort_t *tmpCols = new UShort_t[nRows]; 
  for (int id0=fNPoints[0];id0--;) {
    int id1 = fNPoints[1];
    while (id1>0 && tmpCoefSurf[(id1-1)+id0*fNPoints[1]]==0) id1--;
    tmpCols[id0] = id1;
  }
  // find max significant row
  for (int id0=nRows;id0--;) {if (tmpCols[id0]>0) break; nRows--;}
  // find max significant column and fill the permanent storage for the max sigificant column of each row
  cheb->InitRows(nRows);                  // create needed arrays;
  UShort_t *nColsAtRow = cheb->GetNColsAtRow();
  UShort_t *colAtRowBg = cheb->GetColAtRowBg();
  int nCols = 0;
  int nElemBound2D = 0;
  for (int id0=0;id0<nRows;id0++) {
    nColsAtRow[id0] = tmpCols[id0];     // number of columns to store for this row
    colAtRowBg[id0] = nElemBound2D;     // begining of this row in 2D boundary surface
    nElemBound2D += tmpCols[id0];
    if (nCols<nColsAtRow[id0]) nCols = nColsAtRow[id0];
  }
  cheb->InitCols(nCols);
  delete[] tmpCols;
  //  
  // create the 2D matrix defining the boundary of significance for 3D coeffs.matrix 
  // and count the number of siginifacnt coefficients
  //
  cheb->InitElemBound2D(nElemBound2D);
  UShort_t *coefBound2D0 = cheb->GetCoefBound2D0();
  UShort_t *coefBound2D1 = cheb->GetCoefBound2D1();
  fMaxCoefs = 0; // redefine number of coeffs
  for (int id0=0;id0<nRows;id0++) {
    int nCLoc = nColsAtRow[id0];
    int col0  = colAtRowBg[id0];
    for (int id1=0;id1<nCLoc;id1++) {
      coefBound2D0[col0 + id1] = tmpCoefSurf[id1+id0*fNPoints[1]];  // number of coefs to store for 3-d dimension
      coefBound2D1[col0 + id1] = fMaxCoefs;
      fMaxCoefs += coefBound2D0[col0 + id1];
    }
  }
  //
  // create final compressed 3D matrix for significant coeffs
  cheb->InitCoefs(fMaxCoefs);
  Float_t  *coefs = cheb->GetCoefs();
  int count = 0;
  for (int id0=0;id0<nRows;id0++) {
    int ncLoc = nColsAtRow[id0];
    int col0  = colAtRowBg[id0];
    for (int id1=0;id1<ncLoc;id1++) {
      int ncf2 = coefBound2D0[col0 + id1];
      for (int id2=0;id2<ncf2;id2++) {
	coefs[count++] = tmpCoef3D[id2 + fNPoints[2]*(id1+id0*fNPoints[1])];
      }
    }
  }
  /*
  printf("\n\nNewSurf\n");
  for (int id0=0;id0<fNPoints[0];id0++) {
    for (int id1=0;id1<fNPoints[1];id1++) {
      printf("(%2d %2d) %2d |",id0,id1,tmpCoefSurf[id1+id0*fNPoints[1]]);  
    }
    printf("\n");
  }
  */
  //
  delete[] tmpCoefSurf;
  delete[] tmpCoef1D;
  delete[] tmpCoef2D;
  delete[] tmpCoef3D;
  delete[] fvals;
  //
  printf("\b\b\b\b\b\b\b\b\b\b\b\b");
  printf("100.00%% Done\n");
  return 1;
}
#endif

//_______________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
void AliCheb3D::SaveData(const char* outfile,Bool_t append) const
{
  // writes coefficients data to output text file, optionallt appending on the end of existing file
  TString strf = outfile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf,append ? "a":"w");
  SaveData(stream);
  fclose(stream);
  //
}
#endif

//_______________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
void AliCheb3D::SaveData(FILE* stream) const
{
  // writes coefficients data to existing output stream
  //
  fprintf(stream,"\n# These are automatically generated data for the Chebyshev interpolation of 3D->%dD function\n",fDimOut); 
  fprintf(stream,"#\nSTART %s\n",GetName());
  fprintf(stream,"# Dimensionality of the output\n%d\n",fDimOut);
  fprintf(stream,"# Interpolation abs. precision\n%+.8e\n",fPrec);
  //
  fprintf(stream,"# Lower boundaries of interpolation region\n");
  for (int i=0;i<3;i++) fprintf(stream,"%+.8e\n",fBMin[i]);
  fprintf(stream,"# Upper boundaries of interpolation region\n");
  for (int i=0;i<3;i++) fprintf(stream,"%+.8e\n",fBMax[i]);
  fprintf(stream,"# Parameterization for each output dimension follows:\n");
  //
  for (int i=0;i<fDimOut;i++) GetChebCalc(i)->SaveData(stream);
  fprintf(stream,"#\nEND %s\n#\n",GetName());
  //
}
#endif

//__________________________________________________________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
void AliCheb3D::InvertSign()
{
  // invert the sign of all parameterizations
  for (int i=fDimOut;i--;) {
    AliCheb3DCalc* par =  GetChebCalc(i);
    int ncf = par->GetNCoefs();
    float *coefs = par->GetCoefs();
    for (int j=ncf;j--;) coefs[j] = -coefs[j];
  }
}
#endif


//_______________________________________________
void AliCheb3D::LoadData(const char* inpFile)
{
  // load coefficients data from txt file
  //
  TString strf = inpFile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf.Data(),"r");
  LoadData(stream);
  fclose(stream);
  //
}

//_______________________________________________
void AliCheb3D::LoadData(FILE* stream)
{
  // load coefficients data from stream
  //
  if (!stream) AliFatal("No stream provided");
  TString buffs;
  Clear();
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("START")) AliFatalF("Expected: \"START <fit_name>\", found \"%s\"",buffs.Data());
  SetName(buffs.Data()+buffs.First(' ')+1);
  //
  AliCheb3DCalc::ReadLine(buffs,stream); // N output dimensions
  fDimOut = buffs.Atoi(); 
  if (fDimOut<1) AliFatalF("Expected: '<number_of_output_dimensions>', found \"%s\"",buffs.Data());
  //
  SetDimOut(fDimOut);
  //
  AliCheb3DCalc::ReadLine(buffs,stream); // Interpolation abs. precision
  fPrec = buffs.Atof();
  if (fPrec<=0) AliFatalF("Expected: '<abs.precision>', found \"%s\"",buffs.Data());
  //
  for (int i=0;i<3;i++) { // Lower boundaries of interpolation region
    AliCheb3DCalc::ReadLine(buffs,stream);
    fBMin[i] = buffs.Atof(); 
  }
  for (int i=0;i<3;i++) { // Upper boundaries of interpolation region
    AliCheb3DCalc::ReadLine(buffs,stream);
    fBMax[i] = buffs.Atof(); 
  }
  PrepareBoundaries(fBMin,fBMax);
  //
  // data for each output dimension
  for (int i=0;i<fDimOut;i++) GetChebCalc(i)->LoadData(stream);
  //
  // check end_of_data record
  AliCheb3DCalc::ReadLine(buffs,stream);
  if (!buffs.BeginsWith("END") || !buffs.Contains(GetName())) {
    AliFatalF("Expected \"END %s\", found \"%s\"",GetName(),buffs.Data());
  }
  //
}

//_______________________________________________
void AliCheb3D::SetDimOut(const int d, const float* prec)
{
  // init output dimensions
  fDimOut = d;
  if (fResTmp) delete fResTmp;
  fResTmp = new Float_t[fDimOut];
  fChebCalc.Delete();
  for (int i=0;i<d;i++) {
    AliCheb3DCalc* clc = new AliCheb3DCalc();
    clc->SetPrecision(prec && prec[i]>fgkMinPrec ? prec[i] : fPrec);
    fChebCalc.AddAtAndExpand(clc,i);
  }
}

//_______________________________________________
void AliCheb3D::ShiftBound(int id,float dif)
{
  // modify the bounds of the grid
  //
  if (id<0||id>2) {printf("Maximum 3 dimensions are supported\n"); return;}
  fBMin[id] += dif;
  fBMax[id] += dif;
  fBOffset[id] += dif;
}

//_______________________________________________
#ifdef _INC_CREATION_ALICHEB3D_
TH1* AliCheb3D::TestRMS(int idim,int npoints,TH1* histo)
{
  // fills the difference between the original function and parameterization (for idim-th component of the output)
  // to supplied histogram. Calculations are done in npoints random points. 
  // If the hostgram was not supplied, it will be created. It is up to the user to delete it! 
  if (!fUsrMacro) {
    printf("No user function is set\n");
    return 0;
  }
  float prc = GetChebCalc(idim)->GetPrecision();
  if (prc<fgkMinPrec) prc = fPrec;   // no dimension specific precision
  if (!histo) histo = new TH1D(GetName(),"Control: Function - Parametrization",100,-2*prc,2*prc);
  for (int ip=npoints;ip--;) {
    gRandom->RndmArray(3,(Float_t *)fArgsTmp);
    for (int i=3;i--;) fArgsTmp[i] = fBMin[i] + fArgsTmp[i]*(fBMax[i]-fBMin[i]);
    EvalUsrFunction();
    Float_t valFun = fResTmp[idim];
    Eval(fArgsTmp,fResTmp);
    Float_t valPar = fResTmp[idim];
    histo->Fill(valFun - valPar);
  }
  return histo;
  //
}
#endif

//_______________________________________________
#ifdef _INC_CREATION_ALICHEB3D_

void AliCheb3D::EstimateNPoints(float prec, int gridBC[3][3],Int_t npd1,Int_t npd2,Int_t npd3)
{
  // Estimate number of points to generate a training data
  //
  const int kScp = 9;
  const float kScl[9] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
  //
  const float sclDim[2] = {0.001,0.999};
  const int   compDim[3][2] = { {1,2}, {2,0}, {0,1} };
  static float xyz[3];
  Int_t npdTst[3] = {npd1,npd2,npd3};
  //

  for (int i=3;i--;)for (int j=3;j--;) gridBC[i][j] = -1;
  //
  for (int idim=0;idim<3;idim++) {
    float dimMN = fBMin[idim] + sclDim[0]*(fBMax[idim]-fBMin[idim]);
    float dimMX = fBMin[idim] + sclDim[1]*(fBMax[idim]-fBMin[idim]);
    //
    int id1 = compDim[idim][0]; // 1st fixed dim
    int id2 = compDim[idim][1]; // 2nd fixed dim
    for (int i1=0;i1<kScp;i1++) {
      xyz[ id1 ] = fBMin[id1] + kScl[i1]*( fBMax[id1]-fBMin[id1] );
      for (int i2=0;i2<kScp;i2++) {
	xyz[ id2 ] = fBMin[id2] + kScl[i2]*( fBMax[id2]-fBMin[id2] );
	int* npt = GetNCNeeded(xyz,idim, dimMN,dimMX, prec, npdTst[idim]); // npoints for Bx,By,Bz
	for (int ib=0;ib<3;ib++) if (npt[ib]>gridBC[ib][idim]) gridBC[ib][idim] = npt[ib];
      }
    }
  }
}

/*
void AliCheb3D::EstimateNPoints(float prec, int gridBC[3][3])
{
  // Estimate number of points to generate a training data
  //
  const float sclA[9] = {0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9} ;
  const float sclB[9] = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9} ;
  const float sclDim[2] = {0.01,0.99};
  const int   compDim[3][2] = { {1,2}, {2,0}, {0,1} };
  static float xyz[3];
  //
  for (int i=3;i--;)for (int j=3;j--;) gridBC[i][j] = -1;
  //
  for (int idim=0;idim<3;idim++) {
    float dimMN = fBMin[idim] + sclDim[0]*(fBMax[idim]-fBMin[idim]);
    float dimMX = fBMin[idim] + sclDim[1]*(fBMax[idim]-fBMin[idim]);
    //
    for (int it=0;it<9;it++) { // test in 9 points
      int id1 = compDim[idim][0]; // 1st fixed dim
      int id2 = compDim[idim][1]; // 2nd fixed dim
      xyz[ id1 ] = fBMin[id1] + sclA[it]*( fBMax[id1]-fBMin[id1] );
      xyz[ id2 ] = fBMin[id2] + sclB[it]*( fBMax[id2]-fBMin[id2] );
      //
      int* npt = GetNCNeeded(xyz,idim, dimMN,dimMX, prec); // npoints for Bx,By,Bz
      for (int ib=0;ib<3;ib++) if (npt[ib]>gridBC[ib][idim]) gridBC[ib][idim] = npt[ib];//+2;
      //
    }
  }
}


int* AliCheb3D::GetNCNeeded(float xyz[3],int DimVar, float mn,float mx, float prec)
{
  // estimate needed number of chebyshev coefs for given function description in DimVar dimension
  // The values for two other dimensions must be set beforehand
  //
  static int curNC[3];
  static int retNC[3];
  const int kMaxPoint = 400;
  float* gridVal = new float[3*kMaxPoint];
  float* coefs   = new float[3*kMaxPoint];
  //
  float scale = mx-mn;
  float offs  = mn + scale/2.0;
  scale = 2./scale;
  // 
  int curNP;
  int maxNC=-1;
  int maxNCPrev=-1;
  for (int i=0;i<3;i++) retNC[i] = -1;
  for (int i=0;i<3;i++) fArgsTmp[i] = xyz[i];
  //
  for (curNP=3; curNP<kMaxPoint; curNP+=3) { 
    maxNCPrev = maxNC;
    //
    for (int i=0;i<curNP;i++) { // get function values on Cheb. nodes
      float x = TMath::Cos( TMath::Pi()*(i+0.5)/curNP );
      fArgsTmp[DimVar] =  x/scale+offs; // map to requested interval
      EvalUsrFunction();
      for (int ib=3;ib--;) gridVal[ib*kMaxPoint + i] = fResTmp[ib];
    }
    //
    for (int ib=0;ib<3;ib++) {
      curNC[ib] = AliCheb3D::CalcChebCoefs(&gridVal[ib*kMaxPoint], curNP, &coefs[ib*kMaxPoint],prec);
      if (maxNC < curNC[ib]) maxNC = curNC[ib];
      if (retNC[ib] < curNC[ib]) retNC[ib] = curNC[ib];
    }
    if ( (curNP-maxNC)>3 &&  (maxNC-maxNCPrev)<1 ) break;
    maxNCPrev = maxNC;
    //
  }
  delete[] gridVal;
  delete[] coefs;
  return retNC;
  //
}
*/


int* AliCheb3D::GetNCNeeded(float xyz[3],int DimVar, float mn,float mx, float prec, Int_t npCheck)
{
  // estimate needed number of chebyshev coefs for given function description in DimVar dimension
  // The values for two other dimensions must be set beforehand
  //
  static int retNC[3];
  static int npChLast = 0;
  static float *gridVal=0,*coefs=0;
  if (npCheck<3) npCheck = 3;
  if (npChLast<npCheck) {
    if (gridVal) delete[] gridVal;
    if (coefs)   delete[] coefs;
    gridVal = new float[3*npCheck];
    coefs   = new float[3*npCheck];
    npChLast = npCheck;
  }
  //
  float scale = mx-mn;
  float offs  = mn + scale/2.0;
  scale = 2./scale;
  //
  for (int i=0;i<3;i++) fArgsTmp[i] = xyz[i];
  for (int i=0;i<npCheck;i++) {
    fArgsTmp[DimVar] =  TMath::Cos( TMath::Pi()*(i+0.5)/npCheck)/scale+offs; // map to requested interval
    EvalUsrFunction();
    for (int ib=3;ib--;) gridVal[ib*npCheck + i] = fResTmp[ib];
  } 
  //
  for (int ib=0;ib<3;ib++) retNC[ib] = AliCheb3D::CalcChebCoefs(&gridVal[ib*npCheck], npCheck, &coefs[ib*npCheck],prec);
  return retNC;
  //
}


#endif
