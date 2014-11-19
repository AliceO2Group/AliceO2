
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class AliceO2::Field::Chebyshev3D+;
#pragma link C++ class AliceO2::Field::MagneticField+;
#pragma link C++ class AliceO2::Field::MagneticWrapperChebyshev+;
#pragma link C++ class AliceO2::Field::Chebyshev3DCalc+;

#pragma read sourceClass="AliCheb3D" targetClass="AliceO2::Field::Chebyshev3D" version="[1-]" source="Int_t fDimOut" target="Int_t mOutputArrayDimension" code="{ mOutputArrayDimension = onfile.fDimOut; }"    


//Int_t        fDimOut;            // dimension of the ouput array
//Float_t      fPrec;              // requested precision
//Float_t      fBMin[3];           // min boundaries in each dimension
//Float_t      fBMax[3];           // max boundaries in each dimension
//Float_t      fBScale[3];         // scale for boundary mapping to [-1:1] interval
//Float_t      fBOffset[3];        // offset for boundary mapping to [-1:1] interval
//TObjArray    fChebCalc;          // Chebyshev parameterization for each output dimension


//Int_t mOutputArrayDimension;       ///< dimension of the ouput array
//Float_t mPrecision;                ///< requested precision
//Float_t mMinBoundaries[3];         ///< min boundaries in each dimension
//Float_t mMaxBoundaries[3];         ///< max boundaries in each dimension
//Float_t mBoundaryMappingScale[3];  ///< scale for boundary mapping to [-1:1] interval
//Float_t mBoundaryMappingOffset[3]; ///< offset for boundary mapping to [-1:1] interval
//TObjArray mChebyshevParameter;     ///< Chebyshev parameterization for each output dimension





#endif
