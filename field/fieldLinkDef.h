
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class AliceO2::Field::Chebyshev3D+;
#pragma link C++ class AliceO2::Field::MagneticField+;
#pragma link C++ class AliceO2::Field::MagneticWrapperChebyshev+;
#pragma link C++ class AliceO2::Field::Chebyshev3DCalc+;

//map the old to the new variables:
//AliCheb3D-------OLD
//Int_t        fDimOut;            // dimension of the ouput array
//Float_t      fPrec;              // requested precision
//Float_t      fBMin[3];           // min boundaries in each dimension
//Float_t      fBMax[3];           // max boundaries in each dimension
//Float_t      fBScale[3];         // scale for boundary mapping to [-1:1] interval
//Float_t      fBOffset[3];        // offset for boundary mapping to [-1:1] interval
//TObjArray    fChebCalc;          // Chebyshev parameterization for each output dimension

//Chebyshev3D-------New
//Int_t mOutputArrayDimension;       ///< dimension of the ouput array
//Float_t mPrecision;                ///< requested precision
//Float_t mMinBoundaries[3];         ///< min boundaries in each dimension
//Float_t mMaxBoundaries[3];         ///< max boundaries in each dimension
//Float_t mBoundaryMappingScale[3];  ///< scale for boundary mapping to [-1:1] interval
//Float_t mBoundaryMappingOffset[3]; ///< offset for boundary mapping to [-1:1] interval
//TObjArray mChebyshevParameter;     ///< Chebyshev parameterization for each output dimension

#pragma read sourceClass="AliCheb3D" \
             targetClass="AliceO2::Field::Chebyshev3D" \
             version="[1-]" \
             source="Int_t fDimOut" \
             target="mOutputArrayDimension" \
             code="{ mOutputArrayDimension = onfile.fDimOut; }"

#pragma read sourceClass="AliCheb3D" \
            targetClass="AliceO2::Field::Chebyshev3D" \
            version="[1-]" \
            source="Float_t fPrec" \
            target="mPrecision" \
            code="{ mPrecision = onfile.fPrec; }"


#pragma read sourceClass="AliCheb3D" \
            targetClass="AliceO2::Field::Chebyshev3D" \
            version="[1-]" \
            source="Float_t fBMin[3]" \
            target="mMinBoundaries" \
            code="{ mMinBoundaries[0] = onfile.fBMin[0];mMinBoundaries[1] = onfile.fBMin[1];mMinBoundaries[2] = onfile.fBMin[2]; }"

#pragma read sourceClass="AliCheb3D" \
            targetClass="AliceO2::Field::Chebyshev3D" \
            version="[1-]" \
            source="Float_t fBMax[3]" \
            target="mMaxBoundaries" \
            code="{ mMaxBoundaries[0] = onfile.fBMax[0];mMaxBoundaries[1] = onfile.fBMax[1];mMaxBoundaries[2] = onfile.fBMax[2]; }"

#pragma read sourceClass="AliCheb3D" \
            targetClass="AliceO2::Field::Chebyshev3D" \
            version="[1-]" \
            source="Float_t fBScale[3]" \
            target="mBoundaryMappingScale" \
            code="{ mBoundaryMappingScale[0] = onfile.fBScale[0];mBoundaryMappingScale[1] = onfile.fBScale[1];mBoundaryMappingScale[2] = onfile.fBScale[2]; }"

#pragma read sourceClass="AliCheb3D" \
            targetClass="AliceO2::Field::Chebyshev3D" \
            version="[1-]" \
            source="Float_t fBOffset[3]" \
            target="mBoundaryMappingScale" \
            code="{ mBoundaryMappingScale[0] = onfile.fBOffset[0];mBoundaryMappingScale[1] = onfile.fBOffset[1];mBoundaryMappingScale[2] = onfile.fBOffset[2]; }"

#pragma read sourceClass="AliCheb3D" \
            targetClass="AliceO2::Field::Chebyshev3D" \
            version="[1-]" \
            source="TObjArray fChebCalc" \
            target="mChebyshevParameter" \
            code="{ mChebyshevParameter = onfile.fChebCalc;}"












#endif
