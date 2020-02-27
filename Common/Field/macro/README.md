<!-- doxy
\page refCommonFieldMacros Field Macros
/doxy -->

## Macros for magnetic field manipulations

```o2::field::MagneticWrapperChebyshev``` class allows to dump the field data to text file and recreate it back as a ROOT object from this file. This is useful if one needs to change the name or the namespace of this class or persistent classes it usese internally (e.g. ``MathUtils/Chebyshev3D.h``, ``MathUtils/Chebyshev3DCalc.h``)

* macro ``extractMapsAsText.C``

Converts all magnetics field objects of MapClass in the inpFileName file to text files named as <prefix><map_name> in current directory, e.g.
``root -b -q 'extractMapsAsText.C+("$O2_ROOT/share/Common/maps/mfchebKGI_sym.root")'``

*  macro ``createMapsFromText.C``

Converts all text files with name pattern ``<path>/<prefix>*`` to magnetic field object of MapClass and stores them in the outFileName root file, e.g.
``root -b -q 'createMapsFromText.C+'``
will create local ``mfchebKGI_sym.root``, which can substitute ``O2/Common/maps/mfchebKGI_sym.root`` file.

Currently the MapClass is aliased to ``o2::field::MagneticWrapperChebyshev`` in both cases. If after ``extractMapsAsText.C`` macro the name of the underlying MapClass changes, this has to be reflected in the ``createMapsFromText.C``

