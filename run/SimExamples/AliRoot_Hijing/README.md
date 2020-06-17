<!-- doxy
\page refrunSimExamplesAliRoot_Hijing Example AliRoot_Hijing
/doxy -->

This is a simple simulation example showing how to run event simulation using the Hijing event generator interface from AliRoot.
The configuration of AliGenHijing is performed by the function `hijing(double energy = 5020., double bMin = 0., double bMax = 20.)` defined in the macro `aliroot_hijing.macro`.

The macro file is specified via the argument of `--extGenFile` whereas the specific function call to retrieven the configuration is specified via the argument of `--extGenFunc`.
 
# IMPORTANT
To run this example you need to load an AliRoot package compatible with the O2.
for more details, https://alice.its.cern.ch/jira/browse/AOGM-246

AliRoot needs to be loaded **after** O2 in the following sense:
`alienv enter O2/latest,AliRoot/latest`
The other order may show unresolved symbol problems.
