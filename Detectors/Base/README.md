# devtips

In order to get your detector implementation accessible through a simple interface :

```c++
bool isActive{true};
auto det = o2::Base::createFairModule("DET",isActive);
```

you must register "somewhere" (e.g. in DET/simulation/src/Detector.cxx) a function which is able to create the module for your detector.

```c++
namespace { 
o2::Base::FairModuleRegister myDetCreator("DET", 
   [](bool active) -> FairModule* { return new MyBeautifullDetClass(active, whatever_other_params_you_need); });
}
```

In the example above what is registered is a lambda but you can of course register a "regular" function as well.
Whatever, the function must take a boolean (indicating if the detector is an active one) and return some daughter class of `FairModule`.

Note that the anonymous namespace in the example above is not strictly mandatory but does not hurt either.
