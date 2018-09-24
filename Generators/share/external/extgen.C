/** 
    It is mandatory that the function returns a TGenerator* 
    whereas there are no restrictions on the function name
    and the arguments to the function prototype.

    TGenerator *extgen(double energy = 2760.);
    
    It is mandatory to define the units used by the
    concerned generator to allow for the proper
    conversion to the units used by the simulation.
    The above is done by defining the following
    variable mandatory global variables 

    double momentumUnit; // [GeV/c]
    double energyUnit;   // [GeV/c]
    double positionUnit; // [cm]
    double timeUnit;     // [s]

    and assign the proper values, either initialising them
    or withing the function to be called
**/

double momentumUnit = 1.;
double energyUnit = 1.;
double positionUnit = 1.;
double timeUnit = 1.;

TGenerator*
  extgen()
{
  std::cout << "This is a template function for an external generator" << std::endl;
  auto gen = new TGenerator;
  return gen;
}
