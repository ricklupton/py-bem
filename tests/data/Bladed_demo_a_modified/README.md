The Bladed runs in this folder use a modified version of the standard
Bladed demo_a model:

 - The original thicknesses specified in the blade screen weren't
   being honoured because the correct interpolation of foils wasn't
   set up. So I changed the specified thicknesses to match what Bladed
   was actually calculating.
 
 - The tangential induction factor calculated for the cylinder
   sections didn't agree. This is apparently a bug in Bladed when the
   sign of the tangential induction factor change [according to
   James]. Changing the blade in Bladed to have 21% aerofoil sections
   right to the root gives good agreement.

 - All simulations are run without tip loss and gravity