Calculates trajectories of magnetic nanoparticles passing over an
(Halbach) array of magnets, as presented in

Michiel Stevens, Peng Liu, Tom Niessink, Anouk Mentink, Leon Abelmann
and Leon W. M. M. Terstappen "Optimal Halbach configuration for
flow-through immunomagnetic CTC enrichment".

TrajectoryArray.py : Calculation of trajectories
config.py               : Calculation parameters

Field calculations are performed by integration over charges.
FieldSquare.py      : Field of a uniformely charged rectangle
FieldBlock.py        : Field of a bar magnet (with two FieldSquares of
opposite sign)
FieldDIpole.py      : Field of a dipole, for approximation bar magnets
at large distance
FieldArray.py        : Field of array of bar magnets (FieldBlocks)

