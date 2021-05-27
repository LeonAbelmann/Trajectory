Calculates trajectories of magnetic nanoparticles passing over an
(Halbach) array of magnets, as presented in

Michiel Stevens, Peng Liu, Tom Niessink, Anouk Mentink, Leon Abelmann
and Leon W. M. M. Terstappen "Optimal Halbach configuration for
flow-through immunomagnetic CTC enrichment".

<ul>
<li>Trajectory.py : Calculation of trajectories
<li>config.py     : Calculation parameters
</ul>
 
Field calculations are performed by integration over charges.
<ul>
<li>FieldSquare.py      : Field of a uniformely charged rectangle
<li>FieldBlock.py        : Field of a bar magnet (with two FieldSquares of
opposite sign)
<li>FieldDipole.py      : Field of a dipole, for approximation bar magnets
at large distance
<li>FieldArray.py        : Field of array of bar magnets (FieldBlocks)
</ul>
