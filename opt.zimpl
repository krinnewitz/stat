set components := {0,1,2,3,4,5,6,7,8,9,10,11,12,13};

var c[components] real >= 0.1;

set textures := {1,2,3,4,5,6,7,8,9,10,11,12};

param x[textures * textures * components] := read "stats.txt" as "<1n 2n 3n> 4n" comment "#";

maximize distance:
  sum <i,j,k> in textures cross textures cross components: c[k] * x[i,j,k];

subto pommes:
  sum <i,k> in textures cross components: c[k] * x[i,i,k] <= 15000;

subto c0:
  c[0] <= 1000;
subto c1:
  c[1] <= 1000;
subto c2:
  c[2] <= 1000;
subto c3:
  c[3] <= 1000;
subto c4:
  c[4] <= 1000;
subto c5:
  c[5] <= 1000;
subto c6:
  c[6] <= 1000;
subto c7:
  c[7] <= 1000;
subto c8:
  c[8] <= 1000;
subto c9:
  c[9] <= 1000;
subto c10:
  c[10] <= 1000;
subto c11:
  c[11] <= 1000;
subto c12:
  c[12] <= 1000;
subto c13:
  c[13] <= 1000;
