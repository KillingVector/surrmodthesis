Merge "base.stl";
Physical Surface("VEHICLE") = {1};
Physical Surface("SYMPLANE") = {2};
Physical Surface("FARFIELD") = {3};
Surface Loop(4) = {3};
Surface Loop(5) = {2};
Surface Loop(6) = {1};
Volume(6) = {4, 5, 6};
