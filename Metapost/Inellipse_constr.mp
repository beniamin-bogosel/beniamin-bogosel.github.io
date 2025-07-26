outputformat := "png";
outputtemplate := "%j-%c.%o";

hppp := 0.15;
vppp := 0.15;

prologues:=3;

input hatching;


input cmarrows;
setup_cmarrows(
      brace_name     = "bigbrace";
      parameter_file = "cmr10.mf";
      macro_name     = "bracea");
    setup_cmarrows(
      brace_name     = "Biggbrace";
      parameter_file = "cmr12.mf";
      macro_name     = "braceb");
    setup_cmarrows(
      brace_name     = "extensiblebrace";
      parameter_file = "cmr9.mf";
      macro_name     = "bracec");

vardef TEX primary s =
write "verbatimtex"                    to "mptextmp.mp";
write "\documentclass[12pt]{article}"  to "mptextmp.mp"; 
write "\usepackage[T1]{fontenc}"       to "mptextmp.mp";
write "\usepackage[ansinew]{inputenc}" to "mptextmp.mp";
write "\usepackage{amsmath,amssymb}"   to "mptextmp.mp";
write "\begin{document}"               to "mptextmp.mp";
write "etex"                           to "mptextmp.mp";
write "btex "&s&" etex"                to "mptextmp.mp";
write EOF                              to "mptextmp.mp";
scantokens "input mptextmp"
enddef;

verbatimtex
%&latex
\documentclass{minimal}
\begin{document}
etex
beginfig(0);


u:=25; % 25 = 25bp = 25 PostScript points = 25/72 in
wi:=10; % width in units u
he:=7; % height in units u
hoehe:=he*u; % height
breite:=wi*u;

pair a[];
pair b[];

a1 = (-0.3u,4u);
a2 = (-3u,0);
a3 = (5u,0);


draw a1 withpen pencircle scaled 3;
draw a2 withpen pencircle scaled 3;
draw a3 withpen pencircle scaled 3;

label.top(btex $\mathbf{a}$ etex,a1);
label.llft(btex $\mathbf{b}$ etex,a2);
label.lrt(btex $\mathbf{c}$ etex,a3);

numeric t;
numeric x,y,z;

pair A,B,C,D,E,F;

x = 4;
y = 1.3;
z = 1.5;
t1 = 1;
t2 = 1.3;
t3 = 1/(t1*t2);

b2 = x/(x+z)*a1+z/(x+z)*a3;
%label.rt(btex $b_2$ etex,b2);
b3 = x/(x+y)*a1+y/(x+y)*a2;
b1 = y/(y+z)*a2+z/(y+z)*a3;




%draw a1--b1 dashed evenly withcolor 0.8white;
%draw a2--b2 dashed evenly withcolor 0.8white;
%draw a3--b3 dashed evenly withcolor 0.8white;

pair c[];

% midpoints for finding center

draw b1--b2 withcolor 0.8white dashed evenly;
draw b1--b3 withcolor 0.8white dashed evenly;
c3 = 0.5(b1+b2);
c2 = 0.5(b1+b3);
label.bot(btex $\mathbf{r}$ etex,c3);
label.bot(btex $\mathbf{q}$ etex,c2);

path p[];

numeric f[];
f2 = x/(x+y+z);
f3 = x/(x+z+y);
%if(f1>f2):f1=f2; fi
p2 = a2--(c2+(c2-a2)/f2);
p3 = a3--(c3+(c3-a3)/f3);
%draw p2;
%draw p3;

% find center
pair cen;
cen = p2 intersectionpoint p3;

draw a2--cen withcolor 0.7white;
draw a3--cen withcolor 0.7white;




% symmetric of m
b4 = 2*cen-b1;

label.top(btex $\mathbf{m}'$ etex,b4);



pair d[];
d1 = cen+0.5(a3-a2);
d2 = cen+0.5(a2-a3);
draw d1--d2 withcolor 0.3red+0.7white dashed evenly;

d3 = (a1--a3) intersectionpoint (d1--d2);

numeric fact;
fact = 0.9;
label.urt(btex $\mathbf{s}$ etex,1/fact*d3) scaled fact shifted (0,-0.05u);

d4 = (d1--d2) intersectionpoint (b2--(b2+(b1-b4)));
label.urt(btex $\mathbf{s}'$ etex,1/fact*d4) scaled fact;

numeric l[];
l1 = sqrt(arclength(cen--d3)*arclength(cen--d4));
d5 = cen+l1*(d4-cen)/arclength(d4--cen);
label.llft(btex $\mathbf{d}$ etex,1/fact*d5) scaled fact;
d6 = 2*cen-d5;
label.llft(btex $\mathbf{d}'$ etex,1/fact*d6) scaled fact;

draw a1--a3;
draw a1--a2;
draw a2--a3;
draw d3  withpen pencircle scaled 2;
draw d4  withpen pencircle scaled 2;
draw d5  withpen pencircle scaled 2 withcolor 0.5red;
draw d6  withpen pencircle scaled 2 withcolor 0.5red;

% rotate one diameter
d7 = b4+((d5-cen) rotated 90);
d8 = b4+((d5-cen) rotated -90);
label.lft(btex $\mathbf{e}$ etex,1/fact*d7) scaled fact;
label.lft(btex $\mathbf{e}'$ etex,1/fact*d8) scaled fact;

draw cen--d7;
draw cen--d8;

d9 = cen+50*((d8-cen)/arclength(d8--cen)+(d7-cen)/arclength(d7--cen));


draw d7  withpen pencircle scaled 2 withcolor 0.5red;
draw d8  withpen pencircle scaled 2 withcolor 0.5red;
draw d9  withpen pencircle scaled 2 withcolor 0.5red;

draw (2*cen-d9)--d9 withcolor red;

draw b1--b4 withcolor 0.8white;
draw d7--d8 withcolor 0.8white;

draw b1 withpen pencircle scaled 3;
draw b2 withpen pencircle scaled 3;
draw b3 withpen pencircle scaled 3;
draw b4  withpen pencircle scaled 3 withcolor 0.5red;

label.bot(btex $\mathbf{m}$ etex,b1);
label.urt(btex $\mathbf{n}$ etex,b2);
label.ulft(btex $\mathbf{p}$ etex,b3);

draw c2  withpen pencircle scaled 3 withcolor 0.5red;
draw c3  withpen pencircle scaled 3 withcolor 0.5red;

draw cen withpen pencircle scaled 3 withcolor 0.5red;
label.bot(btex $\mathbf{o}$ etex,cen);

path lin;
lin = (2*cen-d9)--d9;
pair pt;
pt = point 0.05 of lin;
label.rt(btex $\ell$ etex,pt);

numeric la,lb;
l2 = arclength(cen--d7);
l3 = arclength(cen--d8);

la = l2+l3;
lb = abs(l2-l3);

d10 = cen+0.5*la*(d9-cen)/arclength(d9--cen);
d11 = cen+0.5*lb*((d9-cen)/arclength(d9--cen) rotated 90);

%draw d10  withpen pencircle scaled 2 withcolor 0.5red;
%draw d11  withpen pencircle scaled 2 withcolor 0.5red;


path ell;
ell = fullcircle xscaled la yscaled lb rotated angle(d9-cen) shifted cen;



draw ell dashed evenly withcolor blue; 


%label.rt(btex $\mathbf{a}_0$ etex,a0);
%label.rt(btex $\mathbf{a}_1$ etex,a1);
%%label.rt(btex $\mathbf{a}_{n-1}$ etex,a2);
%label.lft(btex $\mathbf{c}$ etex,c);
%label.top(btex $\varepsilon$ etex,0.6a0+0.4c);

endfig;
end;
