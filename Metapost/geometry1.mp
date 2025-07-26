prologues:=3;
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

uu = 10*u;

draw fullcircle scaled u;
draw fullcircle scaled uu;
fill fullcircle scaled u withcolor 0.8white ;

pair cen[];
cen1:=(uu+u,0);

path p[];
p1:= fullcircle scaled uu shifted cen1;
draw p1;
h=length(p1);

numeric c,d;
c = -0.25*h;
d = (-0.25-1/12)*h;

pair A[];
A1:= point c of p1;
draw A1 withpen pencircle scaled 4bp;

A2:= point d of p1;
draw A2 withpen pencircle scaled 4bp;

p2:= subpath (c,d) of p1;
p3:= p2 -- cen1 -- cycle;
draw p3;
fill p3 withcolor 0.8white;


%draw supp1;

%fill pa withcolor .9999blue;
label.rt(btex $\Omega$ etex,.5(2u,2u))  scaled 2;
label(btex $\omega$ etex,.5(0,0)) scaled 2;


label.rt(btex $\Omega$ etex,.5(2u,2u))  scaled 2 shifted cen1;
label(btex $\omega$ etex,.5(-u,-3u)) scaled 2 shifted cen1;







endfig;
end;
