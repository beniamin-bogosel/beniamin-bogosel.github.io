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
for i=0 upto he:
%draw (0, i*u)--(breite, i*u) withcolor .7white;
endfor
for j=0 upto wi:
%draw (j*u, 0)--(j*u, hoehe) withcolor .7white;
endfor;
path p[],pa;
p1:=(-0.01u,6u)..(5u,5u)..(10u,1.7u)--(10u,0u)..(5u,0)..(-0.01u,-0.01u)--(-0.01u,u)..(5u,u)..{dir 0}(9u,1u){dir 120}..(5u,3.9u)..(-0.01u,5u)--cycle;
pa:=(3u,5u)..(5u,6u)..(8u,6u)..(8u,3.8u)..(7.5u,3.4u)..(7u,3u)..cycle;
%draw p1;
fill p1 withcolor .8white;
pair a,b;
h=length(p1);
i=5;
numeric c,d;
c:=0.57*h;
d:=0.59*h;
a:= point c of p1;
b:= point d of p1;
pickup pencircle scaled 1.5;
%draw a;
%draw b;
path q,r,s,qu;
q:= subpath (c,d) of p1;
ce:=0.86*h;
de:=0.88*h;
qu:= subpath (ce,de) of p1;
%draw q;
%draw qu;
r:= b..((a+b)/2 shifted (.1(a-b) rotated -90))..a;
%draw r;
path supp[];
pair au,bu;
au:= point ce of p1;
bu:= point de of p1;
supp1 :=q--au..qu--cycle;

numeric cc[],dd[];
c1:= 0.65*h;
d1:= 0.67*h;
p2:= subpath(c1,d1) of p1;
%draw p2;

c2:= 0.783*h;
d2:= 0.8*h;
p3:= subpath(c2,d2) of p1;
%draw p3;
pair as[],bs[];
as1 = point c1 of p1;
bs1 = point d1 of p1;
as2 = point c2 of p1;
bs2 = point d2 of p1;
supp2:= p2--as2--p3--cycle;
%draw supp2;
fill supp2 withcolor .8white;
 
c3:= 0.3*h;
d3:= 0.318*h;
as3:= point c3 of p1;
bs3:= point d3 of p1;
p4:= subpath(c3,d3) of p1;
%draw p4;

supp3:= p4--(7.5u,0)--(8.5u,0)--cycle;
%draw supp3;
%fill supp3 withcolor .9red;

fill supp1 withcolor .8white;

%fill p1 xscaled -1 withcolor .8white;
%fill supp1 xscaled -1 withcolor .8white;
%fill supp2 xscaled -1 withcolor .8white;







p5:= subpath(d2,ce) of p1;

draw p5 withcolor red;

numeric vv;
vv:=0.909*h;
p6:= subpath(de,vv) of p1;

draw p6 withcolor red;

h:=length(p5);

pair arr[];

numeric N;
N = 10;

for j=0 upto N:
arr1:=  point j*h/N of p5;
arr2:=arr1+(0,-u);

drawarrow arr1--arr2 withcolor blue;
endfor

for j=0 upto N:
arr1:=  point j*h/N of p6;
arr2:=arr1+(0,-u);

drawarrow arr1--arr2 withcolor blue;
endfor



%pair arr[];
%arr1:=(10u,3.8u);
%arr2:=(10u,1.8u);
%drawarrow arr1--arr2 withcolor red;
%label.rt(btex $g$ etex,.5(10u,2.8u)) scaled 2 withcolor red;

%arr1:=(-9u,(hh-0.25)*u);
%arr2:=(-9u,(hh-2.25)*u);
%drawarrow arr1--arr2 withcolor blue;



%label.rt(btex $f$ etex,.5(2u,2.8u)) scaled 2 withcolor blue;


%draw supp1;

%fill pa withcolor .9999blue;
%label.rt(btex $\omega$ etex,.5(7u,4.8u))  scaled 2;
%label.rt(btex $S$ etex,.5(1.5u,3.5u)) scaled 2;
%label.rt(btex $S$ etex,.5(5.6u,3.5u)) scaled 2;
%label.rt(btex $S$ etex,.5(7.5u,0.8u)) scaled 2;

%path pt; pt:=fullcircle scaled u shifted (5u,3u);
%fill pt withcolor .7white;
%label(btex $B_\rho$ etex,(5u,2.5u)) shifted (.1u,.3u);
%label.top(btex $L$ etex,2/3(6.5u,u)) scaled 1.5 shifted (-.2u,-.5u);

endfig;
end
