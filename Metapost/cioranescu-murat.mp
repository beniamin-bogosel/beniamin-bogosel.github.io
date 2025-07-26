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

path p,pa;
p:=(0,4u)..(0,2u)..(2u,2u)..(3u,u)..(3u,0)..(6u,u)..(7u,3u)..(9u,5u)..cycle;
pa:=(3u,5u)..(5u,6u)..(8u,6u)..(8u,3.8u)..(7.5u,3.4u)..(7u,3u)..cycle;
draw p;
fill p withcolor .8white;
pair a,b;
h=length(p);
i=5;
numeric c,d;
c:=0.65*h;
d:=0.7*h;
a:= point c of p;
b:= point d of p;
pickup pencircle scaled 1.5;
%draw a;
%draw b;
path q,r,s;
q:= subpath (c,d) of p;
%draw q withcolor red;
%r:= b..((a+b)/2 shifted (.5(a-b) rotated -90))..a;
%draw r;
%fill buildcycle(r,q) withcolor .5red;
%fill pa withcolor .9999blue;
%label.rt(btex $\Omega$ etex,.5(u,3u))  scaled 2;

draw (-u,9u)-- (10u,9u)--(10u,-2u)--(-u,-2u)--cycle;
for i=0 upto 9:
for j=-1 upto 8:
    draw fullcircle scaled 0.3u shifted (i*u, j*u);
    fill fullcircle scaled 0.3u shifted (i*u, j*u) withcolor white;
endfor
endfor;
endfig;
end;
