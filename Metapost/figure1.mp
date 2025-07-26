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
draw (0, i*u)--(breite, i*u) withcolor .7white;
endfor
for j=0 upto wi:
draw (j*u, 0)--(j*u, hoehe) withcolor .7white;
endfor;
path p,pa;
p:=(0,4u)..(0,2u)..(2u,2u)..(3u,u)..(3u,0)..(6u,u)..(7u,3u)..(9u,5u)..cycle;
pa:=(3u,5u)..(5u,6u)..(8u,6u)..(8u,3.8u)..(7.5u,3.4u)..(7u,3u)..cycle;
draw p;
fill p withcolor .8white;
pair a,b;
h=length(p);
i=5;
numeric c,d;
c:=0.6*h;
d:=0.7*h;
a:= point c of p;
b:= point d of p;
pickup pencircle scaled 1.5;
draw a;
draw b;
path q,r,s;
q:= subpath (c,d) of p;
draw q;
r:= b..((a+b)/2 shifted (.1(a-b) rotated -90))..a;
draw r;
fill buildcycle(r,q) withcolor .9red;
fill pa withcolor .9999blue;
label.rt(btex $\Omega$ etex,.5(u,3u))  scaled 2;
label.rt(btex $E$ etex,.5(5u,5u)) scaled 2;
path pt; pt:=fullcircle scaled u shifted (5u,3u);
fill pt withcolor .7white;
label(btex $B_\rho$ etex,(5u,2.5u)) shifted (.1u,.3u);
label.top(btex $L$ etex,2/3(6.5u,u)) scaled 1.5 shifted (-.2u,-.5u);
endfig;
end;
