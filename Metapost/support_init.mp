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
p1:=(0,10u)..(5u,8u)..(8u,2.5u)--(8u,0u)--(0,0)--(0,u)..(5u,1.2u)..{dir 10}(7u,1.5u){dir 130}..(3u,7.5u)..(0,9u)--cycle;
pa:=(3u,5u)..(5u,6u)..(8u,6u)..(8u,3.8u)..(7.5u,3.4u)..(7u,3u)..cycle;
draw p1;
fill p1 withcolor .8white;
pair a,b;
h=length(p1);
i=5;
numeric c,d;   % low part
c:=0.4*h;     
d:=0.63*h;
a:= point c of p1;
b:= point d of p1;
pickup pencircle scaled 1.5;
%draw a;
%draw b;
path q,r,s,qu;
q:= subpath (c,d) of p1;
ce:=0.8*h;            % up part
de:=0.9*h;
qu:= subpath (ce,de) of p1;
draw q;
draw qu;
pair le;
le:=(-2u,0);
r:= le--(0,0) ;
draw r;
path supp[];
pair au,bu;
au:= point ce of p1;
bu:= point de of p1;
supp1 :=q--au..qu--le--cycle;

numeric cc[],dd[];
c1:= 0.65*h;
d1:= 0.67*h;


fill supp1 withcolor .9red;

real tt = 0.5*u;

for i=1 upto 4:
 for j=0 upto 1:
  draw fullcircle scaled u shifted (2*j*u+0.3u, 1.7*i*u);
  fill fullcircle scaled u shifted (2*j*u+0.3u, 1.7*i*u) withcolor white;
 endfor
endfor


%draw supp1;

%fill pa withcolor .9999blue;
label.rt(btex $\omega$ etex,.5(6u,4.8u))  scaled 2;
label.rt(btex $S$ etex,.5(3.5u,2.5u)) scaled 2;
%label.rt(btex $S$ etex,.5(5.6u,3.5u)) scaled 2;
%label.rt(btex $S$ etex,.5(7.5u,0.8u)) scaled 2;

%path pt; pt:=fullcircle scaled u shifted (5u,3u);
%fill pt withcolor .7white;
%label(btex $B_\rho$ etex,(5u,2.5u)) shifted (.1u,.3u);
%label.top(btex $L$ etex,2/3(6.5u,u)) scaled 1.5 shifted (-.2u,-.5u);








endfig;
end;
