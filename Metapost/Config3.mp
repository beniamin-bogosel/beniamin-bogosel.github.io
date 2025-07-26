prologues:=3;
verbatimtex
%&latex
\documentclass{minimal}
\begin{document}
etex
beginfig(0);


u:=70; % 25 = 25bp = 25 PostScript points = 25/72 in
wi:=10; % width in units u
he:=7; % height in units u
hoehe:=he*u; % height
breite:=wi*u;

pair cen[];
cen1:=(uu+u,0);

path p[];

p1:= (0,0)--(4u,0)--(4u,u)--(0,u)--cycle;
draw p1;

p2:=(2u-1/6u,0.5u)--(2u+1/6u,0.5u)--(2u+1/6u,0.5u+1/3u)--(2u-1/6u,0.5u+1/3u)--cycle;

pair A[];
A1:=(2u-1/6u,0.86u);
A2:=(2u+1/6u,0.86u);

A3:=(2u,0.5u+1/8*u);
A4:=(2u,0.5u-1/8*u);

p3:= A3--A4;

p4:=(0,0)--(u,u);

numeric N;
N:=30;
for i=0 upto N:
  draw p4 scaled 0.03 rotated 180 shifted (0,i/N*u);
endfor

for i=0 upto N:
  draw p4 scaled 0.03 shifted (4u,i/N*u);
endfor



drawdblarrow A1--A2;

A5:=(1.8u,0.5u);
A6:=(1.8u,0.5u+1/3u);

drawdblarrow A5--A6;

draw p2;
fill p2 withcolor 0.8white;


draw (u,1.6u) withpen pencircle withcolor white;

%fill pa withcolor .9999blue;
%label.rt(btex $\Omega$ etex,(2u,2u));
label(btex $\omega$ etex,(2u,0.69u));

label.llft(btex $(0,0)$ etex,(0,0));
label.bot(btex $(4,0)$ etex,(4u,0));
label.top(btex $(4,1)$ etex,(4u,u));
label.ulft(btex $(0,1)$ etex,(0,u));


label.rt(btex \textsc{Configuration 3} etex,(0.3u,1.5u));
label.rt(btex Volume force: $f=(0;-1)$ and $\Gamma_N = \emptyset$   etex,(0.3u,1.35u));

label.lft(btex $\Gamma_D$ etex,(-0.1u,0.5u));

label.rt(btex $\Gamma_D$ etex,(4.1u,0.5u));



label.top(btex $1/3$ etex,1/0.7*(2u,0.86u)) scaled 0.7;
label.lft(btex $1/3$ etex,1/0.7*(1.8u,0.675u)) scaled 0.7;

endfig;
end;
