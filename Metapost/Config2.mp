prologues:=3;
verbatimtex
%&latex
\documentclass{minimal}
\begin{document}
etex
beginfig(0);


u:=100; % 25 = 25bp = 25 PostScript points = 25/72 in
wi:=10; % width in units u
he:=7; % height in units u
hoehe:=he*u; % height
breite:=wi*u;

pair cen[];
cen1:=(uu+u,0);

path p[];

p1:= (0,0)--(2u,0)--(2u,u)--(0,u)--cycle;
draw p1;

p2:=(2u,0.25u)--(2u,0.75u)--(1.75u,0.75u)--(1.75u,0.25u)--cycle;

pair A[];
A1:=(1.75u,0.8u);
A2:=(2u,0.8u);

A3:=(2u,0.5u+1/8*u);
A4:=(2u,0.5u-1/8*u);

p3:= A3--A4;

p4:=(0,0)--(u,u);

numeric N;
N:=30;
for i=0 upto N:
  draw p4 scaled 0.03 rotated 180 shifted (0,i/N*u);
endfor



drawdblarrow A1--A2;

draw p2;
fill p2 withcolor 0.8white;

draw p3 withpen pensquare scaled 2bp;

draw (u,1.6u) withpen pencircle withcolor white;

%fill pa withcolor .9999blue;
%label.rt(btex $\Omega$ etex,(2u,2u));
label(btex $\omega$ etex,(1.9u,0.5u));

label.llft(btex $(0,0)$ etex,(0,0));
label.bot(btex $(2,0)$ etex,(2u,0));
label.top(btex $(2,1)$ etex,(2u,u));
label.ulft(btex $(0,1)$ etex,(0,u));

label.rt(btex $(2,0.75)$ etex,(2u,0.75u));
label.rt(btex $(2,0.25)$ etex,(2u,0.25u));

label.rt(btex \textsc{Configuration 2} etex,(0.3u,1.5u));
label.rt(btex No volume force: $f=0$ \hspace{1cm}   etex,(0.3u,1.4u));
label.rt(btex Surface force:   $g_N = (0;-1)$ etex,(0.3u,1.3u));

label.lft(btex $\Gamma_D$ etex,(-0.1u,0.5u));

label.rt(btex $\Gamma_N = \{(x,y) : x = 2$ and $|y-0.5|<1/8\}$ etex,(2u,0.5u));



label.top(btex $0.25$ etex,2*(1.87u,0.8u)) scaled 0.5;

endfig;
end;
