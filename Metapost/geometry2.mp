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

pair cen[];
cen1:=(uu+u,0);

path p[];

p1:= (0,0)--(10u,0)--(10u,5u)--(5u,5u)--(5u,10u)--(0,10u)--cycle;
draw p1;

draw fullcircle scaled u shifted (2u,2u);
fill fullcircle scaled u shifted (2u,2u) withcolor 0.8white;


%draw supp1;

%fill pa withcolor .9999blue;
label.rt(btex $\Omega$ etex,.5(7u,2u))  scaled 2;
label(btex $\omega$ etex,.5(2u,2u)) scaled 2;

label.llft(btex $(0,0)$ etex,(0,0));
label.bot(btex $(10,0)$ etex,(10u,0));
label.top(btex $(10,5)$ etex,(10u,5u));
label.urt(btex $(5,5)$ etex,(5u,5u));
label.urt(btex $(5,10)$ etex,(5u,10u));
label.ulft(btex $(0,10)$ etex,(0,10u));




endfig;
end;
