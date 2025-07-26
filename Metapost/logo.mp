prologues:=3;
verbatimtex
%&latex
\documentclass{minimal}
\begin{document}
etex
beginfig(0);   
% copy from here to use the online previewer
u:=25; % 25 = 25bp = 25 PostScript points = 25/72 in
wi:=10; % width in units u
he:=7; % height in units u
hoehe:=he*u; % height
breite:=wi*u;
%for i=0 upto he:
%draw (0, i*u)--(breite, i*u) withcolor .7white;
%endfor
%for j=0 upto wi:
%draw (j*u, 0)--(j*u, hoehe) withcolor .7white;
%endfor;
path p,q;
p:=(6u,0.5u)--(6u,0)--(0,0)--(0,6u)--(6u,6u)--(6u,5.5u);
pickup pensquare scaled 20;
draw p;
h=length(p);
numeric c,d,detail;
color a,b,co;
a:=(1,0.45,0);
b:=.2white;
co:=.2white;
detail:=500;
for i=1 upto (detail/2):
q := subpath (h*(i-1)/detail,h*i/detail) of p;
draw q withcolor (2*i/detail)[a,b];
endfor;
for i=detail downto (1+detail/2):
q := subpath (h*(i-1)/detail,h*i/detail) of p;
draw q withcolor (2*(detail-i)/detail)[a,b];
endfor;
label ("B",(1/15)*(2.6u,2.7u)) scaled 15 withcolor co ;
label ("2" infont defaultfont scaled 8,(4.9u,4.1u))
withcolor co;
% end copy here for online previewer
endfig;
end;

