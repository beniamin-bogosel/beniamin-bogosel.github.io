prologues:=3;

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
for i=0 upto he:
%draw (0, i*u)--(breite, i*u) withcolor .7white;
endfor
for j=0 upto wi:
%draw (j*u, 0)--(j*u, hoehe) withcolor .7white;
endfor;
path p[],pa;
%p1:=(0,0)..(5u,0)..(6u,1.5u)..(5u,3u)..(0,3u)..(-u,1.5u)..cycle;

numeric tt;
numeric pi; pi := 180;

pair A[];
numeric n;
n:=7;

pickup pencircle scaled 0.1bp;

A[0] = (u,0);
numeric theta;
theta = 2*pi/n;
for i=1 upto n:
   A[i]:= A[i-1] rotated theta;
endfor;

pair O;
O = (0,0);
p1:= O--A[0]--A[1]--cycle;
p2:= O--A[2]--A[1]--cycle;

color dred ; dred := (.625,0,0) ;


%for i=1 step -0.01 until 0:
%   fill p1 shifted -A[1] scaled (i) shifted A[1] withcolor (i*white)+(1-i)*dred ;
%%   fill p2 shifted -A[1] scaled (i) shifted A[1] withcolor (i*white)+(1-i)*dred ;
%endfor

%draw O--A[0];
%draw O--A[1];
%draw O--A[2];
%draw O--A[3];
%draw O--A[4];
%for i=1 upto n:
%   draw A[i-1]--A[i];
%endfor 

path tri,tris,trir;
numeric h,meshp;

meshp:=5;
h:=1/meshp;

for i=0 upto n-1:
	numeric j;
	j = (i+1) mod n;
	tri:= O--A[i]--A[j]--cycle;

	tris:=tri scaled h;
	pair P[];
	P[0]:= h*A[i];
	P[1]:= h*A[j];
	trir:= tris reflectedabout (P[0],P[1]);
	draw tris;
	draw trir;
	for k=meshp-1 downto 0:
		for s=0 upto k:
			draw tris shifted (s*h*A[i]);
		endfor
		tris:=tris shifted P[1];
	endfor
	draw tri withpen pencircle scaled 0.4pt withcolor 0.5white;
endfor




endfig;
end;
