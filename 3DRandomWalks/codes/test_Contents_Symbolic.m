function test_Contents_Symbolic(str,ang0,nb)

%  test_Contents_Symbolic(str,ang0,nb)
%
%  runs the computation on file using symbolic computations
%  to search for the critical point. It works partially
%
%  needs Matlab Symbolic Toolbox
% 
%  the main input is "str": a string containing the name of the csv file
%                           containing the steps

if nargin<2
  ang0 = 0.5;
  nb   = 2;
end

fileID = fopen(str);
C = textscan(fileID,'%s','delimiter','\n');
fclose(fileID);

Gn = size(C{:},1)

% angles      3 col  1-3
% angles/pi   3 col  4-6
% triangles   9 col  7-15
% eigenvalues 1 col  16
% exponent    1 col  17

Mat_result = zeros(Gn,17);

% symbolic variables
  syms x y z 
  vv = sym('a',[1,3]);

for nn=1:Gn

try
V =  C{1}(nn);
ss = V{:};

ss(ss=='(') = '';
ss(ss==')') = '';
v = str2num(ss);
mat = v';
mat = reshape(mat,3,length(v)/3);


res = test_halfspace3_v3(mat);

if res == 0
  fprintf('It %d out of %d ... computing |\n',nn,Gn); 
  o.mat = mat;

  % only one root different than 1

  ff = 0;
  for i=1:size(mat,2)
    ff = ff+x^mat(1,i)*y^mat(2,i)*z^mat(3,i);
  end
  ff;
  gg = jacobian(ff,[x,y,z]);
  hh = jacobian(gg,[x,y,z]);

  drift = sum(mat,2);
  val = subs(gg,{x,y,z},{1,1,1});
  if or(sum(drift==0)==3,val==0)
    x0 = 1;
    y0 = 1;
    z0 = 1;
  else
  Sols = solve(gg==0,'Real',true);
  indx = find(Sols.x>0);
  indy = find(Sols.y>0);
  indz = find(Sols.z>0);
  
  indx = intersect(indx,indy);
  ind  = intersect(indx,indz);

  x0   = Sols.x(ind);
  y0   = Sols.y(ind); 
  z0   = Sols.z(ind);
  end
  gradver = subs(gg,{x,y,z},{x0,y0,z0});
  h0 = subs(hh,{x,y,z},{x0,y0,z0}) ;
  H = h0;

  aa = H(1,2)/sqrt(H(1,1)*H(2,2));
  bb = H(1,3)/sqrt(H(1,1)*H(3,3));
  cc = H(2,3)/sqrt(H(2,2)*H(3,3));

  %[aa,bb,cc]
  %double([aa,bb,cc])
  
  %mat
  acos(-aa)
  theta1 = double(acos(-aa))
  theta2 = double(acos(-bb));
  theta3 = double(acos(-cc));
  fprintf('Sym  Angles: %f %f %f\n',theta1,theta2,theta3);
  fprintf('Sym  Exact : %f %f %f\n',theta1/pi,theta2/pi,theta3/pi);

%pause

resAng = [theta1,theta2,theta3]/pi;
nbAng = sum(abs(resAng-ang0)<1e-5);

fprintf('number of angles: %d | Desired %d\n',nbAng,nb);




%eG = accurate_bel_Silent(5,A');
eG = 0;



else
  fprintf('Iteration %d out of %d no exponent\n',nn,Gn); 
end

catch
 fprintf('Failed to solve the symbolic equations\n');
end


end

[~,I] = sort(Mat_result(:,17));
Mat_result = Mat_result(I,:);

Mat_result = [Mat_result I];

function [v,g,H] = func(X,o);

evs1 = X(1).^o.mat(1,:);
evs2 = X(2).^o.mat(2,:);
evs3 = X(3).^o.mat(3,:);

g1 = o.mat(1,:).*X(1).^(o.mat(1,:)-1);
g2 = o.mat(2,:).*X(2).^(o.mat(2,:)-1);
g3 = o.mat(3,:).*X(3).^(o.mat(3,:)-1);

v = sum(evs1.*evs2.*evs3);
g = [sum(g1.*evs2.*evs3);sum(evs1.*g2.*evs3);sum(evs1.*evs2.*g3)];

% to add hessian

g11 = o.mat(1,:).*(o.mat(1,:)-1).*X(1).^(o.mat(1,:)-2);
g22 = o.mat(2,:).*(o.mat(2,:)-1).*X(2).^(o.mat(2,:)-2);
g33 = o.mat(3,:).*(o.mat(3,:)-1).*X(3).^(o.mat(3,:)-2);

H11 = sum(g11.*evs2.*evs3);
H22 = sum(evs1.*g22.*evs3);
H33 = sum(evs1.*evs2.*g33);

H12 = sum(g1.*g2.*evs3);
H13 = sum(g1.*evs2.*g3);
H23 = sum(evs1.*g2.*g3);

H = [H11 H12 H13;
     H12 H22 H23;
     H13 H23 H33];

function [c,ceq,CG,CGeq] = constr(X,o);

c = [];
CG = [];

[v,g,H] = func(X,o);

ceq = g;
CGeq = H;


function H = calcs_Hess(x,l,o);

[v,g,H] = func(x,o);



