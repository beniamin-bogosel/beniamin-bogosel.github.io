function Mat_result = test_Contents_Saved(str,ang0,nb)

% test_Contents_Saved(str,ang0,nb)
% 
% tests the contents of a csv file containing steps on each line
% it should work on files provided on the webpage of the article
%
% needs fmincon from Matlab Optimization toolbox
% or minFunc by Mark Schmidt which can be freely obtained on his webpage
%
% inputs
%
% str: string containing the name of the file
%
% optional inputs:
% ang0: value of an angle we search
% nb: number of angles we want

% choose if you want to compute the eigenvalue or not
% set the following parameter to 1 or 0
computeEigenvalue=0;


if nargin<2
  ang0 = 0.5;
  nb = 2;
end

% read the file
fileID = fopen(str);
C = textscan(fileID,'%s','delimiter','\n');
fclose(fileID);

Gn = size(C{:},1)

% structure of the output: a matrix with 18 columns
% containing info about the angles, the triangle, the exponent
% and the original index

% angles      3 col  1-3
% angles/pi   3 col  4-6
% triangles   9 col  7-15
% eigenvalues 1 col  16
% exponent    1 col  17

Mat_result = zeros(Gn,17);

for nn=1:Gn

V =  C{1}(nn);
ss = V{:};

ss(ss=='(') = '';
ss(ss==')') = '';
v = str2num(ss);
mat = v';
% matrix containing the steps
mat = reshape(mat,3,length(v)/3);

% check if points are in the same half-space
res = test_halfspace3_v3(mat);

% if they are not in the same halfspace, proceed
if res == 0

 o.mat = mat;

 X0 = 0.1*[1,1,1];

% choose 1 to use minFunc (which can be freely obtained)
% choose 2 to use fmincon (Matlab Optimization toolbox)
%       all results were tested with fmincon

 variant = 2; 

 if variant==1
  options.Display = 0;
  options.Method = 'newton';
  options.progTol = 1e-16;
  options.MaxIter = 10000;
  options.optTol  = 1e-16;
  options.HessianModify = 2;

  [U,fx] = minFunc(@func,X0(:),options,o);
  [v,g,H] = func(U,o);
  if (abs(max(g))>1e-5)
     bad = 1;
  else
     bad = 0;
  end

 else
  fun = @(x) func(x,o);
  A = [];
  b = [];
  Aeq = [];
  beq = [];
  lb = zeros(size(X0));
  ub = [];
  options.GradConstr = 'on';
  options.GradObj = 'on';
  options.TolFun = 1e-15;
  options.TolCon = 1e-15;
  options.Algorithm = 'interior-point';
  options.MaxIter = 50;
  options.Hessian = 'on';
  options.HessFcn = @(x,l)calcs_Hess(x,l,o);
  options.Display = 'off';
  options.SubproblemAlgorithm = 'cg';
  nonlcon = @(x) constr(x,o);

  % try solving minimization under critical point constraint
  [U,fx,exitflag] = fmincon(fun,X0,A,b,Aeq,beq,lb,ub,nonlcon,options);
  [v,g,H] = func(U,o);
  % if it does not work, try again without constraint
  if abs(max(g))>1e-5
    [U,fx,exitflag] = fmincon(fun,X0,A,b,Aeq,beq,lb,ub,[],options)
  end

  % test if we found a critical point
  [v,g,H] = func(U,o);
  if (abs(max(g))>1e-5)
     bad = 1;
  else
     bad = 0;
  end
 end

% print the solution
 fprintf('Solutions %f %f %f \n',U);
 smat = sum(mat,2);

% check if we have drift or not
 if(sum(smat==0)==3)
  fprintf('No drift: sum zero \n');
  mat
 end

 if(sum(smat==0)==0)
  fprintf('Drift in each direction: solution different from [1,1,1]\n');
  mat
  U
  pause
 end


% if we have a critical point proceed
 if bad == 0
 [v,g,H] = func(U,o);

 H = double(H);

 a = H(1,2)/sqrt(H(1,1)*H(2,2));
 b = H(1,3)/sqrt(H(1,1)*H(3,3));
 c = H(2,3)/sqrt(H(2,2)*H(3,3));

 AA = [1 a b;
     a 1 c;
     b c 1]

 L = chol(AA,'lower');
 A = L^-1;

 norms = sqrt(sum(A.^2));
 A = A./repmat(norms,[3,1]);
%A

 theta1 = acos(dot(A(:,2),A(:,3))/norm(A(:,2))/norm(A(:,3)));
 theta2 = acos(dot(A(:,1),A(:,3))/norm(A(:,1))/norm(A(:,3)));
 theta3 = acos(dot(A(:,2),A(:,1))/norm(A(:,2))/norm(A(:,1)));

 theta1 = acos(-a);
 theta2 = acos(-b);
 theta3 = acos(-c);

 fprintf('     Angles: %f %f %f\n',theta1,theta2,theta3);
 fprintf('     Exact : %f %f %f\n',theta1/pi,theta2/pi,theta3/pi);

 resAng = [theta1,theta2,theta3]/pi;
 nbAng = sum(abs(resAng-ang0)<1e-5);

 fprintf('number of angles: %d | Desired %d\n',nbAng,nb);

 Mat_result(nn,1:3) = [theta1,theta2,theta3];
 Mat_result(nn,4:6) = [theta1,theta2,theta3]/pi;
 Mat_result(nn,7:15) = A(:)'; % points

% if you wish to compute eigenvalues
% if you only want to check infos on the triangle, then don't compute them since it takes more time
 if (computeEigenvalue==1)
   eG = accurate_bel_Silent(5,A');
 else
   eG = 0;
 end

 Mat_result(nn,16) = eG ;
 Mat_result(nn,17) = sqrt(eG+0.25)+1;


  fprintf('Iteration %d out of %d successful | maxgrad %e \n',nn,Gn,max(abs(g))); 
 else
  fprintf('Gradient non zero: Max grad %f \n',max(abs(g)));
  Mat_result(nn,:) = -1;
 end
 else
  fprintf('Iteration %d out of %d no exponent\n',nn,Gn); 
 end


end

% sort results following the eigenvalues
% and keep original indices
[~,I] = sort(Mat_result(:,17));
Mat_result = Mat_result(I,:);

Mat_result = [Mat_result I];

% objective function
% with gradient and Hessian
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


% critical point constraint
function [c,ceq,CG,CGeq] = constr(X,o);

c = [];
CG = [];

[v,g,H] = func(X,o);

ceq = g;
CGeq = H;

% hessian computation function
function H = calcs_Hess(x,l,o);

[v,g,H] = func(x,o);



