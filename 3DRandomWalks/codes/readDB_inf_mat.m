function readDB_inf_mat(mat)

% readDB_inf_mat(mat)
%
% this function computes the triangle and the eigenvalue given a 3D model
%
% mat is a 3xN matrix containing the steps of the 3D model on colums




tic


Mat_result = zeros(1,17);


o.mat = mat;

res = test_halfspace3_v3(mat);

if res==1
  error('All points in a halfspace!! The program cannot continue');
end

X0 = 0.9*[1,1,1]';
epst = 1e-6;
dir = rand(size(X0));

[v,g] = func(X0,o);
v1 = func(X0+epst*dir,o);
v2 = func(X0-epst*dir,o);



options.Display = 0;
options.Method = 'newton';%'lbfgs';
options.progTol = 1e-16;
options.MaxIter = 10000;
options.optTol  = 1e-16;
options.cgSolve = 2;

[U,fx] = minFunc(@func,X0,options,o);

U
[x,g,H] = func(U,o);
g

H = double(H);

a = H(1,2)/sqrt(H(1,1)*H(2,2));
b = H(1,3)/sqrt(H(1,1)*H(3,3));
c = H(2,3)/sqrt(H(2,2)*H(3,3));

AA = [1 a b;
     a 1 c;
     b c 1];

L = chol(AA,'lower');
A = L^-1;

norms = sqrt(sum(A.^2));
A = A./repmat(norms,[3,1]);

sum(A.^2)
A

theta1 = acos(dot(A(:,2),A(:,3))/norm(A(:,2))/norm(A(:,3)));
theta2 = acos(dot(A(:,1),A(:,3))/norm(A(:,1))/norm(A(:,3)));
theta3 = acos(dot(A(:,2),A(:,1))/norm(A(:,2))/norm(A(:,1)));

ang = convert_pts_sphang([zeros(1,6) A(:)']);
ang(1:6)

theta1 = acos(-a);
theta2 = acos(-b);
theta3 = acos(-c);

fprintf('Angles: %f %f %f\n',theta1,theta2,theta3);
fprintf('Exact : %f %f %f\n',theta1/pi,theta2/pi,theta3/pi);


eG = accurate_bel_Silent(7,A',1)




toc

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


