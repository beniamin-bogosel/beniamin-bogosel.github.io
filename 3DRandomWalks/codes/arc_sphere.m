function [pts,theta,orientation] = arc_sphere(p1,p2,n,fourth);

p1 = p1(:)/norm(p1);
p2 = p2(:)/norm(p2);

normal = cross(p1,p2);

theta = acos(dot(p1,p2)/(norm(p1)*norm(p2)));

orientation = 1;
if not(norm(p2-rota(theta,normal)*p1)<1e-6)
   normal = -normal;
   orientation =-1;
end

A     = rota(theta/n,normal);

pts = zeros(3,n+1);
pts(:,1) = p1;
p   = p1;

for i=2:n+1
   p = A*p;
   pts(:,i) = p;
end

if nargin<4
  plot3(pts(1,:),pts(2,:),pts(3,:),'k','LineWidth',1);
else
  plot3(pts(1,:),pts(2,:),pts(3,:),'r','LineWidth',2);
end
axis equal
