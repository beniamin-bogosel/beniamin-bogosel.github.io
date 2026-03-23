function result = DrawSymSphTri(vect,vSym,pl)

AllSym = vSym;

result = zeros(size(AllSym,1),1);


vert = vect(7:15);
vert = reshape(vert(:),[],3);
norms = sqrt(sum(vert.^2));
vert = vert./repmat(norms,[3 1]);


%dot(vert(:,1),vert(:,2))
%dot(vert(:,2),vert(:,3))
%dot(vert(:,1),vert(:,3))

ang = pts2sphangNS(vert);
%cot(ang(1))*cot(ang(2))
%cot(ang(2))*cot(ang(3))
%cot(ang(1))*cot(ang(3))


%fprintf('Angles: %f %f %f\n',vect);

if nargin>2
clf
[X,Y,Z] = sphere(100);factt=1.003;
h       = surf(X/factt,Y/factt,Z/factt);set(h,'FaceColor',0.9*ones(1,3),'EdgeColor','none');


axis equal

hold on


draw_local(vert,1);


end

allv = [1 2 3];

vert0 = vert;

for i=1:size(AllSym,1) 

vSym = AllSym(i,:);

len = length(vSym);
pos = 1;

vert = vert0;

vSym = fliplr(vSym);

for v = vSym(:)'
  other = setdiff(allv,v);
  v1 = vert(:,v);
  v2 = vert(:,other(1));
  v3 = vert(:,other(2));
  cf = [norm(v2)^2 dot(v2,v3); dot(v2,v3) norm(v3)^2];
  res = cf\[dot(v1,v2); dot(v1,v3)];
  a = res(1);
  b = res(2); 
  v0 = v1 - a*v2-b*v3;
  vS = v1-2*v0;
  %plot3(vS(1),vS(2),vS(3),'or');
  vert(:,v) = vS;
  if nargin>2
   if pos<len
    draw_local(vert);
   else
    draw_local(vert,1);
    hold off
   end
  end
  pos = pos+1;
end


result(i) = max(abs(vert(:)-vert0(:)))<1e-6;


end % end main loop

function draw_local(vert,second)

plot3(vert(1,:),vert(2,:),vert(3,:),'.','MarkerSize',10);


if nargin<2
 arc_sphere(vert(:,1),vert(:,2),50);
 arc_sphere(vert(:,2),vert(:,3),50);
 arc_sphere(vert(:,1),vert(:,3),50);
else
 arc_sphere(vert(:,1),vert(:,2),50,1);
 arc_sphere(vert(:,2),vert(:,3),50,1);
 arc_sphere(vert(:,1),vert(:,3),50,1);
end




