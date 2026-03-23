function testing_sphsymm(typ)

% program for testing the symmetry of the triangles in connection with the combinatorial group
% 
% the input typ is an integer, the size of the group
% only typ>=4 is needed. For the other cases we know by observing angles that the combinatorial group is a subgroup of the symmetry group

% Validated

% G4       OK
% G5       OK
% G6       OK
% G7       OK
% G8       OK
% G9       OK
% G10      OK
% G11      OK
% G12  -- no valid triangles






eval(['load Result_infG' num2str(typ)]);
eval(['MatR = Mat_resultG' num2str(typ) ';']);

% generate all possible symmetries

% G7   (ab)^4

% permutations

ind = find(MatR(:,1)>0);
tot = length(ind);
count =0;
special = 0;
pos = 1;
for nn=ind'
  
  initialOK = 0;  

  v = perms(1:3);

  va = 1;
  vb = 2;
  vc = 3;


  rg = relGroup(va,vb,vc,typ);


  for i=1:6
    valid = zeros(1,size(rg,1));
    for j=1:size(rg,1)
      actg = rg{j};
      valid(j) = DrawSymSphTri0(MatR(nn,:),v(i,actg));
    end
      if sum(valid==1)==length(valid)
        v(i,actg);
        count = count+1;
        initialOK = 1;
        break
      end
  end
  
  if initialOK==0  % if the trick does not work for standard reflections
     va = [2 1 2];
     vb = 2;
     vc = 3;

     rg = relGroup(va,vb,vc,typ);
     %pause

    for i=1:6
      valid = zeros(1,size(rg,1));
      for j=1:size(rg,1)
        actg = rg{j};
        valid(j) = DrawSymSphTri0(MatR(nn,:),v(i,actg));
      end
        if sum(valid==1)==length(valid)
          v(i,actg);
          count = count+1;
          special = special+1;
          initialOK = 1;
          break
        end
    end

  end
 
  fprintf('Pos %d out of %d | Count %d | Delta %d \n',pos,tot,count,pos-count);
  pos = pos+1;


end

fprintf('Total %d | Valid %d | Special steps %d\n',tot,count,special);

function res = relGroup(va,vb,vc,typ)

switch typ
case 4
  res{1} = repmat([va vc],[1 2]);  
  res{2} = repmat([va vb],[1 3]);
case 5
  res{1} = repmat([va vb],[1 3]);
case 6
  res{1} = repmat([va vc],[1 2]);  
  res{2} = repmat([va vb],[1 4]);
case 7
  res{1} = repmat([va vb],[1,4]);
case 8
  res{1} = repmat([va vb],[1 3]);  
  res{2} = repmat([vb vc],[1 3]);
case 9
  res{1} = [va vc vb va vc vb vc va vb vc];
case 10
  res{1} = repmat([va vb],[1 3]);  
  res{2} = repmat([vc vb vc va],[1 2]);
case 11
  res{1} = repmat([vc va],[1 3]);  
  res{2} = repmat([va vb],[1 4]);  
  res{3} = repmat([vb va vb vc],[1 2]);

end
